"""Private org state in Cloudflare R2: the enterprise counterpart of the public hub.

Everything an org owns lives under one prefix of the Tree60 bucket, which the Tree60
Worker serves only to that org's signed-in members::

    orgs/{org}/specs.yaml                        what to run (private: names stations)
    orgs/{org}/state/{pairs,obs,models}/...      the calibration state, as locally
    orgs/{org}/site/index.json                   the org's spec list with headline skill
    orgs/{org}/site/stations/{slug}/*.json       published forecasts and verification
    orgs/{org}/inbox/...                         observations pushed by the org's loggers
    orgs/{org}/uploads/...                       CSV files the org has uploaded

A runner pulls the prefix, runs ``wxfuser org-run`` against the local copy, and pushes
it back. The push refuses unless this root was pulled first. That is the same guard the
public hub sync learned the hard way: a runner that never restored would otherwise
upload shallow rebuilt archives over an org's real history.

Credentials come from the environment (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY; R2_BUCKET defaults to tree60-data). Only the private enterprise
runner holds them. The public jobs never do, so a public job cannot reach org data
even by mistake.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

PULL_MARKER = ".r2-pulled"
DEFAULT_BUCKET = "tree60-data"
CONTENT_TYPES = {".json": "application/json", ".yaml": "text/yaml", ".csv": "text/csv",
                 ".jsonl": "application/x-ndjson", ".parquet": "application/vnd.apache.parquet"}


class R2Error(RuntimeError):
    pass


def org_prefix(org: str) -> str:
    from wxfuser.spec import ORG_RE

    if not ORG_RE.match(org):
        raise R2Error(f"org id {org!r} is not a valid org id")
    return f"orgs/{org}/"


def client():
    import boto3

    account = os.environ.get("R2_ACCOUNT_ID")
    endpoint = os.environ.get("R2_ENDPOINT") or (
        f"https://{account}.r2.cloudflarestorage.com" if account else None
    )
    if not endpoint:
        raise R2Error("R2_ACCOUNT_ID (or R2_ENDPOINT) is not set")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def bucket() -> str:
    return os.environ.get("R2_BUCKET") or DEFAULT_BUCKET


def _list(s3, bkt: str, prefix: str) -> dict[str, str]:
    """Key -> ETag for every object under a prefix."""
    out: dict[str, str] = {}
    token = None
    while True:
        kw = {"Bucket": bkt, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            out[obj["Key"]] = obj.get("ETag", "").strip('"')
        if not resp.get("IsTruncated"):
            return out
        token = resp.get("NextContinuationToken")


def pull(org: str, root: str | Path, *, s3=None) -> int:
    """Mirror ``orgs/{org}/`` into ``root``. Returns the number of files fetched.

    An empty prefix is a genuine cold start and still counts as pulled: there is no
    history to protect.
    """
    s3 = s3 or client()
    prefix = org_prefix(org)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    objects = _list(s3, bucket(), prefix)
    for key in objects:
        dest = root / key[len(prefix):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket(), key, str(dest))
    (root / PULL_MARKER).write_text(f"{bucket()}/{prefix}\n")
    return len(objects)


def push(org: str, root: str | Path, *, s3=None, delete_missing: bool = False) -> int:
    """Upload what changed under ``root`` to ``orgs/{org}/``. Returns files uploaded.

    Unchanged files are skipped by comparing the local MD5 with the object's ETag, which
    for a single-part upload is exactly that. ``delete_missing`` removes remote objects
    with no local counterpart. It is off by default, because a partial local copy
    must never be able to delete an org's history.
    """
    s3 = s3 or client()
    prefix = org_prefix(org)
    root = Path(root)
    marker = root / PULL_MARKER
    if not marker.exists() or marker.read_text().strip() != f"{bucket()}/{prefix}":
        raise R2Error(f"refusing to push {root}: it was not pulled from {bucket()}/{prefix}")

    remote = _list(s3, bucket(), prefix)
    uploaded = 0
    local_keys = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == PULL_MARKER or rel.endswith((".DS_Store", ".pyc")):
            continue
        key = prefix + rel
        local_keys.add(key)
        data = path.read_bytes()
        if remote.get(key) == hashlib.md5(data).hexdigest():
            continue
        s3.put_object(Bucket=bucket(), Key=key, Body=data,
                      ContentType=CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        uploaded += 1
    if delete_missing:
        for key in set(remote) - local_keys:
            s3.delete_object(Bucket=bucket(), Key=key)
    return uploaded
