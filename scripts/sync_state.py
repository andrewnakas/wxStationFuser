"""Move station state between the runner and the Hugging Face dataset repo.

The paired archives, fitted champions, and verification history grow without bound and
change on every run. Committing them to git would bury the actual history — the code and
the enrollment decisions — under an unreadable stream of binary diffs, so they live on a
HF dataset repo instead and the git repo stays reviewable.

Failure here is deliberately non-fatal at the workflow level: without prior state a
station rebuilds its archive from the upstream APIs, which is slower but correct.
"""
from __future__ import annotations

import functools
import os
import re
import time
from pathlib import Path

from wxfuser.config import hf_state_repo

STATE_DIR = Path(os.environ.get("WXFUSER_STATE_DIR", "state"))


def _api(*, required: bool = False):
    """The hub client and the token it resolved, if any.

    Read paths pass ``required=False``: the state repo is public, so a restore must still
    work on a runner with no secret — a fork's CI, or a local checkout.
    """
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if required and not token:
        raise RuntimeError("HF_TOKEN is not set; this operation needs write access")
    return HfApi(token=token), token


# Written when a restore succeeds. Its absence tells the upload step that this runner
# never saw the existing state, and therefore must not overwrite it.
RESTORE_MARKER = ".restored"


# The hub allows 1000 API requests per five minutes across the account, and a sharded
# fleet checkpointing its way through thousands of stations will reach that. A 429 is not
# a failure, it is a queue — but only if something waits.
RATE_LIMIT_ATTEMPTS = 6
RATE_LIMIT_FALLBACK_S = 30


def _retry_after_seconds(exc: Exception) -> float | None:
    """How long the hub asked us to wait, from the header or from the message."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("Retry-After") or header.get("retry-after")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    match = re.search(r"[Rr]etry after (\d+(?:\.\d+)?) second", str(exc))
    return float(match.group(1)) if match else None


def _is_rate_limited(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()


def with_rate_limit_retry(call, what: str, attempts: int = RATE_LIMIT_ATTEMPTS):
    """Run a hub call, waiting out rate limits rather than dying on them.

    This is the difference between a fleet that publishes and one that does not. Every
    scheduled refresh between 17 and 22 August failed here: six shards restoring,
    checkpointing and uploading exhausted the quota, the 429 propagated out of the first
    call that met it, and the run banked nothing — for five days, with the site frozen at
    whatever the last success left behind.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limited(exc) or attempt == attempts:
                raise
            wait = _retry_after_seconds(exc) or RATE_LIMIT_FALLBACK_S * attempt
            # A little over what was asked for: every shard is being told the same number
            # at the same moment, and returning together simply re-exhausts the quota.
            wait += 5.0 * attempt
            print(f"  {what}: rate limited, waiting {wait:.0f}s "
                  f"(attempt {attempt}/{attempts})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{what}: exhausted rate-limit retries")


@functools.lru_cache(maxsize=1)
def _repo_exists() -> bool:
    """Whether the state repo is already there.

    Only a genuine absence answers False. Every other failure propagates, because the
    upload guard reads this as "nothing to protect" — so treating a timeout or a bad
    token as absence would license the shallow-overwrite this is here to prevent. A rate
    limit is neither: it is waited out rather than answered.

    Memoised because the answer cannot change within a run, and the call was being made
    on every checkpoint upload.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    api, token = _api()

    def probe():
        try:
            api.repo_info(repo_id=hf_state_repo(), repo_type="dataset", token=token)
            return True
        except RepositoryNotFoundError:
            return False

    return with_rate_limit_retry(probe, "repo_info")


def download(shard: int | None = None, of: int | None = None, paths: str | None = None) -> int:
    """Restore state, and fail loudly if state exists but could not be fetched.

    Swallowing a failed restore is what turns a transient network problem into data loss:
    the run continues with an empty state directory, rebuilds a shallow ten-day archive
    for every station, and uploads that over the deep history it never managed to read.
    A missing repository is fine — that is a genuine cold start.

    A worker restores only the stations it owns. Pulling the whole archive on every shard
    made twenty runners fetch the same hundred-plus megabytes at once, which Hugging Face
    answers with 429s — so the fleet failed on the download step, correctly refusing to
    continue, and the run produced nothing. The split is the same id hash the work uses.
    """
    from huggingface_hub import snapshot_download

    repo = hf_state_repo()
    _, token = _api()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not _repo_exists():
        print(f"{repo} does not exist yet; starting cold")
        return 0

    allow = None
    if paths:
        # The publish job wants the catalogue and the index baseline, not a hundred
        # megabytes of paired archives it will never open.
        allow = [f"{p.strip().rstrip('/')}/**" for p in paths.split(",") if p.strip()]
        print(f"restoring only: {', '.join(allow)}")
    elif shard is not None and of and of > 1:
        allow = _shard_patterns(shard, of)
        if allow:
            # The catalogue and the published index are shared, not per-station, and the
            # jobs that stage them need them whatever slice they hold.
            allow = allow + ["catalogue/**", "site/**"]
            print(f"restoring shard {shard + 1}/{of} only: {len(allow) // 4} stations")

    try:
        path = with_rate_limit_retry(
            lambda: snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(STATE_DIR),
                token=token,  # public repos read without one
                allow_patterns=allow,
            ),
            "snapshot_download",
        )
    except Exception as exc:  # noqa: BLE001
        if not _is_size_mismatch(exc):
            print(f"ERROR: {repo} exists but could not be restored ({exc}).")
            print("Refusing to continue: proceeding would overwrite it with shallow archives.")
            return 1
        print(f"WARN: bulk restore rejected a file ({exc}); restoring file by file, "
              "verifying content by hash instead of recorded size", flush=True)
        try:
            path = _download_verified(repo, token, allow)
        except Exception as exc2:  # noqa: BLE001
            print(f"ERROR: {repo} exists but could not be restored ({exc2}).")
            print("Refusing to continue: proceeding would overwrite it with shallow archives.")
            return 1

    n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
    # Record what was restored, not just that something was. Local state after a scoped
    # restore holds one shard's stations; an unscoped upload of it would look like a
    # complete snapshot, which is the failure the marker exists to prevent.
    scope = "full" if allow is None else (f"{shard}/{of}" if not paths else f"paths:{paths}")
    (STATE_DIR / RESTORE_MARKER).write_text(f"{repo}\n{scope}\n")
    print(f"restored {n} state files from {repo} ({scope})")
    return 0


def _is_size_mismatch(exc: Exception) -> bool:
    text = str(exc).lower()
    return "size mismatch" in text or ("consistency check failed" in text and "size" in text)


def _fetch_bytes(url: str, token: str | None) -> bytes:
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


def _content_matches(entry, data: bytes) -> bool:
    """Whether downloaded bytes are the file the tree lists, judged by content hash.

    LFS files carry the SHA-256 of their content; plain git files carry a blob id. The
    recorded *size* is deliberately not consulted. On 25 September the hub listed
    obs/750_NV_SNTL.parquet at 11,532 bytes while serving 11,625 bytes whose SHA-256
    matched its LFS pointer exactly. The file was intact and only the metadata was
    wrong, yet the client's size check failed every shard restore that touched it.
    """
    import hashlib

    lfs = getattr(entry, "lfs", None)
    if lfs is not None:
        want = getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)
        return hashlib.sha256(data).hexdigest() == want
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return blob == getattr(entry, "blob_id", None)


def _download_verified(repo: str, token: str | None, allow: list[str] | None) -> str:
    """Restore file by file at one pinned revision, checking each file's content hash.

    The fallback for when the bulk restore rejects a file on its recorded size. It is
    slower than a snapshot, but it restores exactly the same set of files and fails
    just as loudly on anything whose content is actually wrong.
    """
    import fnmatch
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_url

    api, _ = _api()
    revision = with_rate_limit_retry(
        lambda: api.repo_info(repo_id=repo, repo_type="dataset", token=token).sha,
        "repo_info",
    )
    tree = with_rate_limit_retry(
        lambda: list(api.list_repo_tree(repo, repo_type="dataset", recursive=True,
                                        revision=revision, token=token)),
        "list_repo_tree",
    )
    files = [e for e in tree if getattr(e, "blob_id", None) is not None]
    if allow is not None:
        files = [e for e in files if any(fnmatch.fnmatch(e.path, pat) for pat in allow)]

    repaired: list[str] = []

    def one(entry) -> None:
        url = hf_hub_url(repo, entry.path, repo_type="dataset", revision=revision)
        data = with_rate_limit_retry(lambda: _fetch_bytes(url, token), f"fetch {entry.path}")
        if not _content_matches(entry, data):
            raise RuntimeError(f"{entry.path}: content does not match its recorded hash")
        if getattr(entry, "size", None) not in (None, len(data)):
            repaired.append(entry.path)
        dest = STATE_DIR / entry.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, files))
    for path in repaired:
        print(f"  restored {path}: content verified, hub size metadata wrong", flush=True)
    print(f"verified restore: {len(files)} files, {len(repaired)} with wrong size metadata")
    return str(STATE_DIR)


def _restored_scope() -> str | None:
    """Which slice this runner restored: "full", "shard/of", or None if it never did."""
    marker = STATE_DIR / RESTORE_MARKER
    if not marker.exists():
        return None
    lines = marker.read_text().splitlines()
    return lines[1].strip() if len(lines) > 1 else "full"


def _shard_patterns(shard: int, of: int) -> list[str] | None:
    """The globs covering exactly the stations this worker owns.

    Used for both halves of the round trip. On the way in it keeps a runner from pulling
    an archive twenty times larger than it needs, which is what drew rate limits when the
    whole fleet started at once. On the way out it stops shards corrupting each other: a
    worker that uploaded the whole folder would push back its startup snapshot of the
    other shards' stations, so one finishing late would silently revert one that finished
    early — deep history replaced by the shallow archive it began with.
    """
    try:
        from wxfuser.data.registry import load_registry
    except Exception:  # noqa: BLE001
        return None
    from wxfuser.cli import shard_of

    stations = load_registry()
    if not stations or of <= 1:
        return None
    # Same id-hash split the work uses, so the paths uploaded are exactly the stations
    # this worker processed even if the registry changed size in between.
    slugs = [s.slug for s in stations if shard_of(s.id, of) == shard]
    patterns: list[str] = []
    for slug in slugs:
        patterns += [
            f"pairs/{slug}.parquet",
            f"obs/{slug}.parquet",
            f"models/{slug}.json",
            f"models/{slug}/**",
        ]
    return patterns


def upload(shard: int | None = None, of: int | None = None, paths: str | None = None) -> int:
    repo = hf_state_repo()
    api, token = _api()
    if not token:
        print("HF_TOKEN not set; skipping state upload")
        return 0
    if not STATE_DIR.exists():
        print("no local state to upload")
        return 0

    # If the hub already holds state that this runner never restored, anything local is a
    # partial rebuild and publishing it would destroy the real thing.
    #
    # Order matters and used not to. Written the other way round, this asked the hub
    # whether the repository exists before checking whether the answer could change
    # anything — so every upload, including every mid-run checkpoint, spent an API call it
    # did not need. Ninety of them per refresh, into a quota of a thousand per five
    # minutes, shared with the restores.
    scope = _restored_scope()
    if scope is None and _repo_exists():
        print("refusing to upload: hub state exists but was never restored here")
        return 1

    # An upload may never claim more than the restore covered. A shard that restored its
    # own slice holds nothing of the others', so publishing unscoped would present one
    # twentieth of the archive as the whole of it.
    if scope and scope != "full" and not paths:
        want = f"{shard}/{of}" if shard is not None and of else None
        if want != scope:
            print(f"refusing to upload {want or 'everything'}: this runner restored only "
                  f"shard {scope}, so it cannot vouch for anything else")
            return 1

    # Each writer publishes only what it owns: a shard its stations, the catalogue job
    # its catalogue. Whole-folder uploads from concurrent jobs collide on the underlying
    # ref, which is what took the catalogue build down.
    if paths:
        allow = [f"{p.strip().rstrip('/')}/**" for p in paths.split(",") if p.strip()]
        print(f"uploading only: {', '.join(allow)}")
    else:
        allow = _shard_patterns(shard, of) if (shard is not None and of) else None
    if allow and not paths:
        print(f"uploading only shard {shard + 1}/{of}: {len(allow) // 4} stations")

    with_rate_limit_retry(
        lambda: api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True,
                                token=token),
        "create_repo",
    )
    def do_upload():
        return api.upload_folder(
            folder_path=str(STATE_DIR),
            repo_id=repo,
            repo_type="dataset",
            token=token,
            commit_message=(
                f"update station state (shard {shard + 1}/{of})"
                if allow
                else "update station state"
            ),
            allow_patterns=allow,
            # Filesystem and interpreter debris would otherwise be published alongside
            # the data and downloaded by every subsequent run.
            ignore_patterns=[
                ".DS_Store", "**/.DS_Store", "__pycache__/**", "*.pyc", "*.tmp",
                RESTORE_MARKER,
            ],
        )

    with_rate_limit_retry(do_upload, "upload_folder")
    print(f"uploaded {STATE_DIR} to {repo}")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="download", choices=["download", "upload"])
    ap.add_argument("--shard", type=int, help="0-based worker index; scopes an upload")
    ap.add_argument("--of", type=int, help="total workers")
    ap.add_argument("--paths", help="comma-separated state subdirectories to upload")
    args = ap.parse_args()

    if args.action == "download":
        return download(args.shard, args.of, args.paths)
    if args.action == "upload":
        return upload(args.shard, args.of, args.paths)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
