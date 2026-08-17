"""Move station state between the runner and the Hugging Face dataset repo.

The paired archives, fitted champions, and verification history grow without bound and
change on every run. Committing them to git would bury the actual history — the code and
the enrollment decisions — under an unreadable stream of binary diffs, so they live on a
HF dataset repo instead and the git repo stays reviewable.

Failure here is deliberately non-fatal at the workflow level: without prior state a
station rebuilds its archive from the upstream APIs, which is slower but correct.
"""
from __future__ import annotations

import os
from pathlib import Path

from wxfuser.config import hf_state_repo

STATE_DIR = Path(os.environ.get("WXFUSER_STATE_DIR", "state"))


def _api():
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    return HfApi(token=token), token


def download() -> int:
    from huggingface_hub import snapshot_download

    repo = hf_state_repo()
    _, token = _api()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(STATE_DIR),
        token=token,  # public repos read without one
    )
    n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
    print(f"restored {n} state files from {repo}")
    return 0


def _shard_patterns(shard: int, of: int) -> list[str] | None:
    """Upload globs covering only the stations this worker owns.

    Without this, sharded runs corrupt each other. Every worker restores the whole state
    at startup, so it holds a snapshot of the other shards' stations from before they ran.
    Uploading the whole folder then pushes those stale copies back, and a worker that
    finishes late silently reverts the work of one that finished early — deep history
    replaced by the shallow archive it started from.
    """
    try:
        from wxfuser.data.registry import load_registry
    except Exception:  # noqa: BLE001
        return None
    stations = load_registry()
    if not stations or of <= 1:
        return None
    slugs = [s.slug for i, s in enumerate(stations) if i % of == shard]
    patterns: list[str] = []
    for slug in slugs:
        patterns += [
            f"pairs/{slug}.parquet",
            f"obs/{slug}.parquet",
            f"models/{slug}.json",
            f"models/{slug}/**",
        ]
    return patterns


def upload(shard: int | None = None, of: int | None = None) -> int:
    repo = hf_state_repo()
    api, token = _api()
    if not token:
        print("HF_TOKEN not set; skipping state upload")
        return 0
    if not STATE_DIR.exists():
        print("no local state to upload")
        return 0

    allow = _shard_patterns(shard, of) if (shard is not None and of) else None
    if allow:
        print(f"uploading only shard {shard + 1}/{of}: {len(allow) // 4} stations")

    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(STATE_DIR),
        repo_id=repo,
        repo_type="dataset",
        token=token,
        commit_message=(
            f"update station state (shard {shard + 1}/{of})" if allow else "update station state"
        ),
        allow_patterns=allow,
        # Filesystem and interpreter debris would otherwise be published alongside the
        # data and downloaded by every subsequent run.
        ignore_patterns=[".DS_Store", "**/.DS_Store", "__pycache__/**", "*.pyc", "*.tmp"],
    )
    print(f"uploaded {STATE_DIR} to {repo}")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="download", choices=["download", "upload"])
    ap.add_argument("--shard", type=int, help="0-based worker index; scopes an upload")
    ap.add_argument("--of", type=int, help="total workers")
    args = ap.parse_args()

    if args.action == "download":
        return download()
    if args.action == "upload":
        return upload(args.shard, args.of)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
