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


# Written when a restore succeeds. Its absence tells the upload step that this runner
# never saw the existing state, and therefore must not overwrite it.
RESTORE_MARKER = ".restored"


def _repo_exists() -> bool:

    api, token = _api(required=False)
    try:
        api.repo_info(repo_id=hf_state_repo(), repo_type="dataset", token=token)
        return True
    except Exception:  # noqa: BLE001
        return False


def download() -> int:
    """Restore state, and fail loudly if state exists but could not be fetched.

    Swallowing a failed restore is what turns a transient network problem into data loss:
    the run continues with an empty state directory, rebuilds a shallow ten-day archive
    for every station, and uploads that over the deep history it never managed to read.
    A missing repository is fine — that is a genuine cold start.
    """
    from huggingface_hub import snapshot_download

    repo = hf_state_repo()
    _, token = _api(required=False)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not _repo_exists():
        print(f"{repo} does not exist yet; starting cold")
        return 0

    try:
        path = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(STATE_DIR),
            token=token,  # public repos read without one
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {repo} exists but could not be restored ({exc}).")
        print("Refusing to continue: proceeding would overwrite it with shallow archives.")
        return 1

    n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
    (STATE_DIR / RESTORE_MARKER).write_text(repo)
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
    if _repo_exists() and not (STATE_DIR / RESTORE_MARKER).exists():
        print("refusing to upload: hub state exists but was never restored here")
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
        ignore_patterns=[
            ".DS_Store", "**/.DS_Store", "__pycache__/**", "*.pyc", "*.tmp",
            RESTORE_MARKER,
        ],
    )
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
        return download()
    if args.action == "upload":
        return upload(args.shard, args.of, args.paths)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
