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
import sys
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


def upload() -> int:
    repo = hf_state_repo()
    api, token = _api()
    if not token:
        print("HF_TOKEN not set; skipping state upload")
        return 0
    if not STATE_DIR.exists():
        print("no local state to upload")
        return 0

    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(STATE_DIR),
        repo_id=repo,
        repo_type="dataset",
        token=token,
        commit_message="update station state",
        # Filesystem and interpreter debris would otherwise be published alongside the
        # data and downloaded by every subsequent run.
        ignore_patterns=[".DS_Store", "**/.DS_Store", "__pycache__/**", "*.pyc", "*.tmp"],
    )
    print(f"uploaded {STATE_DIR} to {repo}")
    return 0


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "download"
    if action == "download":
        return download()
    if action == "upload":
        return upload()
    print(f"unknown action {action!r}; expected download or upload")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
