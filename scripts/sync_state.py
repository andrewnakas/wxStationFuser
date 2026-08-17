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


def _repo_exists() -> bool:
    """Whether the state repo is already there.

    Only a genuine absence answers False. Every other failure propagates, because the
    upload guard reads this as "nothing to protect" — so treating a timeout or a bad
    token as absence would license the shallow-overwrite this is here to prevent.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    api, token = _api()
    try:
        api.repo_info(repo_id=hf_state_repo(), repo_type="dataset", token=token)
        return True
    except RepositoryNotFoundError:
        return False


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
        path = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(STATE_DIR),
            token=token,  # public repos read without one
            allow_patterns=allow,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {repo} exists but could not be restored ({exc}).")
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
    scope = _restored_scope()
    if _repo_exists() and scope is None:
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
        return download(args.shard, args.of, args.paths)
    if args.action == "upload":
        return upload(args.shard, args.of, args.paths)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
