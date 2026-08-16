"""The repository must actually contain the package it claims to ship.

This exists because it didn't. A `.gitignore` entry of `data/`, intended for a local
cache directory at the repo root, matches a directory of that name at *any* depth — so
`src/wxfuser/data/` was silently excluded and the published repository was missing nine
modules. Everything still worked locally, where the files exist on disk; only a fresh
clone was broken, which is precisely the case no one runs before pushing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "wxfuser"


def _is_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists()


def _tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_package_module_is_tracked():
    """Every .py file under the package must be committed, not merely present locally."""
    tracked = _tracked_files()
    on_disk = {
        str(p.relative_to(REPO_ROOT))
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "these package modules exist locally but are not in git, so a fresh clone "
        f"would be broken: {missing}"
    )


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_subpackage_has_an_init():
    """A subpackage without __init__.py installs inconsistently across tools."""
    for pkg_dir in PACKAGE_ROOT.rglob("*"):
        if not pkg_dir.is_dir() or "__pycache__" in pkg_dir.parts:
            continue
        if any(p.suffix == ".py" for p in pkg_dir.iterdir() if p.is_file()):
            assert (pkg_dir / "__init__.py").exists(), f"{pkg_dir} has no __init__.py"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_ignore_rules_for_local_caches_are_anchored():
    """Cache ignore rules must be rooted so they cannot swallow package directories."""
    lines = [
        ln.strip()
        for ln in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # These names also occur inside the package tree, so an unanchored rule is a trap.
    dangerous = {"data/", "state/", "site/", "models/", "verify/", "pipeline/"}
    unanchored = [ln for ln in lines if ln in dangerous]
    assert not unanchored, (
        "these ignore rules match directories at any depth and would exclude package "
        f"code; anchor them with a leading slash: {unanchored}"
    )


def test_configs_are_shipped():
    """The package reads its constants from configs/, so they must exist in a clone."""
    for name in ("models.yaml", "tiers.yaml", "hub.yaml"):
        assert (REPO_ROOT / "configs" / name).exists(), f"configs/{name} is missing"
