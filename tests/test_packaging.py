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


def _declared_dependencies() -> set[str]:
    """Every distribution named in pyproject, across the base and all extras."""
    import re
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    project = cfg.get("project", {})
    specs = list(project.get("dependencies", []))
    for extra in (project.get("optional-dependencies") or {}).values():
        specs.extend(extra)
    # "huggingface-hub>=0.23" -> "huggingface_hub"
    return {
        re.split(r"[<>=!~\[; ]", s, maxsplit=1)[0].strip().lower().replace("-", "_")
        for s in specs
        if s.strip()
    }


def _imported_top_level_modules() -> set[str]:
    """Top-level modules the package imports, including inside functions."""
    import ast

    found: set[str] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return found


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_third_party_import_is_declared():
    """An undeclared dependency works locally and breaks every fresh environment.

    duckdb was imported by the ASOS bulk reader but named nowhere in pyproject. It ran
    fine on a machine where it had been pip-installed by hand, and on CI every airport
    station failed while the other networks succeeded — a partial, confusing failure
    rather than an obvious import error at startup.
    """
    import sys

    # Import name -> distribution that provides it, for the packages where they differ.
    PROVIDED_BY = {
        "yaml": "pyyaml",
        "sklearn": "scikit_learn",
        "dateutil": "python_dateutil",
    }

    stdlib = set(sys.stdlib_module_names)
    first_party = {"wxfuser"}
    declared = _declared_dependencies()

    undeclared = set()
    for m in _imported_top_level_modules():
        if m in stdlib or m in first_party:
            continue
        dist = PROVIDED_BY.get(m, m).lower().replace("-", "_")
        if dist not in declared:
            undeclared.add(m)
    assert not undeclared, (
        f"these are imported but not declared in pyproject.toml, so a fresh install "
        f"would fail at runtime: {sorted(undeclared)}"
    )


def test_shard_assignment_is_stable_as_the_registry_grows():
    """A station must keep its worker when other stations are added.

    Positional round-robin reassigns nearly everything on any size change, which breaks
    runs already in flight: the worker holds a station list from startup while its upload
    step recomputes the split from a registry that has since grown, so it uploads paths it
    never touched.
    """
    from wxfuser.cli import shard_of

    ids = [f"ASOS:{i:04d}" for i in range(500)]
    of = 20
    before = {i: shard_of(i, of) for i in ids}

    grown = ids + [f"SNOTEL:{i}" for i in range(400)]
    after = {i: shard_of(i, of) for i in grown}

    moved = [i for i in ids if before[i] != after[i]]
    assert not moved, f"{len(moved)} stations changed worker when the registry grew"


def test_shard_assignment_is_balanced_and_covers_everything():
    from collections import Counter

    from wxfuser.cli import shard_of

    ids = [f"ASOS:{i:04d}" for i in range(2000)]
    of = 20
    counts = Counter(shard_of(i, of) for i in ids)
    assert sum(counts.values()) == len(ids)
    assert len(counts) == of, "some worker would get no stations at all"
    # Hashing will not be perfectly even, but it must not be badly lopsided.
    assert max(counts.values()) < 2 * min(counts.values())


def _dev_environment_distributions() -> set[str]:
    """What `pip install -e .[dev]` actually puts in the environment.

    Distinct from _declared_dependencies, which unions every extra and so answers only
    "is this named somewhere in pyproject". Installing the package brings the base
    dependencies; the dev extra brings its own, and a `wxfuser[x]` self-reference brings
    that extra too.
    """
    import re
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    project = cfg.get("project", {})
    extras = project.get("optional-dependencies") or {}

    def name_of(spec: str) -> str:
        return re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].strip().lower().replace("-", "_")

    specs = list(project.get("dependencies", []))
    pending, seen = ["dev"], set()
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for spec in extras.get(extra, []):
            ref = re.match(r"^wxfuser\[([^\]]+)\]", spec.strip())
            if ref:
                pending.extend(p.strip() for p in ref.group(1).split(","))
            else:
                specs.append(spec)
    return {name_of(s) for s in specs if s.strip()}


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_the_test_suite_runs_in_the_environment_ci_builds():
    """Tests may only import what `.[dev]` installs.

    A test importing something CI does not install passes locally and errors there. That
    happened with huggingface_hub: it was declared, but only under the train extra, so the
    sync-state tests errored on CI while passing on a developer machine — and those are
    precisely the tests guarding the bug that killed every data job for days.
    """
    import ast
    import sys

    PROVIDED_BY = {"yaml": "pyyaml", "sklearn": "scikit_learn", "dateutil": "python_dateutil"}
    stdlib = set(sys.stdlib_module_names)
    local = {"wxfuser", "sync_state", "validate_enrollment", "conftest"}
    available = _dev_environment_distributions()

    missing: dict[str, set[str]] = {}
    for folder in ("tests", "scripts"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module.split(".")[0]]
                for m in mods:
                    if m in stdlib or m in local:
                        continue
                    if PROVIDED_BY.get(m, m).lower().replace("-", "_") not in available:
                        missing.setdefault(path.name, set()).add(m)

    assert not missing, (
        "these are imported by tests/scripts but not installed by `.[dev]`, so they pass "
        f"locally and error on CI: { {k: sorted(v) for k, v in missing.items()} }"
    )


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_job_touching_the_hub_installs_what_that_needs():
    """A job that runs sync_state.py must install the extra providing huggingface_hub.

    The publish job installed the base package only, so its hub restore raised
    ModuleNotFoundError on every deploy. The call is wrapped in `|| echo`, so the job went
    green while never once reading the hub — which is how the published catalogue
    disappeared and why a merged index had no baseline to build on. A tolerated failure
    that can never succeed is worse than a loud one.
    """
    import re

    import yaml

    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows, "no workflows found"

    offenders = []
    for wf in workflows:
        cfg = yaml.safe_load(wf.read_text())
        for job_name, job in (cfg.get("jobs") or {}).items():
            steps = job.get("steps") or []
            run_text = "\n".join(s.get("run", "") for s in steps if isinstance(s, dict))
            if "sync_state.py" not in run_text:
                continue
            installs = re.findall(r"pip install[^\n]*", run_text)
            # huggingface_hub ships with the train extra, so a bare `pip install -e .`
            # leaves the hub unreachable no matter how the call is guarded.
            if not any("[train]" in i or "[dev]" in i for i in installs):
                offenders.append(f"{wf.name}:{job_name} installs {installs or ['nothing']}")

    assert not offenders, (
        "these jobs run sync_state.py without installing huggingface_hub, so their hub "
        f"access can only ever fail: {offenders}"
    )
