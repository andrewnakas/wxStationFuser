"""The state sync, exercised by actually calling it.

This file exists because ``download()`` and ``upload()`` once shipped with a TypeError on
their first line — a helper gained a keyword argument at one call site and not at the
definition. Nothing imported the module, so the whole suite stayed green while every
bootstrap, refresh, and retrain job on CI died at its first step and the archives sat
frozen for days.

So these tests care less about hub semantics than about the two things that went wrong:
the functions are callable at all, and the guard that protects deep history actually
refuses when it should.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sync_state  # noqa: E402


class _FakeApi:
    def __init__(self, *, exists=True, error=None):
        self._exists = exists
        self._error = error
        self.uploaded = None

    def repo_info(self, **_):
        if self._error:
            raise self._error
        if not self._exists:
            import requests
            from huggingface_hub.errors import RepositoryNotFoundError

            response = requests.Response()
            response.status_code = 404
            raise RepositoryNotFoundError("absent", response=response)
        return object()

    def create_repo(self, **_):
        return None

    def upload_folder(self, **kw):
        self.uploaded = kw


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Point the module at a temp state dir and a controllable fake hub."""
    monkeypatch.setattr(sync_state, "STATE_DIR", tmp_path / "state")

    state = {"api": _FakeApi(), "snapshot_fails": False}

    def fake_api(*, required: bool = False):
        return state["api"], "tok"

    def fake_snapshot(**kw):
        if state["snapshot_fails"]:
            raise OSError("connection reset")
        target = Path(kw["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "pairs").mkdir(exist_ok=True)
        (target / "pairs" / "a.parquet").write_text("x")
        return str(target)

    monkeypatch.setattr(sync_state, "_api", fake_api)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    return state


def test_download_is_callable_and_restores(hub):
    """The regression that started this file: download() must not raise on its own call."""
    assert sync_state.download() == 0
    assert (sync_state.STATE_DIR / sync_state.RESTORE_MARKER).exists()


def test_cold_start_is_not_an_error(hub):
    hub["api"] = _FakeApi(exists=False)
    assert sync_state.download() == 0
    assert not (sync_state.STATE_DIR / sync_state.RESTORE_MARKER).exists()


def test_failed_restore_refuses_rather_than_continuing(hub):
    """A restore that fails must not leave a marker; the upload guard depends on that."""
    hub["snapshot_fails"] = True
    assert sync_state.download() == 1
    assert not (sync_state.STATE_DIR / sync_state.RESTORE_MARKER).exists()


def test_upload_refuses_when_state_was_never_restored(hub):
    sync_state.STATE_DIR.mkdir(parents=True)
    assert sync_state.upload() == 1
    assert hub["api"].uploaded is None


def test_upload_proceeds_after_a_real_restore(hub):
    assert sync_state.download() == 0
    assert sync_state.upload() == 0
    assert hub["api"].uploaded is not None


def test_transient_hub_failure_does_not_read_as_absent(hub):
    """A timeout must propagate, not answer 'no repo' and license an overwrite."""
    hub["api"] = _FakeApi(error=OSError("timeout"))
    with pytest.raises(OSError):
        sync_state._repo_exists()


def test_restore_marker_is_never_published(hub):
    """Uploading the marker would tell the next cold runner it had already restored."""
    assert sync_state.download() == 0
    assert sync_state.upload() == 0
    assert sync_state.RESTORE_MARKER in hub["api"].uploaded["ignore_patterns"]


# ------------------------------------------------- restoring only what a worker needs

def test_a_shard_restores_only_its_own_stations(hub, monkeypatch):
    """Twenty runners each pulling the whole archive is what drew Hugging Face's 429s."""
    seen = {}

    def fake_snapshot(**kw):
        seen.update(kw)
        target = Path(kw["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    assert sync_state.download(shard=0, of=20) == 0

    allow = seen.get("allow_patterns")
    assert allow, "a sharded restore must be scoped, not a full pull"
    assert any(p.startswith("pairs/") for p in allow)
    # Shared state is not per-station and every job needs it whatever slice it holds.
    assert "catalogue/**" in allow and "site/**" in allow


def test_an_unsharded_restore_still_takes_everything(hub):
    assert sync_state.download() == 0
    assert sync_state._restored_scope() == "full"


def test_a_partial_restore_may_not_be_uploaded_as_a_whole_one(hub, monkeypatch):
    """Local state after a scoped restore is one slice; publishing it unscoped would
    present a twentieth of the archive as the entirety of it."""
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **kw: (Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True),
                                      str(kw["local_dir"]))[1])
    assert sync_state.download(shard=3, of=20) == 0
    assert sync_state._restored_scope() == "3/20"

    assert sync_state.upload() == 1, "an unscoped upload after a scoped restore must refuse"
    assert hub["api"].uploaded is None
    assert sync_state.upload(shard=3, of=20) == 0, "its own shard is fine"


def test_a_shard_may_not_upload_a_different_shard(hub, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **kw: (Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True),
                                      str(kw["local_dir"]))[1])
    assert sync_state.download(shard=3, of=20) == 0
    assert sync_state.upload(shard=7, of=20) == 1
