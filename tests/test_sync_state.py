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
        self.info_calls = 0

    def repo_info(self, **_):
        self.info_calls += 1
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
    # The existence probe is memoised for the life of a run, which is right in production
    # — it was costing an API call on every checkpoint upload — and wrong across tests,
    # where each one installs a different hub.
    sync_state._repo_exists.cache_clear()
    # Waiting out a rate limit is the point of the retry, but not at test speed.
    monkeypatch.setattr(sync_state.time, "sleep", lambda *_: None)
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


# ------------------------------------------------------------------ rate limiting

class _RateLimited(Exception):
    """What the hub raises at the quota, near enough for the parts that are read."""

    def __init__(self, seconds=None):
        super().__init__(
            "429 Too Many Requests: you have reached your 'api' rate limit."
            + (f" Retry after {seconds} seconds (0/1000 requests remaining)."
               if seconds else "")
        )
        self.response = type("R", (), {"status_code": 429, "headers": {}})()


def test_a_rate_limit_is_waited_out_rather_than_raised(hub):
    """Five days of scheduled refreshes died here.

    Six shards restoring, checkpointing and uploading exhausted the hub's 1000 requests
    per five minutes; the 429 came out of the first call that met it and the whole run
    banked nothing. A quota is a queue, and something has to wait in it.
    """
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _RateLimited(seconds=7)
        return "done"

    assert sync_state.with_rate_limit_retry(flaky, "test call") == "done"
    assert calls["n"] == 3


def test_the_wait_honours_what_the_hub_asked_for(hub, monkeypatch):
    waited = []
    monkeypatch.setattr(sync_state.time, "sleep", lambda s: waited.append(s))

    def always():
        raise _RateLimited(seconds=102)

    with pytest.raises(_RateLimited):
        sync_state.with_rate_limit_retry(always, "test call", attempts=3)
    # Its number, plus a little, so six shards told the same figure do not return together.
    assert waited and all(s > 102 for s in waited)


def test_a_failure_that_is_not_a_rate_limit_still_propagates(hub):
    """The guard this protects only works if real failures still fail."""
    def broken():
        raise OSError("connection reset")

    with pytest.raises(OSError):
        sync_state.with_rate_limit_retry(broken, "test call")


def test_the_existence_probe_is_not_repeated_within_a_run(hub):
    """It was being asked on every checkpoint upload, ninety times a refresh."""
    hub["api"] = _FakeApi(exists=True)
    sync_state._repo_exists.cache_clear()

    assert sync_state._repo_exists() is True
    assert sync_state._repo_exists() is True
    assert sync_state._repo_exists() is True
    assert hub["api"].info_calls == 1


def test_an_upload_after_a_restore_asks_the_hub_nothing_extra(hub, tmp_path):
    """The specific waste that broke the fleet.

    The guard asked whether the repository exists before checking whether the answer
    could change anything, so a run that had plainly just restored still spent a call on
    it — on every checkpoint, into a quota shared with every other shard.
    """
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / sync_state.RESTORE_MARKER).write_text("repo\nfull\n")
    (tmp_path / "state" / "pairs").mkdir(exist_ok=True)
    sync_state._repo_exists.cache_clear()

    assert sync_state.upload() == 0
    assert hub["api"].info_calls == 0, "an upload after a restore needs no existence probe"


def test_only_trained_keeps_the_stations_that_can_publish(monkeypatch, tmp_path):
    """The filter that refills an empty map.

    A station with no archive fetches a forecast, finds nothing to calibrate against and
    reports itself warming up — honest, and a wasted request when the map is empty and
    the forecast API is rationed.
    """
    from wxfuser import cli
    from wxfuser.data.registry import Station
    from wxfuser.pipeline import core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "ASOS_DEN.parquet").write_bytes(b"")

    stations = [
        Station(id="ASOS:DEN", name="Denver", lat=39.8, lon=-104.7),
        Station(id="ASOS:BOS", name="Boston", lat=42.4, lon=-71.0),
    ]
    assert [s.id for s in cli.only_trained(stations)] == ["ASOS:DEN"]
    assert cli.only_trained([]) == []


def test_a_bounding_box_selects_a_region_and_rejects_a_malformed_one():
    """So a run can be pointed at the mountains in snow season rather than waiting for a
    full pass to reach them."""
    from wxfuser import cli
    from wxfuser.data.registry import Station

    stations = [
        Station(id="ASOS:DEN", name="Denver", lat=39.83, lon=-104.66),   # just outside
        Station(id="ASOS:SEA", name="Seattle", lat=47.44, lon=-122.31),  # inside
        Station(id="ASOS:BOS", name="Boston", lat=42.36, lon=-71.01),    # far outside
    ]
    picked = cli.filter_bbox(stations, "31,-125,49.5,-105")
    assert [s.id for s in picked] == ["ASOS:SEA"]
    assert cli.filter_bbox(stations, None) == stations

    import pytest as _pytest
    with _pytest.raises(SystemExit):
        cli.filter_bbox(stations, "not-a-box")


def test_publishable_stations_are_refreshed_before_the_rest(monkeypatch, tmp_path):
    """Which stations a run reaches first decides what the map shows when it is cut off.

    Six shards over nine thousand stations against a throttled API is hours of work, and
    a shard killed at its timeout contributes only what it already processed. A station
    with no archive cannot publish either way, so doing those first spends the whole
    budget without adding anything to the map.
    """
    from wxfuser import cli
    from wxfuser.data.registry import Station
    from wxfuser.pipeline import core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "ASOS_SEA.parquet").write_bytes(b"")

    stations = [
        Station(id="ASOS:AAA", name="warming", lat=40.0, lon=-100.0),
        Station(id="ASOS:SEA", name="ready", lat=47.4, lon=-122.3),
        Station(id="ASOS:ZZZ", name="warming too", lat=41.0, lon=-101.0),
    ]
    assert [s.id for s in cli.publishable_first(stations)] == [
        "ASOS:SEA", "ASOS:AAA", "ASOS:ZZZ"
    ]

    # With no archives at all — a genuine cold start — the order is left alone rather
    # than every station being labelled unpublishable.
    monkeypatch.setattr(core, "STATE_DIR", tmp_path / "empty")
    assert cli.publishable_first(stations) == stations
