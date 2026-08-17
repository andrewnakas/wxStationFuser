"""Request pacing and the retry policy.

Open-Meteo prices a request by locations x models x variables rather than by count, so a
wide batch draws bursts of 429s. Those are not failures — they are the server asking for a
pause — and a shard that has already downloaded twenty minutes of history should wait them
out rather than die of politeness.
"""
from __future__ import annotations

from urllib.error import HTTPError

import pytest

from wxfuser.data import openmeteo


@pytest.fixture(autouse=True)
def _no_real_waiting(monkeypatch):
    monkeypatch.setattr(openmeteo.time, "sleep", lambda *_: None)
    monkeypatch.setattr(openmeteo, "_last_request_at", 0.0)


def _throttle(n_before_success: int):
    """A urlopen that answers 429 n times, then succeeds."""
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= n_before_success:
            raise HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return _Resp()

    return fake_urlopen, calls


def test_a_burst_of_throttling_is_waited_out(monkeypatch):
    """Eight 429s must not exhaust a five-attempt budget — they are not attempts."""
    fake, calls = _throttle(8)
    monkeypatch.setattr(openmeteo, "urlopen", fake)

    assert openmeteo._http_json("https://x/y") == {"ok": True}
    assert calls["n"] == 9


def test_throttling_still_gives_up_eventually(monkeypatch):
    """Unbounded patience would hang a run against a hard quota."""
    fake, _ = _throttle(10_000)
    monkeypatch.setattr(openmeteo, "urlopen", fake)

    with pytest.raises(openmeteo.OpenMeteoError):
        openmeteo._http_json("https://x/y")


def test_real_errors_still_exhaust_the_retry_budget(monkeypatch):
    """A 500 is a failure, not a request to slow down, and must not retry forever."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise HTTPError(req.full_url, 500, "Server Error", {}, None)

    monkeypatch.setattr(openmeteo, "urlopen", fake_urlopen)
    with pytest.raises(openmeteo.OpenMeteoError):
        openmeteo._http_json("https://x/y", retries=3)
    assert calls["n"] == 3
