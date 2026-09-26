"""FusionSpec: keys, isolation, validation, org files, and exceedance probabilities."""
from __future__ import annotations

import numpy as np
import pytest

from wxfuser.data.registry import Station
from wxfuser.pipeline import emit
from wxfuser.spec import FusionSpec, SpecError, exceedance, load_org

SNOWBIRD = Station(id="766:UT:SNTL", name="Snowbird", lat=40.57, lon=-111.66, elev_m=2795)


def _spec(**kw):
    base = dict(org="wasatch-patrol", station=SNOWBIRD, models=["hrrr", "gefs"],
                variables=["air_temp_c", "wind_gust_ms"])
    base.update(kw)
    return FusionSpec(**base)


def test_key_depends_on_the_calibration_not_its_presentation():
    a = _spec()
    assert a.key() == _spec(models=["gefs", "hrrr"]).key()  # order is not a choice
    assert a.key() == _spec(thresholds={"wind_gust_ms": [20.0]}, label="x").key()
    assert a.key() != _spec(models=["hrrr"]).key()
    assert a.key() != _spec(org="other-org").key()


def test_two_specs_on_one_station_never_share_storage(tmp_path, monkeypatch):
    from wxfuser.pipeline import core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    a, b = _spec().to_station(), _spec(models=["gefs"]).to_station()
    assert core.pairs_path(a) != core.pairs_path(b)
    assert core.pairs_path(a) != core.pairs_path(SNOWBIRD)
    assert a.resolved_models() == ["hrrr", "gefs"]
    assert a.id == SNOWBIRD.id  # observations still come from the real station


def test_validation_refuses_what_cannot_be_calibrated():
    with pytest.raises(ValueError):
        _spec(models=["hrrr", "gfs_seamless"]).validate()  # mixed sources
    with pytest.raises(SpecError, match="does not forecast"):
        _spec(thresholds={"precip_1h_mm": [2.0]}).validate()
    with pytest.raises(SpecError, match="org id"):
        _spec(org="Wasatch Patrol").validate()
    with pytest.raises(SpecError, match="unknown models"):
        _spec(models=["hrrr", "nam"]).validate()


def test_org_file_resolves_public_and_private_stations(tmp_path):
    f = tmp_path / "specs.yaml"
    f.write_text("""
org: wasatch-patrol
stations:
  - {id: "ORG:wasatch-patrol:collins", name: Collins, lat: 40.58, lon: -111.64, elev_m: 2900}
specs:
  - station: "ORG:wasatch-patrol:collins"
    models: [hrrr, gefs]
    variables: [air_temp_c, wind_gust_ms]
    thresholds: {wind_gust_ms: [20, 25]}
  - station: "766:UT:SNTL"
    models: [hrrr]
    variables: [air_temp_c]
""")
    org, specs = load_org(f)
    assert org == "wasatch-patrol"
    assert [s.station.id for s in specs] == ["ORG:wasatch-patrol:collins", "766:UT:SNTL"]
    assert specs[0].thresholds == {"wind_gust_ms": [20.0, 25.0]}
    # A public station's location comes from the registry, not the org's file.
    assert specs[1].station.name == "Snowbird"


def test_org_stations_must_carry_their_org_prefix(tmp_path):
    f = tmp_path / "specs.yaml"
    f.write_text("""
org: wasatch-patrol
stations:
  - {id: "ORG:someone-else:x", name: X, lat: 40, lon: -111}
specs: []
""")
    with pytest.raises(SpecError, match="must be named"):
        load_org(f)


LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def test_exceedance_reads_the_quantile_function():
    q = np.array([[5.0, 8.0, 10.0, 12.0, 15.0]])
    p = exceedance(LEVELS, q, [10.0, 12.0, 100.0, -100.0])
    assert p["10"][0] == pytest.approx(0.5)
    assert p["12"][0] == pytest.approx(0.25)
    # Beyond the dense grid the answer saturates at its resolution, never 0 or 1.
    assert p["100"][0] == pytest.approx(0.01)
    assert p["-100"][0] == pytest.approx(0.99)


def test_exceedance_of_zero_on_a_dry_plateau_is_the_wet_probability():
    """Precipitation quantiles pin q <= 1 - p_occ to zero; P(X > 0) must equal p_occ."""
    q = np.array([[0.0, 0.0, 0.0, 0.5, 2.0]])  # p_occ between 0.25 and 0.5
    p = exceedance(LEVELS, q, [0.0])["0"][0]
    assert 0.25 <= p <= 0.5


def test_spec_payload_extends_rather_than_replaces():
    base = {"schema_version": 1, "hourly": {"time": ["t"], "wind_gust_ms": {"q50": [9.0]}},
            "raw": {}, "skill": {}}
    out = emit.spec_forecast_json(base, {"key": "k"}, {"wind_gust_ms": {"20": np.array([0.3])}},
                                  "2026-09-26T01:00Z")
    assert out["schema_version"] == 2
    assert out["hourly"]["wind_gust_ms"]["q50"] == [9.0]
    assert out["hourly"]["wind_gust_ms"]["p_exceed"] == {"20": [0.3]}
    assert out["obs_latest"] == "2026-09-26T01:00Z"
    # The public payload object is not mutated, down to its per-variable blocks.
    assert base["schema_version"] == 1
    assert "p_exceed" not in base["hourly"]["wind_gust_ms"]
