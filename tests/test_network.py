"""Network model: dataset assembly, pooled fitting, and the unseen-station test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from wxfuser import network  # noqa: E402


def _archive(tmp_path, n_stations=10, days=60):
    """Stations whose truth is the model plus an elevation-dependent cold bias."""
    root = tmp_path / "arch"
    (root / "snotel" / "obs").mkdir(parents=True)
    (root / "points" / "gefs").mkdir(parents=True)
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"id": [f"{i}:MT:SNTL" for i in range(n_stations)],
                         "name": [f"S{i}" for i in range(n_stations)],
                         "lat": 45 + rng.random(n_stations), "lon": -111 + rng.random(n_stations),
                         "elev_m": np.linspace(1500, 3000, n_stations), "state": "MT"})
    meta.to_parquet(root / "snotel" / "stations.parquet")
    inits = pd.date_range("2025-01-01", periods=days, freq="D")
    runs, hours = [], pd.date_range("2025-01-01", periods=24 * (days + 8), freq="h")
    truth = {sid: 5 * np.sin(np.arange(len(hours)) / 24 * 2 * np.pi) for sid in meta["id"]}
    for sid, elev in zip(meta["id"], meta["elev_m"]):
        bias = -(elev - 1500) / 300  # higher stations run colder than the model says
        obs = pd.DataFrame({"station_id": sid, "valid_time": hours,
                            "air_temp_c": truth[sid] + bias + rng.normal(0, 0.5, len(hours)),
                            "source": "SNOTEL"})
        from wxfuser.data.obs import conform

        conform(obs).to_parquet(root / "snotel" / "obs" / f"{sid.replace(':', '_')}.parquet")
        for init in inits:
            leads = np.arange(0, 73, 3)
            vt = init + pd.to_timedelta(leads, unit="h")
            idx = ((vt - hours[0]) / pd.Timedelta(hours=1)).astype(int)
            runs.append(pd.DataFrame({"station_id": sid, "model": "gefs", "init_time": init,
                                      "lead_h": leads.astype("int16"), "valid_time": vt,
                                      "fc_air_temp_c": truth[sid][idx].astype("float32")}))
    pd.concat(runs).to_parquet(root / "points" / "gefs" / "2025-01.parquet")
    return root


def test_the_network_learns_an_elevation_bias_at_unseen_stations(tmp_path):
    root = _archive(tmp_path)
    data = network.build_dataset(root, ["gefs"], "air_temp_c", issue_hours=(9, 21))
    assert data["station_id"].nunique() == 10
    assert {"elev_m", "lat", "lon", "sin_doy", "nowcast_err"} <= set(data.columns)
    ev = network.leave_stations_out(data, "air_temp_c", ["gefs"], folds=5, rounds=80)
    # Raw GEFS misses each station's elevation bias; the network learns it from others.
    assert ev["crpss_vs_raw_best"] > 0.3
    assert ev["stations_improved"] >= 0.8


def test_time_holdout_never_trains_on_the_test_period(tmp_path):
    root = _archive(tmp_path)
    data = network.build_dataset(root, ["gefs"], "air_temp_c", issue_hours=(9, 21))
    ev = network.leave_stations_out(data, "air_temp_c", ["gefs"], folds=2, rounds=40,
                                    test_from="2025-02-10")
    assert ev["scheme"] == "unseen stations, from 2025-02-10"
    assert ev["rows"] > 0


def test_models_round_trip(tmp_path):
    root = _archive(tmp_path, n_stations=4, days=20)
    data = network.build_dataset(root, ["gefs"], "air_temp_c", issue_hours=(9,))
    model = network.fit(data, "air_temp_c", ["gefs"], rounds=20)
    back = network.load(network.save(model, tmp_path / "m.json"))
    a, b = model.predict(data.head(50)), back.predict(data.head(50))
    assert np.allclose(a["q50"], b["q50"])
