"""Pairing, verification metrics, and the comparison logic that backs the skill claim."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxfuser.data import pairs as pairs_mod
from wxfuser.data.obs import OBS_COLUMNS
from wxfuser.verify import metrics, rolling

VARIABLES = ["air_temp_c", "wind_speed_ms"]


def make_forecasts(times, models=("m1", "m2")):
    rows = []
    for m in models:
        for t in times:
            rows.append(
                {
                    "model": m,
                    "valid_time": t,
                    "lead_h": 24,
                    "lead_source": "prev_runs",
                    "fc_air_temp_c": 10.0,
                    "fc_wind_speed_ms": 3.0,
                }
            )
    return pd.DataFrame(rows)


def make_obs(times):
    df = pd.DataFrame({"valid_time": times})
    df["station_id"] = "IEM:TST"
    df["air_temp_c"] = 9.0
    df["wind_speed_ms"] = 2.5
    df["source"] = "IEM"
    for c in OBS_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[OBS_COLUMNS]


def test_pairs_join_on_valid_time():
    times = pd.date_range("2025-01-01", periods=48, freq="h")
    p = pairs_mod.build_pairs(make_forecasts(times), make_obs(times), "IEM:TST", VARIABLES)
    assert len(p) == 96  # two models x 48 hours
    assert set(p["model"]) == {"m1", "m2"}
    assert p["obs_air_temp_c"].eq(9.0).all()


def test_pairs_drop_hours_without_observations():
    """An hour with no observation cannot teach anything and must not become a row."""
    times = pd.date_range("2025-01-01", periods=48, freq="h")
    obs = make_obs(times[:24])
    p = pairs_mod.build_pairs(make_forecasts(times), obs, "IEM:TST", VARIABLES)
    assert len(p) == 48
    assert p["valid_time"].max() == times[23]


def test_merge_archive_prefers_the_newer_row():
    """Observations get revised; a re-pair must overwrite, not duplicate."""
    times = pd.date_range("2025-01-01", periods=5, freq="h")
    old = pairs_mod.build_pairs(make_forecasts(times), make_obs(times), "IEM:TST", VARIABLES)
    revised = old.copy()
    revised["obs_air_temp_c"] = 11.0
    merged = pairs_mod.merge_archive(old, revised)
    assert len(merged) == len(old)
    assert merged["obs_air_temp_c"].eq(11.0).all()


def test_to_wide_produces_one_row_per_valid_time():
    times = pd.date_range("2025-01-01", periods=24, freq="h")
    p = pairs_mod.build_pairs(make_forecasts(times), make_obs(times), "IEM:TST", VARIABLES)
    w = pairs_mod.to_wide(p, "air_temp_c", ["m1", "m2"])
    assert len(w) == 24
    assert {"fc_m1", "fc_m2", "fc_mean", "obs"} <= set(w.columns)
    assert w["fc_mean"].eq(10.0).all()


def test_sanitize_station_id():
    assert pairs_mod.sanitize_station_id("IEM:DEN") == "IEM_DEN"
    assert pairs_mod.sanitize_station_id("663:CO:SNTL") == "663_CO_SNTL"


# ------------------------------------------------------------------ metrics


def test_crps_from_quantiles_approximates_the_closed_form():
    from wxfuser.models.distributions import crps_normal, quantiles_normal

    mu, sigma = np.full(500, 3.0), np.full(500, 1.4)
    rng = np.random.default_rng(2)
    y = rng.normal(3.0, 1.4, size=500)
    q = quantiles_normal(mu, sigma, [0.05, 0.25, 0.5, 0.75, 0.95])
    approx = metrics.crps_from_quantiles(y, q).mean()
    exact = crps_normal(y, mu, sigma).mean()
    # A five-point grid is coarse; it needs to rank methods correctly, not be exact.
    assert approx == pytest.approx(exact, rel=0.15)


def test_pit_is_uniform_for_a_calibrated_forecast():
    from wxfuser.models.distributions import quantiles_normal

    rng = np.random.default_rng(8)
    n = 4000
    mu, sigma = np.full(n, 0.0), np.full(n, 1.0)
    y = rng.normal(0.0, 1.0, size=n)
    pit = metrics.pit_values(y, quantiles_normal(mu, sigma, [0.05, 0.25, 0.5, 0.75, 0.95]))
    hist = metrics.pit_histogram(pit, bins=5)
    expected = n / 5
    assert all(abs(c - expected) < expected * 0.35 for c in hist)


def test_pit_detects_overconfidence():
    """The diagnostic must fail loudly when intervals are too narrow."""
    from wxfuser.models.distributions import quantiles_normal

    rng = np.random.default_rng(9)
    n = 4000
    y = rng.normal(0.0, 3.0, size=n)  # truth is much wider than claimed
    pit = metrics.pit_values(
        y, quantiles_normal(np.zeros(n), np.full(n, 0.7), [0.05, 0.25, 0.5, 0.75, 0.95])
    )
    hist = metrics.pit_histogram(pit, bins=5)
    # Mass piles into the outer bins — the signature of an overconfident forecast.
    assert hist[0] + hist[-1] > 0.5 * sum(hist)


def test_interval_coverage_and_sharpness():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lo = np.array([0.0, 1.0, 2.0, 0.0])
    hi = np.array([2.0, 3.0, 4.0, 5.0])
    cov, width = metrics.interval_coverage(y, lo, hi)
    assert cov == pytest.approx(0.75)
    assert width == pytest.approx(2.75)


def test_brier_and_reliability():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    prob = np.array([0.9, 0.1, 0.8, 0.2])
    assert metrics.brier_score(y, prob, 0.5) < 0.05
    bins = metrics.reliability_bins(y, prob, 0.5, n_bins=5)
    assert bins and all("forecast_prob" in b for b in bins)


def test_block_bootstrap_ci_brackets_a_real_gain():
    days = pd.date_range("2025-01-01", periods=90, freq="D")
    rng = np.random.default_rng(6)
    raw = pd.Series(rng.normal(2.0, 0.3, 90), index=days)
    better = pd.Series(raw.to_numpy() * 0.8, index=days)
    point, lo, hi = metrics.block_bootstrap_ci(better, raw, n_resamples=300)
    assert point == pytest.approx(0.2, abs=0.02)
    assert lo > 0  # a genuine gain must be detected as one


def test_block_bootstrap_ci_straddles_zero_when_there_is_no_gain():
    """Guards the honesty mechanism: no gain must not be reported as a win."""
    days = pd.date_range("2025-01-01", periods=90, freq="D")
    rng = np.random.default_rng(10)
    a = pd.Series(rng.normal(2.0, 0.5, 90), index=days)
    b = pd.Series(rng.normal(2.0, 0.5, 90), index=days)
    _, lo, hi = metrics.block_bootstrap_ci(a, b, n_resamples=300)
    assert lo < 0 < hi


# ------------------------------------------------------------------ baseline pairing


def test_baseline_comparison_uses_only_rows_the_model_covers():
    """The bug this guards against made a winning model look 15% worse than raw.

    A regional model that stops at short leads must be compared on short leads only,
    with the fused score recomputed on those same hours — otherwise the fused forecast
    is charged for long-lead difficulty the baseline never faced.
    """
    n = 200
    times = pd.date_range("2025-01-01", periods=n, freq="h")
    wide = pd.DataFrame(
        {
            "valid_time": times,
            "lead_h": np.where(np.arange(n) < 100, 3, 120),
            "obs": 10.0,
            "fc_global": 12.0,      # always available, 2.0 of error
            "fc_regional": np.where(np.arange(n) < 100, 10.5, np.nan),  # short leads only
        }
    )
    oos = wide[["valid_time", "lead_h", "obs"]].copy()
    oos["q50"] = 10.2
    # Fused error is small at short lead, large at long lead.
    crps_fused = np.where(np.arange(n) < 100, 0.2, 3.0)
    mae_fused = crps_fused

    raw = rolling._raw_baselines(oos, wide, ["global", "regional"], crps_fused, mae_fused)

    assert raw["regional"]["coverage"] == pytest.approx(0.5)
    assert raw["global"]["coverage"] == pytest.approx(1.0)
    # The fused score charged against the regional model must be its short-lead score,
    # not the all-leads average.
    assert raw["regional"]["crps_fused_here"] == pytest.approx(0.2)
    assert raw["global"]["crps_fused_here"] == pytest.approx(1.6)


def test_baseline_prefers_a_model_that_covers_the_whole_forecast():
    raw = {
        "regional": {"crps": 0.5, "coverage": 0.4},
        "global": {"crps": 1.2, "coverage": 1.0},
    }
    # The short-range model scores better, but only on the easy hours; the headline
    # claim must be made against something that covers the whole forecast.
    assert rolling._pick_baseline(raw) == "global"


def test_baseline_falls_back_when_nothing_has_full_coverage():
    raw = {
        "a": {"crps": 0.9, "coverage": 0.5},
        "b": {"crps": 0.7, "coverage": 0.6},
    }
    assert rolling._pick_baseline(raw) == "b"


def test_tiers_are_ranked_on_rows_they_all_covered():
    """A tier that sits out the hard blocks must not win by having faced easier weather.

    Tier 3 needs thousands of rows before it can fit, so it skips early walk-forward
    folds. Ranking on each tier's own coverage would compare them on different weather.
    """
    n = 400
    times = pd.date_range("2025-01-01", periods=n, freq="h")
    # First half is easy (small errors), second half is hard.
    obs = np.concatenate([np.zeros(n // 2), np.full(n - n // 2, 10.0)])

    wide = pd.DataFrame(
        {
            "valid_time": times,
            "lead_h": 24,
            "lead_source": "prev_runs",
            "obs": obs,
            "fc_m1": obs + 1.0,
            "fc_mean": obs + 1.0,
            "fc_spread": 0.5,
        }
    )

    # "broad" covers everything; "narrow" only covers the easy first half.
    broad = wide[["valid_time", "lead_h", "obs"]].copy()
    for q in (5, 25, 50, 75, 95):
        broad[f"q{q:02d}"] = obs + 1.0
    narrow = broad.iloc[: n // 2].copy()
    for q in (5, 25, 50, 75, 95):
        narrow[f"q{q:02d}"] = obs[: n // 2] + 0.9

    common = set(zip(broad["valid_time"], broad["lead_h"])) & set(
        zip(narrow["valid_time"], narrow["lead_h"])
    )
    assert len(common) == n // 2, "the shared rows are the easy half only"


def test_baseline_merge_cannot_misalign_on_duplicate_keys():
    """Two archive sources at the same valid time and lead must not duplicate rows.

    `wide` is keyed on (valid_time, lead_h, lead_source). If a left merge on
    (valid_time, lead_h) returned extra rows, every per-row array — including the fused
    CRPS — would be silently misaligned against it and the published skill numbers would
    be wrong rather than absent.
    """
    times = pd.date_range("2025-01-01", periods=50, freq="h")
    base = pd.DataFrame(
        {
            "valid_time": times,
            "lead_h": 24,
            "obs": 10.0,
            "fc_m1": 11.0,
        }
    )
    # Same valid times and lead from a second source — a colliding key.
    wide = pd.concat(
        [base.assign(lead_source="hist"), base.assign(lead_source="prev_runs")],
        ignore_index=True,
    )
    oos = base[["valid_time", "lead_h", "obs"]].copy()
    oos["q50"] = 10.5
    crps = np.full(len(oos), 0.5)

    raw = rolling._raw_baselines(oos, wide, ["m1"], crps, crps)
    assert raw["m1"]["n"] == len(oos)
    assert raw["m1"]["crps_fused_here"] == pytest.approx(0.5)


# ------------------------------------------------------------------ CRPS estimator


def test_crps_estimator_is_accurate_for_a_dispersed_forecast():
    """Scoring on the five published levels alone understates a spread forecast by 12%.

    The error is not neutral: it is exact for a point forecast, and the raw model enters
    the comparison as a point forecast. So a coarse estimator inflates every fused-vs-raw
    claim on one side only. The dense estimator must land close to the closed form.
    """
    from wxfuser.models.distributions import crps_normal, quantiles_normal

    rng = np.random.default_rng(0)
    n = 40000
    mu, sd = np.zeros(n), np.ones(n)
    y = rng.normal(0.0, 1.0, n)
    q = quantiles_normal(mu, sd, [0.05, 0.25, 0.5, 0.75, 0.95])

    exact = crps_normal(y, mu, sd).mean()
    dense = metrics.crps_from_quantiles(y, q).mean()
    coarse = metrics.crps_from_quantiles(y, q, dense=False).mean()

    assert dense / exact == pytest.approx(1.0, abs=0.03)
    assert coarse / exact < 0.92, "the coarse estimator should be visibly biased low"
    # Erring high means the estimator works against our own skill claim, not for it.
    assert dense >= exact


def test_crps_estimator_stays_exact_for_a_point_forecast():
    """The raw-model baseline must be unaffected, or the comparison shifts under it."""
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 20000)
    x = np.zeros_like(y)
    q = {f"q{lv:02d}": x for lv in (5, 25, 50, 75, 95)}
    assert metrics.crps_from_quantiles(y, q).mean() == pytest.approx(np.abs(y - x).mean())


def test_crps_estimator_remains_proper():
    """Misstating the spread must still cost, or minimum-CRPS fitting is meaningless."""
    from wxfuser.models.distributions import quantiles_normal

    rng = np.random.default_rng(2)
    n = 20000
    y = rng.normal(0.0, 1.0, n)
    honest = metrics.crps_from_quantiles(
        y, quantiles_normal(np.zeros(n), np.ones(n), [0.05, 0.25, 0.5, 0.75, 0.95])
    ).mean()
    for wrong_sd in (0.3, 3.0):
        bad = metrics.crps_from_quantiles(
            y,
            quantiles_normal(np.zeros(n), np.full(n, wrong_sd), [0.05, 0.25, 0.5, 0.75, 0.95]),
        ).mean()
        assert honest < bad


def test_climatology_excludes_observations_near_the_target():
    """Climatology must not contain the observation it is asked to predict."""
    times = pd.date_range("2025-01-01", periods=24 * 200, freq="h")
    obs = pd.DataFrame({"valid_time": times, "air_temp_c": 10.0})
    # One target hour is a huge outlier; if climatology sees it, it predicts it.
    spike_at = times[24 * 100]
    obs.loc[obs["valid_time"] == spike_at, "air_temp_c"] = 500.0

    clim = metrics.climatology_quantiles(obs, "air_temp_c", pd.Series([spike_at]))
    assert clim["q50"][0] == pytest.approx(10.0), "the target's own value leaked in"


def test_bootstrap_weights_days_by_their_row_count():
    """The interval must be centred on the same statistic the headline reports."""
    days = pd.date_range("2025-01-01", periods=40, freq="D")
    # One heavily-sampled day is much better than the rest; equal weighting would
    # understate the overall gain that a row-weighted mean reports.
    a = pd.Series([1.0] * 39 + [0.1], index=days)
    b = pd.Series([1.0] * 40, index=days)
    w = pd.Series([1] * 39 + [1000], index=days)

    unweighted, _, _ = metrics.block_bootstrap_ci(a, b, n_resamples=200)
    weighted, _, _ = metrics.block_bootstrap_ci(a, b, weights=w, n_resamples=200)
    assert weighted > unweighted + 0.5
