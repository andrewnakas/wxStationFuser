"""Tier behaviour on synthetic data with a known answer.

Real forecast data cannot tell you whether a method is correct, only whether it looks
plausible. These tests generate data from a process whose parameters we chose, so
"did the fit recover the truth" has an actual answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from wxfuser.models import select, tier0_bias, tier1_emos, tier2
from wxfuser.verify import metrics


def synthetic_pairs(
    n_days: int = 200,
    *,
    bias: float = 2.0,
    slope: float = 0.9,
    spread_effect: float = 0.5,
    seed: int = 11,
    models=("m1", "m2", "m3"),
) -> pd.DataFrame:
    """Forecasts and observations from a known EMOS process.

    obs = intercept + slope * mean_forecast + noise, where the noise scale grows with
    inter-model disagreement — exactly the structure EMOS claims to model, so a correct
    implementation should recover it.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    truth = 10.0 + 8.0 * np.sin(2 * np.pi * np.arange(n) / (24 * 365)) + \
        4.0 * np.sin(2 * np.pi * np.arange(n) / 24)

    spread = np.abs(rng.normal(1.0, 0.4, size=n))
    cols = {}
    for i, m in enumerate(models):
        cols[f"fc_{m}"] = truth + bias + rng.normal(0, spread * (0.8 + 0.2 * i))

    fc = np.column_stack([cols[f"fc_{m}"] for m in models])
    fc_mean = fc.mean(axis=1)
    fc_spread = fc.std(axis=1, ddof=0)
    noise_sd = 1.0 + spread_effect * fc_spread
    obs = 0.0 + slope * fc_mean + rng.normal(0, noise_sd) - slope * bias

    df = pd.DataFrame(
        {
            "valid_time": times,
            "lead_h": np.tile(np.arange(1, 25), n_days)[:n],
            "lead_source": "prev_runs",
            "obs": obs,
            "fc_mean": fc_mean,
            "fc_spread": fc_spread,
            **cols,
        }
    )
    return df


MODELS = ["m1", "m2", "m3"]


# ------------------------------------------------------------------ tier 0


def test_tier0_learns_a_constant_bias():
    """The point of Tier 0: remove a systematic offset within a couple of weeks."""
    df = synthetic_pairs(n_days=60, slope=1.0, spread_effect=0.0)
    df["obs"] = df["fc_mean"] - 3.0  # model runs exactly 3 degrees warm
    state = tier0_bias.fit(df, "air_temp_c")
    learned = np.mean(list(state["bias_lead"].values()))
    assert learned == pytest.approx(3.0, abs=0.3)


def test_tier0_corrects_the_median():
    df = synthetic_pairs(n_days=60, slope=1.0, spread_effect=0.0)
    df["obs"] = df["fc_mean"] - 3.0
    state = tier0_bias.fit(df, "air_temp_c")
    pred = tier0_bias.predict(
        state, df["fc_mean"].to_numpy(), df["lead_h"].to_numpy(), df["valid_time"]
    )
    err_before = np.abs(df["obs"] - df["fc_mean"]).mean()
    err_after = np.abs(df["obs"] - pred["q50"]).mean()
    assert err_after < err_before * 0.2


def test_tier0_does_not_shift_precipitation():
    """An additive bias on a mostly-zero variable smears dry hours; it must be skipped."""
    df = synthetic_pairs(n_days=60)
    df["obs"] = np.where(np.random.default_rng(1).random(len(df)) < 0.9, 0.0, 2.0)
    df["fc_mean"] = 0.3
    state = tier0_bias.fit(df, "precip_1h_mm")
    assert all(abs(v) < 1e-9 for v in state["bias_lead"].values())


def test_tier0_quantiles_stay_non_negative_and_ordered():
    df = synthetic_pairs(n_days=60)
    df["obs"] = np.clip(df["obs"], 0, None) * 0.1
    df["fc_mean"] = np.clip(df["fc_mean"], 0, None) * 0.1
    state = tier0_bias.fit(df, "wind_speed_ms")
    pred = tier0_bias.predict(
        state, df["fc_mean"].to_numpy(), df["lead_h"].to_numpy(), df["valid_time"]
    )
    stack = np.column_stack([pred[f"q{q:02d}"] for q in (5, 25, 50, 75, 95)])
    assert (stack >= 0).all()
    assert (np.diff(stack, axis=1) >= -1e-9).all()


# ------------------------------------------------------------------ tier 1


def test_emos_recovers_known_coefficients():
    df = synthetic_pairs(n_days=300, slope=0.9, spread_effect=0.5)
    state = tier1_emos.fit(df, "air_temp_c", MODELS)
    assert state["buckets"], "EMOS produced no fitted buckets"
    par = next(iter(state["buckets"].values()))
    # The three model columns are near-identical, so their coefficients share the slope.
    assert sum(par["b"]) == pytest.approx(0.9, abs=0.12)
    # Spread genuinely predicts error here, so the fit must give it positive weight.
    assert par["d"] > 0.05


def test_emos_is_calibrated_on_data_it_understands():
    """PIT uniformity is the real test of a probabilistic forecast, not its mean error."""
    df = synthetic_pairs(n_days=400, seed=5)
    train, test = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    state = tier1_emos.fit(train, "air_temp_c", MODELS)
    pred = tier1_emos.predict(state, test, MODELS)
    pit = metrics.pit_values(test["obs"].to_numpy(), pred)
    # Kolmogorov-Smirnov against uniform; a miscalibrated forecast fails this badly.
    assert stats.kstest(pit, "uniform").statistic < 0.08


def test_emos_beats_the_raw_model_out_of_sample():
    """The product claim, on data where the gain is guaranteed to exist."""
    df = synthetic_pairs(n_days=300, bias=2.5, slope=0.9)
    train, test = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    state = tier1_emos.fit(train, "air_temp_c", MODELS)
    pred = tier1_emos.predict(state, test, MODELS)
    y = test["obs"].to_numpy()
    fused = metrics.crps_from_quantiles(y, pred).mean()
    raw = np.abs(y - test["fc_mean"].to_numpy()).mean()
    assert fused < raw


def test_emos_clamps_relative_humidity_to_100():
    df = synthetic_pairs(n_days=200)
    df["obs"] = np.clip(df["obs"] * 4 + 60, 0, 100)
    for m in MODELS:
        df[f"fc_{m}"] = np.clip(df[f"fc_{m}"] * 4 + 60, 0, 120)
    state = tier1_emos.fit(df, "rh_pct", MODELS)
    pred = tier1_emos.predict(state, df, MODELS)
    assert pred["q95"].max() <= 100.0 + 1e-9
    assert pred["q05"].min() >= 0.0 - 1e-9


def test_precip_occurrence_tracks_the_models():
    """Two-part precipitation: wetter model input must raise the occurrence probability."""
    rng = np.random.default_rng(4)
    df = synthetic_pairs(n_days=200)
    wet = rng.random(len(df)) < 0.25
    for m in MODELS:
        df[f"fc_{m}"] = np.where(wet, rng.gamma(2.0, 0.8, len(df)), 0.0)
    df["fc_mean"] = df[[f"fc_{m}" for m in MODELS]].mean(axis=1)
    df["obs"] = np.where(wet & (rng.random(len(df)) < 0.8), rng.gamma(2.0, 0.9, len(df)), 0.0)

    state = tier1_emos.fit(df, "precip_1h_mm", MODELS)
    pred = tier1_emos.predict(state, df, MODELS)
    p = pred["p_occ"]
    assert p[df["fc_mean"] > 0.5].mean() > p[df["fc_mean"] <= 0.1].mean() + 0.2
    assert (pred["q50"] >= 0).all()


# ------------------------------------------------------------------ tier 2


def test_tier2_captures_a_diurnal_error_cycle():
    """Tier 2 exists for error that varies with time of day; give it exactly that."""
    df = synthetic_pairs(n_days=400, slope=1.0, spread_effect=0.0)
    hod = pd.to_datetime(df["valid_time"]).dt.hour.to_numpy()
    df["obs"] = df["fc_mean"] + 3.0 * np.sin(2 * np.pi * hod / 24)

    train, test = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    s1 = tier1_emos.fit(train, "air_temp_c", MODELS)
    s2 = tier2.fit(train, "air_temp_c", MODELS)
    y = test["obs"].to_numpy()
    c1 = metrics.crps_from_quantiles(y, tier1_emos.predict(s1, test, MODELS)).mean()
    c2 = metrics.crps_from_quantiles(y, tier2.predict(s2, test, MODELS)).mean()
    assert c2 < c1


# ------------------------------------------------------------------ selection


def test_selection_prefers_the_lowest_out_of_sample_score():
    scores = {"crps_tier0": 1.4, "crps_tier1": 1.0, "crps_tier2": 1.2}
    champion, _ = select.choose(scores, ["tier0", "tier1", "tier2"])
    assert champion == "tier1"


def test_selection_keeps_the_incumbent_on_a_narrow_margin():
    """Hysteresis: a trivial lead must not flip the published method every refresh."""
    scores = {"crps_tier1": 1.000, "crps_tier2": 0.995}  # 0.5% better, margin is 2%
    champion, decision = select.choose(scores, ["tier1", "tier2"], previous_champion="tier1")
    assert champion == "tier1"
    assert "margin" in decision["reason"]


def test_selection_switches_on_a_clear_margin():
    scores = {"crps_tier1": 1.00, "crps_tier2": 0.80}
    champion, _ = select.choose(scores, ["tier1", "tier2"], previous_champion="tier1")
    assert champion == "tier2"


def test_tier_eligibility_grows_with_history():
    short = synthetic_pairs(n_days=10)
    assert select.eligible_tiers(short) == ["tier0"]
    medium = synthetic_pairs(n_days=40)
    assert "tier1" in select.eligible_tiers(medium)
    assert "tier2" not in select.eligible_tiers(medium)
    long = synthetic_pairs(n_days=200)
    assert "tier2" in select.eligible_tiers(long)


def test_tier2_drops_annual_harmonics_it_cannot_identify():
    """Fitting a 365-day cycle to two months of data extrapolates catastrophically.

    This was observed in production data: Tier 2 scored CRPS 7.5 against Tier 1's 1.28
    because every lead bucket fitted an unconstrained annual sine through a 54-day arc.
    """
    short = synthetic_pairs(n_days=60, slope=1.0, spread_effect=0.0)
    state = tier2.fit(short, "air_temp_c", MODELS)
    for par in state["buckets"].values():
        assert par["harmonics"]["doy"] == 0, "annual harmonic fitted to a short window"
        assert par["harmonics"]["hod"] > 0, "diurnal harmonic needs only a day of data"


def test_tier2_keeps_annual_harmonics_when_the_weighting_can_see_the_year():
    """Rewritten, and the reason matters.

    This test used to assert that 400 days of history keeps the annual harmonics, on the
    reasoning that a long window identifies an annual cycle. It does not: with a 30-day
    time constant, 90% of the fitting weight in that same 400-day fixture sits inside 88
    days. The harmonics it asserted were being fitted from a season and extrapolated over
    the year, and measured on production data that cost up to 24% of the CRPS — the
    numbers are in the README and in configs/tiers.yaml.

    So the rule is no longer "a long window" but "a window the weighting can actually
    see", and the mechanism is still here: lengthen the time constant and the annual
    terms come back.
    """
    long = synthetic_pairs(n_days=400, slope=1.0, spread_effect=0.0)

    state = tier2.fit(long, "air_temp_c", MODELS)
    assert all(p["harmonics"]["doy"] == 0 for p in state["buckets"].values()), (
        "a 30-day time constant cannot support an annual cycle"
    )

    from wxfuser.config import load_configs

    cfg = load_configs()["tiers"]["tier1"]
    before = (cfg["tau_days_short"], cfg["tau_days_long"])
    cfg["tau_days_short"], cfg["tau_days_long"] = 400.0, 400.0
    try:
        state = tier2.fit(long, "air_temp_c", MODELS)
        assert any(p["harmonics"]["doy"] > 0 for p in state["buckets"].values()), (
            "with the weighting spread over the whole window they are identifiable again"
        )
    finally:
        cfg["tau_days_short"], cfg["tau_days_long"] = before


def test_tier2_never_collapses_against_tier1_on_short_history():
    """The guard that matters: Tier 2 must not be catastrophically worse than Tier 1."""
    df = synthetic_pairs(n_days=90, slope=0.95, spread_effect=0.3, seed=21)
    train, test = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    y = test["obs"].to_numpy()
    c1 = metrics.crps_from_quantiles(
        y, tier1_emos.predict(tier1_emos.fit(train, "air_temp_c", MODELS), test, MODELS)
    ).mean()
    c2 = metrics.crps_from_quantiles(
        y, tier2.predict(tier2.fit(train, "air_temp_c", MODELS), test, MODELS)
    ).mean()
    assert c2 < c1 * 1.5


def test_the_weighted_gate_sees_through_a_deep_archive():
    """Calendar span says a year; the weights say one season.

    Measured on two years of grid history: buckets spanned 253-412 days and passed the
    calendar gate, while 90% of the exponential weight sat inside the most recent 68-135
    days. Four annual coefficients per bucket were fitted from that and extrapolated
    across the rest of the year, costing up to 49% of the CRPS.
    """
    import numpy as np
    import pandas as pd

    from wxfuser.models.tier2 import weighted_span_days

    times = pd.Series(pd.date_range("2025-01-01", periods=400, freq="D"))
    ref = times.max()
    age = (ref - times).dt.total_seconds() / 86400.0
    weights = np.exp(-age / 60.0)

    assert (times.max() - times.min()).days == 399
    span = weighted_span_days(times, weights)
    assert span < 200, f"weighted span {span:.0f} d should not look like a year"

    # Flat weights are the degenerate case, and there the two agree.
    assert weighted_span_days(times, np.ones(len(times))) > 350


def test_the_annual_harmonic_gate_defaults_to_the_measured_choice():
    """`weighted` won 17 of 18 comparisons on production short-lead data.

    Pinned because the alternative is not a preference: `calendar` cost up to 24% of the
    CRPS on Seattle's winter wind and 18% on its summer temperature, and a silent revert
    would put that back without anything failing.
    """
    from wxfuser.config import load_configs

    assert load_configs()["tiers"]["tier2"]["harmonic_gate"] == "weighted"
