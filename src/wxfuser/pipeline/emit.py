"""Static JSON writers — the only contract between the Python pipeline and the website.

Everything the site can show is decided here. Two principles:

  * **Publish the baseline next to the claim.** Every forecast ships with the raw model's
    own numbers and the measured skill difference including its confidence interval. A
    reader can always check the improvement rather than take it on faith, and when the
    interval straddles zero the site says so rather than rounding up to a win.
  * **Say which method produced this.** The tier, how much history it was trained on, and
    whether the intervals are trustworthy yet all travel with the forecast, so a
    two-day-old station is never presented with the same confidence as a two-year-old one.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.config import quantiles
from wxfuser.models.select import label_for

SCHEMA_VERSION = 1


def _clean(value):
    """JSON-safe scalars: NaN/inf become null rather than invalid JSON."""
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _series(arr) -> list:
    return [_clean(v) for v in np.asarray(arr).tolist()]


def forecast_json(
    station: dict,
    models: list[str],
    fc: pd.DataFrame,
    predictions: dict[str, dict],
    methods: dict[str, dict],
    skill: dict[str, dict],
) -> dict:
    """Assemble the per-station forecast payload.

    ``predictions`` maps variable -> quantile arrays aligned with ``fc``'s valid_time
    ordering; ``fc`` is the wide live-forecast frame (one row per valid_time).
    """
    times = pd.to_datetime(fc["valid_time"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "station": _clean(station),
        "models_used": models,
        "method": _clean(methods),
        "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%MZ") for t in times]},
        "raw": {},
        "skill": _clean(skill),
    }

    qkeys = [f"q{int(q * 100):02d}" for q in quantiles()]
    for var, pred in predictions.items():
        block = {}
        for qk in qkeys:
            if qk in pred:
                block[qk] = _series(pred[qk])
        if "p_occ" in pred:
            block["p_occ"] = _series(pred["p_occ"])
        block["calibrated"] = bool(pred.get("calibrated", False))
        payload["hourly"][var] = block

    for model in models:
        col_block = {}
        for var in predictions:
            col = f"fc_{var}__{model}"
            if col in fc:
                col_block[var] = _series(fc[col])
        if col_block:
            payload["raw"][model] = col_block

    return payload


def write_json(payload: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    return p


def index_json(entries: list[dict]) -> dict:
    """Site-wide index: every enrolled station with its headline skill number.

    This is what the map colours itself by, so it stays small — no time series, just the
    single number and enough metadata to place a marker.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stations": _clean(entries),
    }


def method_block(tier: str, wide: pd.DataFrame, calibrated: bool, models: list[str]) -> dict:
    from wxfuser.models.select import training_days

    return {
        "tier": tier,
        "label": label_for(tier),
        "train_days": round(training_days(wide), 1) if wide is not None else 0.0,
        "n_pairs": int(len(wide)) if wide is not None else 0,
        "calibrated": bool(calibrated),
        "models": models,
    }


def skill_block(scorecard: dict) -> dict:
    """Condense a verification scorecard into the headline numbers the forecast carries.

    ``beats_raw`` is deliberately conservative: it is true only when the bootstrap
    interval on the gain lies entirely above zero.
    """
    if not scorecard or scorecard.get("insufficient_data"):
        return {"status": "warming_up", "reason": "not enough verified pairs yet"}
    ci = scorecard.get("crpss_vs_raw_ci90") or [None, None]
    return {
        "status": "verified",
        "crps": scorecard.get("crps"),
        "crps_raw_best": scorecard.get("crps_raw_best"),
        "raw_best_model": scorecard.get("raw_best_model"),
        "crpss_vs_raw": scorecard.get("crpss_vs_raw"),
        "crpss_vs_raw_ci90": ci,
        "beats_raw": scorecard.get("beats_raw", False),
        "crpss_vs_climo": scorecard.get("crpss_vs_climo"),
        # The deterministic comparison travels with the probabilistic one: a CRPS win
        # next to an MAE loss means the gain is in the uncertainty, not the central
        # estimate, and the site says so rather than quoting the flattering number.
        "mae_median": scorecard.get("mae_median"),
        "mae_raw_best": scorecard.get("mae_raw_best"),
        "mae_gain_vs_raw": scorecard.get("mae_gain_vs_raw"),
        "coverage50": scorecard.get("coverage50"),
        "coverage90": scorecard.get("coverage90"),
        "evaluation_scheme": scorecard.get("evaluation_scheme"),
        "n_eval": scorecard.get("n_eval"),
    }
