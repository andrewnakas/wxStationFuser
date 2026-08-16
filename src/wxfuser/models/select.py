"""Champion/challenger selection — which tier actually gets published.

Every tier is eligible only once the station has enough paired history to fit it
honestly, and eligibility is not the same as being better. A station with three years of
data can still be served best by plain EMOS; the literature is clear that added model
complexity buys little for temperature at a well-behaved site. So we do not assume the
highest eligible tier wins — we measure.

Two rules keep this stable:

  * **Out-of-sample only.** Tiers are compared on rolling-origin CRPS, never on the data
    they were fitted to, because a more flexible model always wins in-sample.
  * **Hysteresis.** A challenger must beat the champion by a margin (default 2%) to take
    over. Without it, two near-equivalent tiers trade places every refresh and the
    published method flaps for no reason a user could understand.
"""
from __future__ import annotations

import pandas as pd

from wxfuser.config import load_configs

TIER_ORDER = ["tier0", "tier1", "tier2", "tier3"]

TIER_LABELS = {
    "tier0": "Adaptive bias correction",
    "tier1": "EMOS (min-CRPS)",
    "tier2": "EMOS + seasonal/diurnal harmonics",
    "tier3": "Gradient-boosted quantile regression",
}


def training_days(wide: pd.DataFrame) -> float:
    """Span of paired history in days — the gate for tier eligibility."""
    if wide.empty:
        return 0.0
    vt = pd.to_datetime(wide["valid_time"])
    return float((vt.max() - vt.min()).total_seconds() / 86400.0)


def eligible_tiers(wide: pd.DataFrame) -> list[str]:
    """Tiers whose data requirements this station meets."""
    cfg = load_configs()["tiers"]
    days = training_days(wide)
    n = len(wide)
    out = ["tier0"]
    if days >= cfg["tier1"]["min_days"] and n >= 200:
        out.append("tier1")
    if days >= cfg["tier2"]["min_days"] and n >= 1000:
        out.append("tier2")
    if days >= cfg["tier3"]["min_days"] and n >= 4000:
        from wxfuser.models import tier3_gbm

        # LightGBM is an optional dependency; without it this tier simply never competes
        # rather than failing the run.
        if tier3_gbm.available():
            out.append("tier3")
    return out


def choose(
    scores: dict,
    eligible: list[str],
    previous_champion: str | None = None,
) -> tuple[str, dict]:
    """Pick the champion from rolling-origin scores.

    ``scores`` maps ``crps_{tier}`` -> mean out-of-sample CRPS (from
    ``verify.rolling.rolling_origin_scores``). Returns the champion and a record of the
    comparison for publication, so a user can see why a method was chosen.
    """
    margin = float(load_configs()["tiers"]["select"]["promote_margin"])

    available = {
        t: scores.get(f"crps_{t}")
        for t in eligible
        if scores.get(f"crps_{t}") is not None and scores.get(f"crps_{t}") == scores.get(f"crps_{t}")
    }
    if not available:
        return "tier0", {"reason": "no tier produced an out-of-sample score", "scores": {}}

    best_tier = min(available, key=available.get)
    best_score = available[best_tier]

    champion = best_tier
    reason = "lowest out-of-sample CRPS"
    if previous_champion and previous_champion in available and previous_champion != best_tier:
        incumbent = available[previous_champion]
        # Only unseat the incumbent on a clear margin.
        if incumbent > 0 and (incumbent - best_score) / incumbent < margin:
            champion = previous_champion
            reason = (
                f"{best_tier} led by less than the {margin:.0%} promotion margin; "
                f"kept {previous_champion}"
            )

    return champion, {
        "reason": reason,
        "scores": {k: float(v) for k, v in available.items()},
        "raw_best": scores.get("crps_raw_best"),
        "raw_best_model": scores.get("raw_best_model"),
        "promote_margin": margin,
    }


def label_for(tier: str, wide: pd.DataFrame | None = None) -> str:
    base = TIER_LABELS.get(tier, tier)
    if wide is not None and not wide.empty:
        return f"{base} · trained on {training_days(wide):.0f} d"
    return base
