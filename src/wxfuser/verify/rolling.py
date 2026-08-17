"""Walk-forward verification — how we earn the right to claim a skill gain.

The only defensible way to score an adaptive system is to score it the way it is used:
fit on the past, predict the future, refit, repeat. This module walks an origin through
the paired archive, refits on everything strictly before the origin, scores the following
block, and accumulates the out-of-sample predictions.

The refit cadence matters more than it looks. An earlier version of this file fitted once
on the first 75% of history and scored the last 25%; the fused median came out *worse*
than the raw model for temperature, because a training window weighted toward spring was
being asked to predict August with no refitting in between. That is not a property of
EMOS — it is a property of never updating it, and the deployed system refits every few
hours. Scoring with periodic refits removes an artefact of the evaluation, not a real
error, and it is still strictly out-of-sample: no fit ever sees the block it is scored on.

Everything published comes from the accumulated walk-forward predictions, so the numbers
on the site describe the system as operated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wxfuser.config import bucket_for_lead, bucket_labels, load_configs
from wxfuser.models import tier0_bias, tier1_emos, tier2, tier3_gbm
from wxfuser.verify import metrics

# Variables with a point mass (dry hours) need randomised PIT to be interpretable.
POINT_MASS_VARIABLES = {"precip_1h_mm"}
PRECIP_VARIABLE = "precip_1h_mm"
# The wet/dry threshold the occurrence model is fitted at, from tier1_emos.fit_precip.
OCCURRENCE_THRESHOLD = 0.1


def _is_occurrence_threshold(thr: float) -> bool:
    return abs(float(thr) - OCCURRENCE_THRESHOLD) < 1e-9


# Share of the walk-forward output used to choose the method; the remainder scores it.
SELECTION_FRACTION = 0.6

# Days of per-day scores published per variable; the rest is summarised in the stats.
DAILY_SERIES_DAYS = 90


def _rank_tiers(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict, bool]:
    """Score each tier, preferring the rows every candidate covered.

    Tiers do not all produce a forecast for every block — Tier 3 needs a couple of
    thousand rows before it can fit at all, so it sits out the early ones. Ranking them on
    whatever each happened to cover would compare them on different weather.
    """
    common = None
    for frame in frames.values():
        keys = set(zip(frame["valid_time"], frame["lead_h"]))
        common = keys if common is None else (common & keys)

    scores: dict[str, float] = {}
    scores_all: dict[str, float] = {}
    for tier, frame in frames.items():
        if frame.empty:
            continue
        qc_all = {k: frame[k].to_numpy() for k in frame if k.startswith("q")}
        y_all = frame["obs"].to_numpy(dtype=float)
        scores_all[tier] = float(np.nanmean(metrics.crps_from_quantiles(y_all, qc_all)))

        if common:
            mask = np.array(
                [(t, ll) in common for t, ll in zip(frame["valid_time"], frame["lead_h"])]
            )
            sub = frame[mask]
            if len(sub) >= 50:
                y = sub["obs"].to_numpy(dtype=float)
                qc = {k: sub[k].to_numpy() for k in sub if k.startswith("q")}
                scores[tier] = float(np.nanmean(metrics.crps_from_quantiles(y, qc)))

    # Every candidate must be scored on the shared rows for the ranking to mean anything.
    # If some tier could not be, fall back to its own coverage — but say so, because that
    # is the very "compared on different weather" bias this exists to avoid.
    comparable = bool(scores) and set(scores) == set(scores_all)
    if not comparable:
        for tier, score in scores_all.items():
            scores.setdefault(tier, score)
    return scores, scores_all, comparable

FIT = {
    "tier0": lambda w, v, m: tier0_bias.fit(w, v),
    "tier1": lambda w, v, m: tier1_emos.fit(w, v, m),
    "tier2": lambda w, v, m: tier2.fit(w, v, m),
    "tier3": lambda w, v, m: tier3_gbm.fit(w, v, m),
}

PREDICT = {
    "tier0": lambda st, fc, m: tier0_bias.predict(
        st, fc["fc_mean"].to_numpy(), fc["lead_h"].to_numpy(), fc["valid_time"]
    ),
    "tier1": lambda st, fc, m: tier1_emos.predict(st, fc, m),
    "tier2": lambda st, fc, m: tier2.predict(st, fc, m),
    "tier3": lambda st, fc, m: tier3_gbm.predict(st, fc, m),
}


def _has_fit(state: dict | None, tier: str) -> bool:
    if not state:
        return False
    if tier == "tier0":
        return bool(state.get("resid_q"))
    if tier == "tier3":
        return bool(state.get("boosters"))
    return bool(state.get("buckets"))


def walk_forward(
    wide: pd.DataFrame,
    variable: str,
    models: list[str],
    tiers: tuple[str, ...],
    *,
    refit_every_days: int = 14,
    min_train_days: int = 21,
    max_refits: int = 30,
) -> dict[str, pd.DataFrame]:
    """Accumulate out-of-sample predictions for each tier by walking the archive.

    Returns tier -> frame of (valid_time, lead_h, obs, q05..q95[, p_occ]) covering every
    block that was predicted before being seen.
    """
    if wide.empty:
        return {}
    df = wide.sort_values("valid_time").reset_index(drop=True)
    t0 = pd.to_datetime(df["valid_time"]).min()
    t1 = pd.to_datetime(df["valid_time"]).max()
    span = (t1 - t0).total_seconds() / 86400.0
    if span < min_train_days + refit_every_days:
        return {}

    # Keep the number of refits bounded so a multi-year station does not spend an hour
    # in verification; widen the block instead.
    n_blocks = int((span - min_train_days) // refit_every_days)
    if n_blocks > max_refits:
        refit_every_days = int(np.ceil((span - min_train_days) / max_refits))
        n_blocks = max_refits

    collected: dict[str, list[pd.DataFrame]] = {t: [] for t in tiers}
    origin = t0 + pd.Timedelta(days=min_train_days)
    while origin < t1:
        end = origin + pd.Timedelta(days=refit_every_days)
        train = df[pd.to_datetime(df["valid_time"]) < origin]
        test = df[
            (pd.to_datetime(df["valid_time"]) >= origin)
            & (pd.to_datetime(df["valid_time"]) < end)
        ]
        if len(train) >= 100 and len(test) >= 5:
            for tier in tiers:
                try:
                    state = FIT[tier](train, variable, models)
                    if not _has_fit(state, tier):
                        continue
                    pred = PREDICT[tier](state, test, models)
                except Exception as exc:  # noqa: BLE001
                    print(f"    WARN: {tier}/{variable} @ {origin.date()} failed ({exc})",
                          flush=True)
                    continue
                if not pred or not pred.get("calibrated"):
                    continue
                block = test[["valid_time", "lead_h", "obs"]].copy()
                for k, v in pred.items():
                    if k.startswith("q") or k == "p_occ":
                        block[k] = np.asarray(v)
                collected[tier].append(block)
        origin = end

    return {
        t: pd.concat(frames, ignore_index=True)
        for t, frames in collected.items()
        if frames
    }


def _raw_baselines(
    oos: pd.DataFrame,
    wide: pd.DataFrame,
    models: list[str],
    crps_fused: np.ndarray,
    mae_fused: np.ndarray,
) -> dict:
    """Score each raw model against the fused forecast on *exactly the same rows*.

    This is subtler than it looks, and getting it wrong flatters or slanders the system
    badly. Regional models are the trap: HRRR covers North America out to 48 hours and
    returns nothing beyond, so averaging its error over the rows where it exists compares
    it only on short leads — the easy ones — while the fused forecast is being averaged
    over everything out to seven days. An earlier version did exactly that and made the
    fused median look 15% worse than a model it in fact beat in every lead bucket.

    So each model is scored on its own coverage, and the fused forecast is re-scored on
    that same subset. ``coverage`` is published too, so a model that only competes on
    short leads is visibly doing so.
    """
    key = ["valid_time", "lead_h"]
    # `wide` is keyed on (valid_time, lead_h, lead_source), so two archive sources could
    # in principle contribute the same valid time at the same nominal lead. A left merge
    # would then return more rows than `oos`, and every per-row array below — including
    # crps_fused — would be silently misaligned against it, corrupting the published
    # skill numbers rather than failing. Today the sources cannot collide (the seamless
    # archive is pinned to lead 3, previous-runs uses multiples of 24), but the
    # comparison must not depend on that holding.
    lookup = wide.drop_duplicates(subset=key, keep="last")
    joined = oos[key].merge(lookup, on=key, how="left")
    if len(joined) != len(oos):  # pragma: no cover - defensive
        raise AssertionError(
            f"baseline merge changed row count ({len(oos)} -> {len(joined)}); "
            "the fused and raw scores would not correspond to the same hours"
        )
    y = joined["obs"].to_numpy(dtype=float)
    n = len(y)
    out = {}
    for m in models:
        col = f"fc_{m}"
        if col not in joined:
            continue
        v = joined[col].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        out[m] = {
            "crps": float(np.mean(np.abs(y[ok] - v[ok]))),
            "mae": float(np.mean(np.abs(y[ok] - v[ok]))),
            # The fused numbers restricted to this model's rows — the only fair pairing.
            "crps_fused_here": float(np.nanmean(crps_fused[ok])),
            "mae_fused_here": float(np.nanmean(mae_fused[ok])),
            "coverage": float(ok.sum() / max(n, 1)),
            "n": int(ok.sum()),
            "values": v,
            "ok": ok,
        }
    return out


def _pick_baseline(raw: dict, min_coverage: float = 0.9) -> str | None:
    """Choose the headline baseline: the strongest model that covers the whole forecast.

    Preferring full-coverage models keeps the headline claim honest — beating a
    short-range model only on short range is not the claim being made. If no model
    covers enough, fall back to the best available and let ``coverage`` speak.
    """
    if not raw:
        return None
    full = {m: r for m, r in raw.items() if r["coverage"] >= min_coverage}
    pool = full or raw
    return min(pool, key=lambda m: pool[m]["crps"])


def _randomized_pit(y: np.ndarray, pred: dict, p_occ: np.ndarray | None) -> np.ndarray:
    """PIT for a distribution with a point mass at zero.

    For dry observations the predictive CDF jumps from 0 to P(dry), so any single value
    in that range is consistent with calibration; drawing uniformly within the jump is
    the standard randomised PIT (Gneiting & Ranjan 2013). Without this, a perfectly
    calibrated precipitation forecast produces a histogram spiked at 1 and looks broken.
    """
    rng = np.random.default_rng(7)
    pit = metrics.pit_values(y, pred)
    if p_occ is None or len(pit) != len(y):
        return pit
    p_dry = 1.0 - np.clip(p_occ, 0.0, 1.0)
    # "Dry" must mean the same thing here as it does to the occurrence model, whose
    # probability is P(y > OCCURRENCE_THRESHOLD). Testing y <= 0 instead would treat a
    # trace observation as wet while the probability it is scored against calls it dry.
    dry = y <= OCCURRENCE_THRESHOLD
    pit = pit.copy()
    pit[dry] = rng.uniform(0.0, 1.0, size=int(dry.sum())) * p_dry[dry]
    return pit


def scorecard(
    wide: pd.DataFrame,
    variable: str,
    models: list[str],
    tiers: tuple[str, ...] = ("tier0", "tier1", "tier2"),
    obs_history: pd.DataFrame | None = None,
    *,
    refit_every_days: int = 14,
) -> dict:
    """Full walk-forward verification for one variable, plus the tier comparison.

    Reports both the probabilistic score (CRPS) and the deterministic one (MAE of the
    published median). Both are published because they answer different questions, and
    because a CRPS win alongside an MAE loss means the gain is coming from expressing
    uncertainty rather than from a better central estimate — which a reader deserves to
    know rather than have averaged away.
    """
    cfg = load_configs()["tiers"]["verify"]
    if wide.empty or len(wide) < 200:
        return {"insufficient_data": True, "n_pairs": int(len(wide))}

    oos_by_tier = walk_forward(
        wide, variable, models, tiers, refit_every_days=refit_every_days
    )
    if not oos_by_tier:
        return {"insufficient_data": True, "n_pairs": int(len(wide)),
                "reason": "not enough history for a walk-forward evaluation"}

    # Picking the best of several candidates on a set of rows and then quoting that
    # candidate's score on the *same* rows is optimistically biased: the selection step is
    # not itself out-of-sample, so the winner carries whatever luck won it the comparison.
    # With near-tied tiers that alone can flip `beats_raw`. So the walk-forward output is
    # split in time — the earlier part chooses the method, the later part scores it — and
    # both halves remain strictly out-of-sample with respect to fitting.
    all_times = np.sort(
        np.unique(np.concatenate([f["valid_time"].to_numpy() for f in oos_by_tier.values()]))
    )
    split_at = all_times[int(len(all_times) * SELECTION_FRACTION)] if len(all_times) > 20 else None

    def _slice(frame: pd.DataFrame, early: bool) -> pd.DataFrame:
        if split_at is None:
            return frame
        t = frame["valid_time"].to_numpy()
        return frame[t < split_at] if early else frame[t >= split_at]

    select_frames = {t: _slice(f, True) for t, f in oos_by_tier.items()}
    report_frames = {t: _slice(f, False) for t, f in oos_by_tier.items()}
    # If the split leaves either side too thin to be meaningful, score on everything and
    # record that selection and reporting shared rows.
    holdout = split_at is not None and all(
        len(select_frames[t]) >= 100 and len(report_frames[t]) >= 100 for t in oos_by_tier
    )
    if not holdout:
        select_frames = report_frames = oos_by_tier

    tier_scores, tier_scores_all, comparable = _rank_tiers(select_frames)
    champion = min(tier_scores, key=tier_scores.get)
    oos = report_frames[champion]
    y = oos["obs"].to_numpy(dtype=float)
    qcols = {k: oos[k].to_numpy() for k in oos if k.startswith("q")}
    crps = metrics.crps_from_quantiles(y, qcols)
    day = pd.to_datetime(oos["valid_time"]).dt.floor("D")
    daily_fused = pd.DataFrame({"day": day.to_numpy(), "c": crps}).groupby("day")["c"].mean()

    out: dict = {
        "variable": variable,
        "n_pairs": int(len(wide)),
        "n_eval": int(len(oos)),
        "evaluation_scheme": (
            f"walk-forward, refit every {refit_every_days} d, "
            "each block predicted before being seen"
        ),
        "champion": champion,
        # Scored on the rows every tier covered, so the ranking is a like-for-like one.
        "tier_crps": tier_scores,
        "tier_crps_full_coverage": tier_scores_all,
        "tier_comparison_like_for_like": comparable,
        # True when the method was chosen on rows it was not then scored on.
        "selection_held_out": bool(holdout),
        "crps": float(np.nanmean(crps)),
        "mae_median": float(np.nanmean(np.abs(y - oos["q50"].to_numpy()))),
    }

    # Raw-model baselines, each scored against the fused forecast on its own rows.
    mae_rows = np.abs(y - oos["q50"].to_numpy())
    raw = _raw_baselines(oos, wide, models, crps, mae_rows)
    daily_raw = None
    if raw:
        out["crps_raw_by_model"] = {
            m: {
                "crps": r["crps"],
                "crps_fused_same_rows": r["crps_fused_here"],
                "crpss": metrics.crpss(r["crps_fused_here"], r["crps"]),
                "coverage": r["coverage"],
                "n": r["n"],
            }
            for m, r in raw.items()
        }
        best = _pick_baseline(raw)
        rb = raw[best]
        out["raw_best_model"] = best
        out["raw_best_coverage"] = rb["coverage"]
        out["crps_raw_best"] = rb["crps"]
        out["mae_raw_best"] = rb["mae"]
        # Compare fused-on-these-rows with raw-on-these-rows, never fused-on-everything.
        out["crps_fused_vs_best"] = rb["crps_fused_here"]
        out["crpss_vs_raw"] = metrics.crpss(rb["crps_fused_here"], rb["crps"])
        out["mae_gain_vs_raw"] = (
            float(1.0 - rb["mae_fused_here"] / rb["mae"]) if rb["mae"] > 0 else None
        )

        err = np.abs(y[rb["ok"]] - rb["values"][rb["ok"]])
        paired = pd.DataFrame(
            {
                "day": day.to_numpy()[rb["ok"]],
                "raw": err,
                "fused": crps[rb["ok"]],
            }
        )
        grouped = paired.groupby("day")
        daily_raw = grouped["raw"].mean()
        daily_fused_here = grouped["fused"].mean()
        # Weight each day by how many hours it contributed, matching how the point
        # estimate averages, so the interval brackets the number it is quoted against.
        daily_weight = grouped.size()
        point, lo, hi = metrics.block_bootstrap_ci(
            daily_fused_here, daily_raw, weights=daily_weight
        )
        out["crpss_vs_raw_ci90"] = [lo, hi]
        # Only a bootstrap interval entirely above zero counts as a win.
        out["beats_raw"] = bool(np.isfinite(lo) and lo > 0)

    if obs_history is not None and not obs_history.empty:
        clim = climatology_baseline(oos, obs_history, variable)
        if clim:
            out["crps_climatology"] = clim
            out["crpss_vs_climo"] = metrics.crpss(out["crps"], clim)

    # Calibration.
    p_occ = oos["p_occ"].to_numpy() if "p_occ" in oos else None
    pit = (
        _randomized_pit(y, qcols, p_occ)
        if variable in POINT_MASS_VARIABLES
        else metrics.pit_values(y, qcols)
    )
    out["pit_hist"] = metrics.pit_histogram(pit)
    # Only claim randomisation when it actually happened: it needs the occurrence
    # probability, which a Tier-0 champion does not produce. Labelling an unrandomised
    # precipitation histogram as randomised would present the dry-mass spike as if it
    # were a calibration failure.
    out["pit_randomized"] = bool(variable in POINT_MASS_VARIABLES and p_occ is not None)
    cov50, sharp50 = metrics.interval_coverage(y, oos["q25"].to_numpy(), oos["q75"].to_numpy())
    cov90, sharp90 = metrics.interval_coverage(y, oos["q05"].to_numpy(), oos["q95"].to_numpy())
    out["coverage50"], out["sharpness50"] = cov50, sharp50
    out["coverage90"], out["sharpness90"] = cov90, sharp90

    # Threshold events. The occurrence probability from the precipitation model answers
    # exactly one question — P(precip > its own wet threshold) — so it may only be used
    # for that threshold. Reusing it for a heavier event (P(> 1 mm)) scores the wrong
    # probability against the outcome and makes the forecast look wildly overconfident
    # about heavy rain, with a plausible-looking number.
    briers = {}
    for thr in cfg["thresholds"].get(variable, []):
        if variable == PRECIP_VARIABLE and p_occ is not None and _is_occurrence_threshold(thr):
            prob = p_occ
            basis = "occurrence model"
        else:
            prob = metrics.prob_exceed_from_quantiles(qcols, thr)
            basis = "predictive quantiles"
        if prob is None or len(prob) != len(y):
            continue
        briers[str(thr)] = {
            "brier": metrics.brier_score(y, prob, thr),
            "reliability": metrics.reliability_bins(y, prob, thr),
            "probability_from": basis,
        }
    if briers:
        out["brier"] = briers

    # Where the gain comes from, by lead time.
    lead_bucket = np.array([bucket_for_lead(int(h)) for h in oos["lead_h"]])
    by_lead = {}
    for label in bucket_labels():
        sel = lead_bucket == label
        if sel.sum() < 20:
            continue
        entry = {"n": int(sel.sum()), "crps": float(np.nanmean(crps[sel]))}
        if raw:
            # Within a bucket, compare against the best model that actually covers it —
            # the headline baseline may not reach this lead at all.
            local = {
                m: r for m, r in raw.items() if (r["ok"] & sel).sum() > 10
            }
            if local:
                lm = min(
                    local,
                    key=lambda m: float(
                        np.mean(np.abs(y[local[m]["ok"] & sel] - local[m]["values"][local[m]["ok"] & sel]))
                    ),
                )
                ok = local[lm]["ok"] & sel
                entry["raw_model"] = lm
                entry["crps_raw"] = float(np.mean(np.abs(y[ok] - local[lm]["values"][ok])))
                entry["crps_fused_same_rows"] = float(np.nanmean(crps[ok]))
                entry["crpss_vs_raw"] = metrics.crpss(entry["crps_fused_same_rows"],
                                                      entry["crps_raw"])
        by_lead[label] = entry
    out["by_lead"] = by_lead

    # The daily series drives one chart and was the largest single item on a station page
    # — bigger than the forecast it verifies. The recent window is what a reader looks at,
    # and the summary statistics above already cover the whole evaluation.
    daily_raw_map = daily_raw.to_dict() if daily_raw is not None else {}
    daily_items = list(daily_fused.items())[-DAILY_SERIES_DAYS:]
    out["daily_days_shown"] = len(daily_items)
    out["daily"] = [
        {
            "date": str(d.date()),
            "crps_fused": round(float(v), 4),
            "crps_raw": round(float(daily_raw_map.get(d, np.nan)), 4)
            if np.isfinite(daily_raw_map.get(d, np.nan))
            else None,
        }
        for d, v in daily_items
    ]
    return out


def climatology_baseline(oos: pd.DataFrame, obs_history: pd.DataFrame, variable: str):
    """Mean CRPS of an hour-of-day/day-of-year climatology over the evaluated times."""
    from wxfuser.data.obs import OBS_FOR_VARIABLE

    obs_col = OBS_FOR_VARIABLE.get(variable)
    if not obs_col or obs_history.empty or oos.empty or obs_col not in obs_history:
        return None
    clim = metrics.climatology_quantiles(obs_history, obs_col, oos["valid_time"])
    if not clim or all(np.all(np.isnan(v)) for v in clim.values()):
        return None
    c = metrics.crps_from_quantiles(oos["obs"].to_numpy(dtype=float), clim)
    c = c[np.isfinite(c)]
    return float(c.mean()) if len(c) else None


def rolling_origin_scores(
    wide: pd.DataFrame,
    variable: str,
    models: list[str],
    *,
    tiers: tuple[str, ...] = ("tier0", "tier1"),
    refit_every_days: int = 14,
) -> dict:
    """Lightweight tier comparison used by champion selection.

    Same walk-forward machinery as the scorecard but with a coarser refit cadence, since
    selection only needs the ranking between tiers, not publication-grade numbers.
    """
    oos = walk_forward(wide, variable, models, tiers, refit_every_days=refit_every_days)
    out: dict = {"n_pairs": int(len(wide))}
    if not oos:
        return out

    for tier, frame in oos.items():
        y = frame["obs"].to_numpy(dtype=float)
        c = metrics.crps_from_quantiles(
            y, {k: frame[k].to_numpy() for k in frame if k.startswith("q")}
        )
        out[f"crps_{tier}"] = float(np.nanmean(c))

    any_frame = next(iter(oos.values()))
    y = any_frame["obs"].to_numpy(dtype=float)
    zeros = np.zeros(len(y))
    raw = _raw_baselines(any_frame, wide, models, zeros, zeros)
    if raw:
        best = _pick_baseline(raw)
        out["raw_best_model"] = best
        out["crps_raw_best"] = raw[best]["crps"]
    return out
