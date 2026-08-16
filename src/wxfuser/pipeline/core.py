"""The per-station work: gather data, train the tiers, verify, publish.

Bootstrap and refresh differ only in how much history they pull, so both call through
here. The sequence for a station is always the same:

  1. obtain observations (deep history on enrollment, a trailing window on refresh)
  2. obtain model forecasts for the same valid times (archived) and the future (live)
  3. pair them, merge into the station's archive
  4. fit every eligible tier, score them out-of-sample, pick a champion
  5. predict the coming week with the champion and write the JSON the site reads

Step 4 is the part that must not be skipped on refresh. A system that trains once and
then only predicts drifts as the model changes underneath it; refitting the cheap tiers
every run is what makes this adaptive rather than a one-time calibration.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.data import obs as obs_mod
from wxfuser.data import openmeteo
from wxfuser.data import pairs as pairs_mod
from wxfuser.data.registry import Station
from wxfuser.models import select, tier0_bias, tier1_emos, tier2, tier3_gbm
from wxfuser.pipeline import emit
from wxfuser.pipeline import state as state_mod
from wxfuser.verify import rolling

STATE_DIR = Path(os.environ.get("WXFUSER_STATE_DIR", "state"))
SITE_DIR = Path(os.environ.get("WXFUSER_SITE_DIR", "site"))


def pairs_path(station: Station) -> Path:
    return STATE_DIR / "pairs" / f"{station.slug}.parquet"


def obs_path(station: Station) -> Path:
    return STATE_DIR / "obs" / f"{station.slug}.parquet"


# --------------------------------------------------------------------------- observations


def gather_observations(station: Station, start: date, end: date) -> pd.DataFrame:
    """Observations for a window, combining every source this station can use.

    Sources are merged rather than chosen: NWS gives the freshest hours but only a week
    of them, GHCN-h gives years but lands days late. Taking both and preferring the
    higher-priority source per hour yields a series that is simultaneously current and
    deep, which neither source provides alone.
    """
    frames: list[pd.DataFrame] = []

    primary = obs_mod.fetch_obs(station.id, start, end, iem_network=station.iem_network)
    if not primary.empty:
        frames.append(primary)

    # Supplementary sources under their own ids, relabelled to the station's canonical id
    # so the pairing key stays single-valued.
    if station.nws_id and not station.id.startswith("NWS:"):
        extra = obs_mod.fetch_nws_hourly(station.nws_id, max(start, end - timedelta(days=7)), end)
        if not extra.empty:
            extra["station_id"] = station.id
            frames.append(extra)
    if station.ghcnh_id and not station.id.startswith("GHCNH:"):
        extra = obs_mod.fetch_ghcnh_hourly(station.ghcnh_id, start, end)
        if not extra.empty:
            extra["station_id"] = station.id
            frames.append(extra)

    if not frames:
        return pd.DataFrame(columns=obs_mod.OBS_COLUMNS)

    for f in frames:
        f["station_id"] = station.id
    combined = pd.concat(frames, ignore_index=True)
    combined = obs_mod.normalize_hourly(combined)
    return obs_mod.qc(combined)


def merge_obs_history(station: Station, new_obs: pd.DataFrame) -> pd.DataFrame:
    """Keep a local observation history so climatology has something to stand on."""
    path = obs_path(station)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if old.empty and new_obs.empty:
        return new_obs
    combined = pd.concat([old, new_obs], ignore_index=True) if not old.empty else new_obs
    combined = obs_mod.normalize_hourly(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False, compression="zstd")
    return combined


# --------------------------------------------------------------------------- archive


def update_archive(station: Station, new_pairs: pd.DataFrame) -> pd.DataFrame:
    path = pairs_path(station)
    existing = pairs_mod.read_archive(path)
    merged = pairs_mod.merge_archive(existing, new_pairs)
    if not merged.empty:
        pairs_mod.write_archive(merged, path)
    return merged


def backfill_pairs(station: Station, years: float = 2.0, previous_runs_days: int = 92) -> pd.DataFrame:
    """Pull paired history for a newly enrolled station.

    This is what makes a station useful the moment it is enrolled rather than months
    later. Two archives are combined:

      * the seamless historical forecast archive, which reaches back years but only
        represents short lead times, and
      * the previous-runs archive, which is lead-resolved and therefore the only honest
        basis for multi-day calibration, but reaches back ~3 months per request.
    """
    models = station.resolved_models()
    variables = station.resolved_variables()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(365 * years))

    print(f"  fetching observations {start} .. {end}", flush=True)
    observations = gather_observations(station, start, end)
    print(f"  obs rows: {len(observations)}", flush=True)
    if observations.empty:
        return pd.DataFrame()
    merge_obs_history(station, observations)

    obs_start = pd.to_datetime(observations["valid_time"]).min().date()
    eff_start = max(start, obs_start)

    frames = []
    print(f"  fetching archived forecasts ({len(models)} models)", flush=True)
    hist = openmeteo.fetch_historical(station.lat, station.lon, models, variables, eff_start, end)
    if not hist.empty:
        frames.append(hist)
        print(f"  historical-forecast rows: {len(hist)}", flush=True)

    prev = openmeteo.fetch_previous_runs(
        station.lat, station.lon, models, variables, past_days=previous_runs_days
    )
    if not prev.empty:
        frames.append(prev)
        print(f"  previous-runs rows: {len(prev)}", flush=True)

    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)
    built = pairs_mod.build_pairs(forecasts, observations, station.id, variables)
    print(f"  paired rows: {len(built)}", flush=True)
    return update_archive(station, built)


def incremental_pairs(station: Station, lookback_days: int = 10) -> pd.DataFrame:
    """Refresh step: re-pair a trailing window so late-arriving obs are picked up."""
    models = station.resolved_models()
    variables = station.resolved_variables()
    end = date.today()
    start = end - timedelta(days=lookback_days)

    observations = gather_observations(station, start, end)
    if observations.empty:
        return pairs_mod.read_archive(pairs_path(station))
    merge_obs_history(station, observations)

    frames = []
    prev = openmeteo.fetch_previous_runs(
        station.lat, station.lon, models, variables, past_days=max(lookback_days, 7)
    )
    if not prev.empty:
        frames.append(prev)
    hist = openmeteo.fetch_historical(
        station.lat, station.lon, models, variables, start, end - timedelta(days=1)
    )
    if not hist.empty:
        frames.append(hist)

    if not frames:
        return pairs_mod.read_archive(pairs_path(station))
    forecasts = pd.concat(frames, ignore_index=True)
    built = pairs_mod.build_pairs(forecasts, observations, station.id, variables)
    return update_archive(station, built)


# --------------------------------------------------------------------------- training


FIT_FUNCS = {
    "tier0": lambda w, v, m: tier0_bias.fit(w, v),
    "tier1": lambda w, v, m: tier1_emos.fit(w, v, m),
    "tier2": lambda w, v, m: tier2.fit(w, v, m),
    "tier3": lambda w, v, m: tier3_gbm.fit(w, v, m),
}

PREDICT_FUNCS = {
    "tier0": lambda st, fc, m: tier0_bias.predict(
        st, fc["fc_mean"].to_numpy(), fc["lead_h"].to_numpy(), fc["valid_time"]
    ),
    "tier1": lambda st, fc, m: tier1_emos.predict(st, fc, m),
    "tier2": lambda st, fc, m: tier2.predict(st, fc, m),
    "tier3": lambda st, fc, m: tier3_gbm.predict(st, fc, m),
}


def train_variable(
    archive: pd.DataFrame,
    variable: str,
    models: list[str],
    previous_champion: str | None = None,
    *,
    scorecard: dict | None = None,
) -> dict:
    """Fit the tiers for one variable and decide which to publish.

    Without a ``scorecard`` (the refresh path) this fits on data up to now and publishes
    with the previously chosen champion — cheap, and correct, because that choice came
    from a full walk-forward evaluation that three more hours of data will not overturn.
    With one (bootstrap and weekly retrain) the champion is re-decided from its per-tier
    out-of-sample scores.
    """
    wide = pairs_mod.to_wide(archive, variable, models)
    if wide.empty or len(wide) < 30:
        return {"variable": variable, "status": "insufficient_data", "n_pairs": int(len(wide))}

    eligible = [t for t in select.eligible_tiers(wide) if t in FIT_FUNCS]

    if scorecard:
        # The scorecard already ran a walk-forward over every eligible tier; reusing its
        # per-tier scores avoids a second, identical evaluation.
        scores = {f"crps_{t}": v for t, v in (scorecard.get("tier_crps") or {}).items()}
        scores["crps_raw_best"] = scorecard.get("crps_raw_best")
        scores["raw_best_model"] = scorecard.get("raw_best_model")
        champion, decision = select.choose(scores, eligible, previous_champion)
    else:
        scores = {}
        champion = previous_champion if previous_champion in eligible else eligible[-1]
        decision = {"reason": "reused the last evaluated champion", "scores": {}}

    # Only fit what we might publish, plus cheap fallbacks in case the champion cannot
    # produce a forecast this run; fitting the losers every refresh is wasted work.
    to_fit = [champion] + [t for t in ("tier1", "tier0") if t != champion and t in eligible]
    to_fit = [t for t in to_fit if t]
    states: dict[str, dict] = {}
    for tier in dict.fromkeys(to_fit):
        try:
            states[tier] = FIT_FUNCS[tier](wide, variable, models)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARN: fit {tier}/{variable} failed ({exc})", flush=True)

    if champion not in states:
        champion = next(iter(states), None)

    return {
        "variable": variable,
        "status": "ok" if states else "fit_failed",
        "champion": champion,
        "eligible": eligible,
        "selection": decision,
        "rolling_scores": {k: v for k, v in scores.items() if k != "daily"},
        "states": states,
        "wide": wide,
        "n_pairs": int(len(wide)),
        "train_days": select.training_days(wide),
    }


def predict_variable(trained: dict, fc_wide: pd.DataFrame, models: list[str]) -> dict:
    """Run the champion (falling back down the tiers if it cannot produce a forecast)."""
    champion = trained.get("champion")
    states = trained.get("states", {})
    order = [champion] + [t for t in ("tier3", "tier2", "tier1", "tier0") if t != champion]
    for tier in order:
        state = states.get(tier)
        if not state:
            continue
        try:
            pred = PREDICT_FUNCS[tier](state, fc_wide, models)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARN: predict {tier} failed ({exc})", flush=True)
            continue
        if pred and pred.get("calibrated"):
            pred["_tier_used"] = tier
            return pred
    return {"calibrated": False, "_tier_used": None}


# --------------------------------------------------------------------------- live forecast


def live_forecast_wide(station: Station, models: list[str], variables: list[str]) -> pd.DataFrame:
    """One row per valid_time with each model's prediction, ready for the tiers.

    Columns: valid_time, lead_h, fc_{model} (for the variable being predicted), plus
    fc_{var}__{model} for every variable so the published JSON can show raw model traces.
    """
    long = openmeteo.fetch_forecast(station.lat, station.lon, models, variables, forecast_days=7)
    if long.empty:
        return pd.DataFrame()

    base = (
        long[["valid_time", "lead_h"]]
        .drop_duplicates("valid_time")
        .sort_values("valid_time")
        .reset_index(drop=True)
    )
    for var in variables:
        for model in models:
            sel = long[long["model"] == model][["valid_time", f"fc_{var}"]]
            sel = sel.rename(columns={f"fc_{var}": f"fc_{var}__{model}"})
            base = base.merge(sel, on="valid_time", how="left")
    return base


def variable_frame(fc_wide: pd.DataFrame, variable: str, models: list[str]) -> pd.DataFrame:
    """Slice the live forecast down to one variable, in the shape the tiers expect."""
    out = fc_wide[["valid_time", "lead_h"]].copy()
    cols = []
    for model in models:
        src = f"fc_{variable}__{model}"
        dst = f"fc_{model}"
        out[dst] = fc_wide[src] if src in fc_wide else np.nan
        cols.append(dst)
    with np.errstate(invalid="ignore"):
        out["fc_mean"] = out[cols].mean(axis=1, skipna=True)
        out["fc_spread"] = out[cols].std(axis=1, skipna=True, ddof=0)
    return out


# --------------------------------------------------------------------------- station run


def run_station(
    station: Station,
    *,
    bootstrap: bool = False,
    years: float = 2.0,
    evaluate: bool | None = None,
) -> dict:
    """Full cycle for one station; returns its index entry.

    ``evaluate`` defaults to True on bootstrap (a new station has no stored decision) and
    False on refresh (reuse the weekly evaluation). The weekly retrain passes it
    explicitly.
    """
    if evaluate is None:
        evaluate = bootstrap
    models = station.resolved_models()
    variables = station.resolved_variables()
    print(f"[{station.id}] {station.name} — {len(models)} models, {len(variables)} variables",
          flush=True)

    stored = state_mod.load(STATE_DIR, station.slug)
    archive = backfill_pairs(station, years=years) if bootstrap else incremental_pairs(station)
    if archive is None or archive.empty:
        print(f"[{station.id}] no paired data; skipping", flush=True)
        return {"id": station.id, "name": station.name, "lat": station.lat,
                "lon": station.lon, "status": "no_data"}

    obs_history = pd.read_parquet(obs_path(station)) if obs_path(station).exists() else pd.DataFrame()

    fc_wide = live_forecast_wide(station, models, variables)
    if fc_wide.empty:
        print(f"[{station.id}] live forecast unavailable; skipping", flush=True)
        return {"id": station.id, "name": station.name, "lat": station.lat,
                "lon": station.lon, "status": "no_forecast"}

    predictions: dict[str, dict] = {}
    methods: dict[str, dict] = {}
    skill: dict[str, dict] = {}
    scorecards: dict[str, dict] = {}

    for variable in variables:
        card = None
        if evaluate:
            wide_pre = pairs_mod.to_wide(archive, variable, models)
            eligible = [t for t in select.eligible_tiers(wide_pre) if t in FIT_FUNCS]
            card = rolling.scorecard(
                wide_pre, variable, models, tiers=tuple(eligible), obs_history=obs_history
            )
            if card.get("insufficient_data"):
                card = None

        trained = train_variable(
            archive,
            variable,
            models,
            previous_champion=state_mod.champion_for(stored, variable),
            scorecard=card,
        )
        if trained.get("status") != "ok":
            continue
        wide = trained["wide"]
        vf = variable_frame(fc_wide, variable, models)
        pred = predict_variable(trained, vf, models)
        if not pred.get("calibrated"):
            continue
        predictions[variable] = pred
        methods[variable] = emit.method_block(
            pred.get("_tier_used") or trained["champion"], wide,
            pred.get("calibrated", False), models
        )

        if card:
            state_mod.record(
                stored, variable,
                champion=trained["champion"],
                scorecard=card,
                selection=trained.get("selection"),
                evaluated_at=pd.Timestamp.utcnow().tz_localize(None).isoformat(timespec="seconds"),
            )
        else:
            # Reuse the last evaluation rather than quoting an unverified number.
            card = state_mod.scorecard_for(stored, variable) or {}
            state_mod.record(stored, variable, champion=trained["champion"])

        scorecards[variable] = card
        skill[variable] = emit.skill_block(card)
        gain = skill[variable].get("crpss_vs_raw")
        mae_gain = skill[variable].get("mae_gain_vs_raw")
        print(
            f"  {variable}: {trained['champion']} ({trained['train_days']:.0f} d) "
            f"CRPSS={gain if gain is None else round(gain, 3)} "
            f"MAE gain={mae_gain if mae_gain is None else round(mae_gain, 3)}",
            flush=True,
        )

    state_mod.save(STATE_DIR, station.slug, stored)

    if not predictions:
        return {"id": station.id, "name": station.name, "lat": station.lat,
                "lon": station.lon, "status": "warming_up"}

    payload = emit.forecast_json(
        station.public_dict(), models, fc_wide, predictions, methods, skill
    )
    emit.write_json(payload, SITE_DIR / "stations" / station.slug / "forecast.json")
    emit.write_json(
        {"schema_version": emit.SCHEMA_VERSION, "station": station.public_dict(),
         "variables": scorecards},
        SITE_DIR / "stations" / station.slug / "verify.json",
    )

    headline = skill.get("air_temp_c", {})
    return {
        "id": station.id,
        "name": station.name,
        "slug": station.slug,
        "lat": station.lat,
        "lon": station.lon,
        "elev_m": station.elev_m,
        "status": "ok",
        "models": models,
        "variables": list(predictions),
        "crpss_vs_raw": headline.get("crpss_vs_raw"),
        "beats_raw": headline.get("beats_raw", False),
        "method": methods.get("air_temp_c", {}).get("label"),
        "train_days": methods.get("air_temp_c", {}).get("train_days"),
    }
