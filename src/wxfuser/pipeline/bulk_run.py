"""Running many stations together, where the single-station path would not scale.

``core.run_station`` fetches one station's observations and one station's forecasts, then
trains it. That is the right shape for one station and the wrong shape for five thousand,
because both fetches are network round trips and the cost is linear in stations.

This module inverts the loop. Data acquisition happens once for the whole group — one
DuckDB query against the ASOS archive covers every airport in the shard, and one
Open-Meteo request covers a hundred locations — and only the fitting, which is genuinely
per-station work, runs in a loop. For a 500-station shard that turns roughly 1,000 HTTP
requests into about 15.

Training still dominates the wall clock, which is why the workflows shard across runners.
What this removes is the part that would have hit rate limits long before it hit the clock.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from wxfuser.data import bulk, openmeteo
from wxfuser.data import obs as obs_mod
from wxfuser.data import pairs as pairs_mod
from wxfuser.data.registry import Station
from wxfuser.pipeline import core, emit
from wxfuser.pipeline import state as state_mod


def _group_by_source(stations: list[Station]) -> dict[str, list[Station]]:
    groups: dict[str, list[Station]] = defaultdict(list)
    for s in stations:
        prefix = s.id.split(":", 1)[0] if ":" in s.id else "SNOTEL"
        groups[prefix].append(s)
    return groups


def gather_observations_bulk(
    stations: list[Station], start: date, end: date
) -> dict[str, pd.DataFrame]:
    """Observations for every station, fetched with as few round trips as possible.

    ASOS stations come from one archive query regardless of how many there are. The other
    networks have no bulk endpoint, so they still cost one request each — which is why a
    shard made mostly of ASOS stations runs far faster than one made of Meteostat.
    """
    out: dict[str, pd.DataFrame] = {}
    groups = _group_by_source(stations)

    asos = groups.get("ASOS", [])
    if asos:
        print(f"  bulk ASOS: {len(asos)} stations, {start}..{end}", flush=True)
        frame = bulk.asos_observations([s.id for s in asos], start, end)
        if not frame.empty:
            for sid, g in frame.groupby("station_id"):
                out[str(sid)] = obs_mod.qc(obs_mod.normalize_hourly(g))

    for prefix, group in groups.items():
        if prefix == "ASOS":
            continue
        for st in group:
            try:
                frame = obs_mod.fetch_obs(st.id, start, end, iem_network=st.iem_network)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: obs {st.id} failed ({exc})", flush=True)
                continue
            if frame.empty:
                continue
            out[st.id] = obs_mod.qc(obs_mod.normalize_hourly(frame))
    return out


def gather_forecasts_bulk(
    stations: list[Station],
    variables: list[str],
    *,
    bootstrap: bool,
    years: float,
) -> dict[str, pd.DataFrame]:
    """Model forecasts for every station, batched across locations.

    Stations are grouped by their model set first: a batch request carries one `models`
    parameter, so mixing a CONUS station (which gets HRRR) with a European one would
    either drop HRRR or request it pointlessly.
    """
    by_models: dict[tuple[str, ...], list[Station]] = defaultdict(list)
    for s in stations:
        by_models[tuple(s.resolved_models())].append(s)

    out: dict[str, list[pd.DataFrame]] = defaultdict(list)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(365 * years))

    for models, group in by_models.items():
        coords = [(s.id, s.lat, s.lon) for s in group]
        model_list = list(models)
        print(f"  forecasts for {len(group)} stations, models={model_list}", flush=True)

        prev = openmeteo.fetch_previous_runs_batch(coords, model_list, variables)
        if not prev.empty:
            for sid, g in prev.groupby("station_id"):
                out[str(sid)].append(g)

        if bootstrap:
            hist = openmeteo.fetch_historical_batch(
                coords, model_list, variables, start, end
            )
            if not hist.empty:
                for sid, g in hist.groupby("station_id"):
                    out[str(sid)].append(g)

    return {
        sid: pd.concat(frames, ignore_index=True) for sid, frames in out.items() if frames
    }


def run_stations(
    stations: list[Station],
    *,
    bootstrap: bool = False,
    evaluate: bool = False,
    years: float = 2.0,
    lookback_days: int = 10,
) -> list[dict]:
    """Fetch once for the whole group, then fit and publish each station.

    Returns one index entry per station, in the same shape ``core.run_station`` produces,
    so the site index does not care which path produced a station.
    """
    if not stations:
        return []
    variables = stations[0].resolved_variables()

    end = date.today()
    start = end - timedelta(days=int(365 * years)) if bootstrap else end - timedelta(days=lookback_days)

    print(f"[bulk] {len(stations)} stations, "
          f"{'bootstrap' if bootstrap else 'incremental'} {start}..{end}", flush=True)

    observations = gather_observations_bulk(stations, start, end)
    print(f"  observations for {len(observations)}/{len(stations)} stations", flush=True)

    forecasts = gather_forecasts_bulk(
        stations, variables, bootstrap=bootstrap, years=years
    )
    print(f"  forecasts for {len(forecasts)}/{len(stations)} stations", flush=True)

    live = _live_forecasts(stations, variables)
    print(f"  live forecasts for {len(live)}/{len(stations)} stations", flush=True)

    entries = []
    for n, st in enumerate(stations, 1):
        try:
            entries.append(
                _finish_station(st, observations.get(st.id), forecasts.get(st.id),
                                live.get(st.id), variables, evaluate=evaluate)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{st.id}] FAILED: {exc}", flush=True)
            entries.append({"id": st.id, "name": st.name, "lat": st.lat, "lon": st.lon,
                            "status": "error"})
        if n % 25 == 0:
            ok = sum(1 for e in entries if e.get("status") == "ok")
            print(f"  progress: {n}/{len(stations)} stations, {ok} published", flush=True)
    return entries


def _live_forecasts(stations: list[Station], variables: list[str]) -> dict[str, pd.DataFrame]:
    """Current forecasts for every station, batched, reshaped one frame per station."""
    by_models: dict[tuple[str, ...], list[Station]] = defaultdict(list)
    for s in stations:
        by_models[tuple(s.resolved_models())].append(s)

    out: dict[str, pd.DataFrame] = {}
    for models, group in by_models.items():
        coords = [(s.id, s.lat, s.lon) for s in group]
        long = openmeteo.fetch_forecast_batch(coords, list(models), variables)
        if long.empty:
            continue
        for sid, g in long.groupby("station_id"):
            out[str(sid)] = _to_wide_live(g, list(models), variables)
    return out


def _to_wide_live(long: pd.DataFrame, models: list[str], variables: list[str]) -> pd.DataFrame:
    """One row per valid_time with fc_{var}__{model} columns, as core.emit expects."""
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


def _finish_station(
    station: Station,
    observations: pd.DataFrame | None,
    forecasts: pd.DataFrame | None,
    live: pd.DataFrame | None,
    variables: list[str],
    *,
    evaluate: bool,
) -> dict:
    """Pair, train and publish one station from data already in memory."""
    from wxfuser.models import select
    from wxfuser.verify import rolling

    base = {"id": station.id, "name": station.name, "lat": station.lat, "lon": station.lon}
    if observations is None or observations.empty:
        return {**base, "status": "no_observations"}
    if forecasts is None or forecasts.empty:
        return {**base, "status": "no_forecasts"}
    if live is None or live.empty:
        return {**base, "status": "no_forecast"}

    core.merge_obs_history(station, observations)
    built = pairs_mod.build_pairs(forecasts, observations, station.id, variables)
    archive = core.update_archive(station, built)
    if archive.empty:
        return {**base, "status": "no_data"}

    stored = state_mod.load(core.STATE_DIR, station.slug)
    obs_path = core.obs_path(station)
    obs_history = pd.read_parquet(obs_path) if obs_path.exists() else pd.DataFrame()

    models = station.resolved_models()
    predictions, methods, skill, scorecards = {}, {}, {}, {}

    for variable in variables:
        card = None
        if evaluate:
            wide_pre = pairs_mod.to_wide(archive, variable, models)
            eligible = [t for t in select.eligible_tiers(wide_pre) if t in core.FIT_FUNCS]
            card = rolling.scorecard(
                wide_pre, variable, models, tiers=tuple(eligible), obs_history=obs_history
            )
            if card.get("insufficient_data"):
                card = None

        trained = core.train_variable(
            archive, variable, models,
            previous_champion=state_mod.champion_for(stored, variable),
            scorecard=card,
        )
        if trained.get("status") != "ok":
            continue
        wide = trained["wide"]
        vf = core.variable_frame(live, variable, models)
        pred = core.predict_variable(trained, vf, models)
        if not pred.get("calibrated"):
            continue

        predictions[variable] = pred
        methods[variable] = emit.method_block(
            pred.get("_tier_used") or trained["champion"], wide,
            pred.get("calibrated", False), models
        )
        if card:
            state_mod.record(
                stored, variable, champion=trained["champion"], scorecard=card,
                selection=trained.get("selection"),
                evaluated_at=pd.Timestamp.utcnow().tz_localize(None).isoformat(timespec="seconds"),
            )
        else:
            card = state_mod.scorecard_for(stored, variable) or {}
            state_mod.record(stored, variable, champion=trained["champion"])
        scorecards[variable] = card
        skill[variable] = emit.skill_block(card)

    state_mod.save(core.STATE_DIR, station.slug, stored)
    if not predictions:
        return {**base, "status": "warming_up"}

    payload = emit.forecast_json(
        station.public_dict(), models, live, predictions, methods, skill
    )
    emit.write_json(payload, core.SITE_DIR / "stations" / station.slug / "forecast.json")
    emit.write_json(
        {"schema_version": emit.SCHEMA_VERSION, "station": station.public_dict(),
         "variables": scorecards},
        core.SITE_DIR / "stations" / station.slug / "verify.json",
    )

    headline = skill.get("air_temp_c", {})
    return {
        **base,
        "slug": station.slug,
        "elev_m": station.elev_m,
        "status": "ok",
        "models": models,
        "variables": list(predictions),
        "crpss_vs_raw": headline.get("crpss_vs_raw"),
        "beats_raw": headline.get("beats_raw", False),
        "method": methods.get("air_temp_c", {}).get("label"),
        "train_days": methods.get("air_temp_c", {}).get("train_days"),
    }
