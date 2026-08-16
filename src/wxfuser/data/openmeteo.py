"""Open-Meteo clients — the model side of the fusion.

Three endpoints, each answering a different question:

  * **forecast** (``api.open-meteo.com``) — what the models predict for the next week.
    This is what we post-process and publish.
  * **historical forecast** (``historical-forecast-api.open-meteo.com``) — the archived
    *forecast as issued* for past dates, back to 2021-2022 depending on model. This is
    what makes the system useful on day one: a station enrolled today can be trained on
    two years of paired forecast/obs immediately instead of waiting months to accumulate
    an archive. Its caveat is that it stores the "seamless" best-available forecast, so
    the lead time of a given row is short and not precisely known — rows from here are
    tagged ``lead_source="hist"`` and only train the short-lead buckets.
  * **previous runs** (``previous-runs-api.open-meteo.com``) — the same valid times but
    resolved by lead: ``temperature_2m_previous_day3`` is what the model said about this
    hour three days ago. This is the honest source for long-lead training, available for
    most models from 2024.

Response naming, verified against the live API: hourly keys are
``{omvar}[_previous_day{N}]_{model}`` when more than one model is requested, and plain
``{omvar}[_previous_day{N}]`` when exactly one is. We always request explicitly and
normalise both shapes into long form.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from wxfuser.config import model_catalogue, model_supports, variable_map

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

USER_AGENT = "wxfuser/0.1 (+https://github.com/nakas/wxStationFuser)"

# Columns every forecast frame carries.
FC_COLUMNS = ["model", "valid_time", "lead_h", "lead_source"]


class OpenMeteoError(RuntimeError):
    pass


# Minimum spacing between requests. The free tier allows 600/minute, but a burst of
# batched multi-location calls is far heavier than a burst of single-point ones and gets
# throttled well before that nominal rate. Pacing costs a little wall clock and saves
# whole batches from being lost to 429s.
MIN_REQUEST_INTERVAL_S = 0.6
_last_request_at = 0.0


def _http_json(url: str, *, timeout: int = 120, retries: int = 5) -> dict | list:
    """GET JSON, paced between calls and backing off hard on throttling.

    429 is the expected failure here and it resolves only by waiting, so it gets a much
    longer backoff than a transient network error — an earlier version topped out around
    nine seconds and lost entire model-years of history to sustained throttling.
    """
    global _last_request_at

    req = Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        gap = time.monotonic() - _last_request_at
        if gap < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - gap)
        try:
            _last_request_at = time.monotonic()
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last = exc
            if exc.code == 429 and attempt < retries - 1:
                # Back off in minutes, not seconds: the quota window is per-minute.
                wait = min(20 * (attempt + 1), 90)
                print(f"  rate limited; waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    assert last is not None
    raise OpenMeteoError(f"Open-Meteo request failed after {retries} attempts: {last}") from last


def _om_vars(variables: list[str]) -> list[str]:
    vmap = variable_map()
    return [vmap[v] for v in variables if v in vmap]


def _series(hourly: dict, key: str, n: int) -> pd.Series:
    vals = hourly.get(key)
    if vals is None:
        return pd.Series([np.nan] * n, dtype="float64")
    return pd.Series(vals, dtype="float64")


def _resolve_key(hourly: dict, om_var: str, model: str, n_models: int, day: int | None) -> str | None:
    """Locate the hourly key for (variable, model, previous-day offset).

    Single-model requests come back unsuffixed, multi-model requests suffixed with the
    model id; we accept either so the caller never has to care how many models it asked
    for.
    """
    stem = om_var if day is None else f"{om_var}_previous_day{day}"
    for candidate in (f"{stem}_{model}", stem):
        if candidate in hourly:
            return candidate
    return None


def _frame_from_hourly(
    hourly: dict,
    models: list[str],
    variables: list[str],
    *,
    lead_source: str,
    day: int | None = None,
) -> pd.DataFrame:
    """Long-form frame: one row per (model, valid_time), one column per variable."""
    times = hourly.get("time") if hourly else None
    if not times:
        return pd.DataFrame(columns=[*FC_COLUMNS, *(f"fc_{v}" for v in variables)])

    valid = pd.to_datetime(times)
    n = len(times)
    vmap = variable_map()
    frames = []
    for model in models:
        block = pd.DataFrame({"valid_time": valid})
        block["model"] = model
        got_any = False
        for var in variables:
            om_var = vmap.get(var)
            col = f"fc_{var}"
            # A model that does not carry the variable gets NaN, never a null column
            # mistaken for real data.
            if om_var is None or not model_supports(model, var):
                block[col] = np.nan
                continue
            key = _resolve_key(hourly, om_var, model, len(models), day)
            if key is None:
                block[col] = np.nan
                continue
            block[col] = _series(hourly, key, n)
            got_any = True
        if got_any:
            frames.append(block)

    if not frames:
        return pd.DataFrame(columns=[*FC_COLUMNS, *(f"fc_{v}" for v in variables)])
    out = pd.concat(frames, ignore_index=True)
    out["lead_source"] = lead_source
    return out


# --------------------------------------------------------------------------- live forecast


def fetch_forecast(
    lat: float,
    lon: float,
    models: list[str],
    variables: list[str],
    *,
    forecast_days: int = 7,
    past_days: int = 0,
) -> pd.DataFrame:
    """Current multi-model forecast at a point.

    One call covers every model, so publishing a station costs a single request.
    ``lead_h`` is measured from the request time floored to the hour — an approximation
    of true model init time, which the live endpoint does not expose. It is only used
    for bucketing, and rows are re-paired against observations later using valid_time.
    """
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(_om_vars(variables)),
        "models": ",".join(models),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "forecast_days": str(forecast_days),
    }
    if past_days:
        params["past_days"] = str(past_days)
    payload = _http_json(f"{FORECAST_URL}?" + urlencode(params))
    if not isinstance(payload, dict):
        raise OpenMeteoError("unexpected multi-location response for a single point")

    out = _frame_from_hourly(payload.get("hourly", {}), models, variables, lead_source="live")
    if out.empty:
        return out
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    out["lead_h"] = ((out["valid_time"] - now) / pd.Timedelta(hours=1)).astype(int)
    out = out[out["lead_h"] >= 0].reset_index(drop=True)
    return out[[*FC_COLUMNS, *(f"fc_{v}" for v in variables)]]


# --------------------------------------------------------------------------- historical


def fetch_historical(
    lat: float,
    lon: float,
    models: list[str],
    variables: list[str],
    start: date,
    end: date,
    *,
    chunk_days: int = 365,
) -> pd.DataFrame:
    """Archived forecasts (as issued) over a date range, chunked to keep responses sane.

    Rows are tagged ``lead_source="hist"`` with a nominal short lead: the archive stores
    the seamless best-available forecast, so these rows legitimately train the 1-6 h
    bucket but must not be presented as evidence of 5-day skill.
    """
    catalogue = model_catalogue()
    frames: list[pd.DataFrame] = []
    for model in models:
        meta = catalogue.get(model, {})
        model_start = _as_date(meta.get("archive_from")) or start
        eff_start = max(start, model_start)
        if eff_start > end:
            continue
        cursor = eff_start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            params = {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "start_date": cursor.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(_om_vars(variables)),
                "models": model,
                "wind_speed_unit": "ms",
                "timezone": "GMT",
            }
            try:
                payload = _http_json(f"{HISTORICAL_URL}?" + urlencode(params))
                hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
                f = _frame_from_hourly(hourly, [model], variables, lead_source="hist")
                if not f.empty:
                    frames.append(f)
            except OpenMeteoError as exc:
                print(f"  WARN: historical {model} {cursor}..{chunk_end} failed ({exc})", flush=True)
            cursor = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=[*FC_COLUMNS, *(f"fc_{v}" for v in variables)])
    out = pd.concat(frames, ignore_index=True)
    out["lead_h"] = 3  # nominal: seamless archive is short-lead by construction
    return out[[*FC_COLUMNS, *(f"fc_{v}" for v in variables)]]


# --------------------------------------------------------------------------- previous runs


def fetch_previous_runs(
    lat: float,
    lon: float,
    models: list[str],
    variables: list[str],
    *,
    past_days: int = 92,
    days: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
) -> pd.DataFrame:
    """Lead-resolved archived forecasts: what each model said N days before valid time.

    This is the only source that honestly trains long leads. ``past_days`` is capped by
    the API at 92; longer histories come from repeated calls in the bootstrap.
    Each requested day offset becomes rows with ``lead_h ~ 24*N``, which is the lead at
    the *start* of that forecast day — good enough for bucket assignment.
    """
    vmap = variable_map()
    frames: list[pd.DataFrame] = []
    for model in models:
        wanted = [v for v in variables if model_supports(model, v)]
        if not wanted:
            continue
        hourly_params = [
            f"{vmap[v]}_previous_day{d}" for d in days for v in wanted if v in vmap
        ]
        if not hourly_params:
            continue
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "hourly": ",".join(hourly_params),
            "models": model,
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "past_days": str(min(past_days, 92)),
            "forecast_days": "1",
        }
        try:
            payload = _http_json(f"{PREVIOUS_RUNS_URL}?" + urlencode(params))
        except OpenMeteoError as exc:
            print(f"  WARN: previous-runs {model} failed ({exc})", flush=True)
            continue
        hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
        for d in days:
            f = _frame_from_hourly(hourly, [model], wanted, lead_source="prev_runs", day=d)
            if f.empty:
                continue
            f["lead_h"] = 24 * d
            for v in variables:
                if f"fc_{v}" not in f:
                    f[f"fc_{v}"] = np.nan
            frames.append(f)

    if not frames:
        return pd.DataFrame(columns=[*FC_COLUMNS, *(f"fc_{v}" for v in variables)])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=[f"fc_{v}" for v in variables], how="all").reset_index(drop=True)
    return out[[*FC_COLUMNS, *(f"fc_{v}" for v in variables)]]


# --------------------------------------------------------------------------- batching

# Locations per request. Open-Meteo accepts comma-separated coordinates and returns one
# result object per location, in order. This is what makes thousands of stations viable:
# a refresh over 5,000 stations is ~50 requests instead of 5,000.
BATCH_SIZE = 100


def _batch_request(
    base_url: str,
    coords: list[tuple[str, float, float]],
    params: dict,
    models: list[str],
    variables: list[str],
    *,
    lead_source: str,
    day: int | None = None,
) -> pd.DataFrame:
    """One multi-location request; returns long-form rows tagged with station_id.

    The response is a list aligned with the requested coordinates. A single-location
    request comes back as a bare object instead, so both shapes are accepted — otherwise
    a batch that happens to contain one station would break.
    """
    if not coords:
        return pd.DataFrame()

    q = dict(params)
    q["latitude"] = ",".join(f"{la:.4f}" for _, la, _ in coords)
    q["longitude"] = ",".join(f"{lo:.4f}" for _, _, lo in coords)
    payload = _http_json(f"{base_url}?" + urlencode(q))
    results = payload if isinstance(payload, list) else [payload]

    frames = []
    for (sid, _, _), res in zip(coords, results):
        hourly = res.get("hourly", {}) if isinstance(res, dict) else {}
        f = _frame_from_hourly(hourly, models, variables, lead_source=lead_source, day=day)
        if f.empty:
            continue
        f["station_id"] = sid
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_forecast_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    *,
    forecast_days: int = 7,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Current multi-model forecasts for many stations, batched by location."""
    base = {
        "hourly": ",".join(_om_vars(variables)),
        "models": ",".join(models),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "forecast_days": str(forecast_days),
    }
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    frames = []
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        try:
            f = _batch_request(FORECAST_URL, chunk, base, models, variables, lead_source="live")
        except OpenMeteoError as exc:
            print(f"  WARN: forecast batch {i // batch_size} failed ({exc})", flush=True)
            continue
        if not f.empty:
            frames.append(f)
        print(f"  forecast: {min(i + batch_size, len(coords))}/{len(coords)} stations",
              flush=True)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["lead_h"] = ((out["valid_time"] - now) / pd.Timedelta(hours=1)).astype(int)
    return out[out["lead_h"] >= 0].reset_index(drop=True)


def fetch_historical_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    start: date,
    end: date,
    *,
    batch_size: int = BATCH_SIZE,
    chunk_days: int = 365,
) -> pd.DataFrame:
    """Archived forecasts for many stations, batched by location and chunked by date.

    One model at a time: the archive's per-model availability differs, and asking for a
    model outside its window wastes the whole batch rather than one column.
    """
    catalogue = model_catalogue()
    frames = []
    for model in models:
        model_start = _as_date(catalogue.get(model, {}).get("archive_from")) or start
        cursor = max(start, model_start)
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            base = {
                "start_date": cursor.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(_om_vars(variables)),
                "models": model,
                "wind_speed_unit": "ms",
                "timezone": "GMT",
            }
            chunk_rows = 0
            for i in range(0, len(coords), batch_size):
                chunk = coords[i : i + batch_size]
                try:
                    f = _batch_request(
                        HISTORICAL_URL, chunk, base, [model], variables, lead_source="hist"
                    )
                except OpenMeteoError as exc:
                    print(f"  WARN: historical {model} {cursor} batch failed ({exc})",
                          flush=True)
                    continue
                if not f.empty:
                    frames.append(f)
                    chunk_rows += len(f)
            print(f"  historical {model} {cursor}..{chunk_end}: {chunk_rows} rows", flush=True)
            cursor = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["lead_h"] = 3  # seamless archive is short-lead by construction
    return out.reset_index(drop=True)


def fetch_previous_runs_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    *,
    past_days: int = 92,
    days: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Lead-resolved archived forecasts for many stations, batched by location."""
    vmap = variable_map()
    frames = []
    for model in models:
        wanted = [v for v in variables if model_supports(model, v)]
        hourly_params = [f"{vmap[v]}_previous_day{d}" for d in days for v in wanted if v in vmap]
        if not hourly_params:
            continue
        model_rows = 0
        base = {
            "hourly": ",".join(hourly_params),
            "models": model,
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "past_days": str(min(past_days, 92)),
            "forecast_days": "1",
        }
        for i in range(0, len(coords), batch_size):
            chunk = coords[i : i + batch_size]
            try:
                payload_frames = []
                for d in days:
                    f = _batch_request(
                        PREVIOUS_RUNS_URL, chunk, base, [model], wanted,
                        lead_source="prev_runs", day=d,
                    )
                    if f.empty:
                        continue
                    f["lead_h"] = 24 * d
                    payload_frames.append(f)
            except OpenMeteoError as exc:
                print(f"  WARN: previous-runs {model} batch failed ({exc})", flush=True)
                continue
            frames.extend(payload_frames)
            model_rows += sum(len(f) for f in payload_frames)
        # Per-model, not cumulative: an earlier version printed the running total for each
        # model, so a model that returned nothing still appeared to have contributed.
        print(f"  previous-runs {model}: {model_rows} rows", flush=True)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for v in variables:
        if f"fc_{v}" not in out:
            out[f"fc_{v}"] = np.nan
    return out.dropna(
        subset=[f"fc_{v}" for v in variables], how="all"
    ).reset_index(drop=True)


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def probe_availability(lat: float, lon: float, model: str, variables: list[str]) -> dict[str, bool]:
    """Which variables a model actually returns at this point, checked against the API.

    configs/models.yaml records what we believe; this verifies it for a specific location
    (regional models return nothing outside their domain). Used at enrollment so a
    station never gets trained on a model that is silently all-NaN there.
    """
    vmap = variable_map()
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(vmap[v] for v in variables if v in vmap),
        "models": model,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "forecast_days": "2",
    }
    try:
        payload = _http_json(f"{FORECAST_URL}?" + urlencode(params))
    except OpenMeteoError:
        return {v: False for v in variables}
    hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
    out = {}
    for v in variables:
        key = _resolve_key(hourly, vmap.get(v, ""), model, 1, None)
        vals = hourly.get(key) if key else None
        out[v] = bool(vals) and any(x is not None for x in vals)
    return out
