"""Hourly surface observations — the ground truth the fusion is trained and verified on.

Every source is normalised to one schema:

  station_id, valid_time (UTC, top of hour), air_temp_c, dewpoint_c,
  relative_humidity_pct, wind_speed_ms, wind_gust_ms, wind_dir_deg, precip_1h_mm, source

Station ids carry a network prefix so ``fetch_obs`` can route without a lookup:
``IEM:``, ``SYN:``, ``MS:``, ``NWS:``, ``GHCNH:``, or a bare SNOTEL triplet.

Sources and what each is for:
  * **NWS** (api.weather.gov) — no key, CORS-open, but only ~7 days of retention. The
    incremental "what happened since the last run" source for US stations, and the only
    one the browser can reach directly for instant mode.
  * **GHCN-h** (NCEI) — the deep-history source, per-station files covering the full
    period of record. Replaces ISD, which NOAA froze in Aug 2025.
  * **IEM** — bulk ASOS/AWOS/RAWS archive, no key, good global airport coverage.
  * **Meteostat** — ~22k global stations, bulk yearly files, lags a few days.
  * **Synoptic** — mesonets (RAWS, state networks) that none of the above carry. Needs a
    token, and as of 2026 the free Open Access tier is restricted to accredited US
    academic users, so for most projects this source requires a paid plan. Everything
    else works without it.

QC lives in ``qc()`` and is applied by the pipeline, never inside the fetchers, so raw
pulls stay cacheable and auditable.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
SYNOPTIC_TS = "https://api.synopticdata.com/v2/stations/timeseries"
NWS_API = "https://api.weather.gov"
GHCNH_BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-station"

USER_AGENT = "wxfuser/0.1 (+https://github.com/nakas/wxStationFuser)"


def F_TO_C(f):  # noqa: N802
    return (f - 32.0) * 5.0 / 9.0


IN_TO_MM = 25.4
MPH_TO_MS = 0.44704
KT_TO_MS = 0.514444

OBS_COLUMNS = [
    "station_id",
    "valid_time",
    "air_temp_c",
    "dewpoint_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_dir_deg",
    "precip_1h_mm",
    "source",
]

# Canonical model-side variable names -> obs column names. The paired archive uses the
# model-side names, so this is the bridge between the two halves of the system.
OBS_FOR_VARIABLE = {
    "air_temp_c": "air_temp_c",
    "rh_pct": "relative_humidity_pct",
    "wind_speed_ms": "wind_speed_ms",
    "wind_gust_ms": "wind_gust_ms",
    "precip_1h_mm": "precip_1h_mm",
}


def _http(url: str, *, timeout: int = 90, retries: int = 3, headers: dict | None = None) -> bytes:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    assert last is not None
    raise last


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OBS_COLUMNS)


def _finish(out: pd.DataFrame, station_id: str, source: str) -> pd.DataFrame:
    out["station_id"] = station_id
    out["source"] = source
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[OBS_COLUMNS]


# --------------------------------------------------------------------------- NWS


def fetch_nws_hourly(stid: str, start: date, end: date) -> pd.DataFrame:
    """Recent observations from api.weather.gov (no key, ~7 days retained).

    The API returns per-METAR records with values in SI wrapped as
    ``{"value": x, "unitCode": ...}``; nulls are common and expected. We resample to the
    top of the hour the same way as the other sub-hourly sources.
    """
    params = {
        "start": f"{start.isoformat()}T00:00:00Z",
        "end": f"{end.isoformat()}T23:59:59Z",
        "limit": "500",
    }
    url = f"{NWS_API}/stations/{stid}/observations?" + urlencode(params)
    try:
        payload = json.loads(_http(url, headers={"Accept": "application/geo+json"}).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: NWS {stid} failed ({exc})", flush=True)
        return _empty()

    features = payload.get("features", [])
    if not features:
        return _empty()

    def val(props: dict, key: str):
        node = props.get(key) or {}
        v = node.get("value")
        return np.nan if v is None else float(v)

    rows = []
    for feat in features:
        p = feat.get("properties", {})
        ts = p.get("timestamp")
        if not ts:
            continue
        rows.append(
            {
                "valid_time": pd.to_datetime(ts, utc=True, format="ISO8601").tz_localize(None),
                "air_temp_c": val(p, "temperature"),
                "dewpoint_c": val(p, "dewpoint"),
                "relative_humidity_pct": val(p, "relativeHumidity"),
                "wind_speed_ms": val(p, "windSpeed") / 3.6,  # API reports km/h
                "wind_gust_ms": val(p, "windGust") / 3.6,
                "wind_dir_deg": val(p, "windDirection"),
                "precip_1h_mm": val(p, "precipitationLastHour"),
            }
        )
    if not rows:
        return _empty()

    out = pd.DataFrame(rows).set_index("valid_time").sort_index()
    out = (
        out.resample("1h")
        .agg(
            {
                "air_temp_c": "mean",
                "dewpoint_c": "mean",
                "relative_humidity_pct": "mean",
                "wind_speed_ms": "mean",
                "wind_gust_ms": "max",
                "wind_dir_deg": "median",
                "precip_1h_mm": "max",
            }
        )
        .dropna(how="all")
        .reset_index()
    )
    return _finish(out, f"NWS:{stid}", "NWS")


# --------------------------------------------------------------------------- GHCN-h


def fetch_ghcnh_hourly(ghcnh_id: str, start: date, end: date) -> pd.DataFrame:
    """Deep observation history from GHCN-hourly (NCEI), by-station period-of-record file.

    GHCN-h superseded ISD, which NOAA froze in August 2025 — new code must not depend on
    ISD. Files are pipe-separated with a header row; column names vary slightly by
    release, so we resolve them case-insensitively and tolerate absent fields.
    """
    url = f"{GHCNH_BASE}/GHCNh_{ghcnh_id}_por.psv"
    try:
        raw = _http(url, timeout=180)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: GHCN-h {ghcnh_id} unavailable ({exc})", flush=True)
        return _empty()

    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return _empty()
    try:
        df = pd.read_csv(io.StringIO(text), sep="|", low_memory=False, quoting=csv.QUOTE_NONE)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: GHCN-h {ghcnh_id} unparseable ({exc})", flush=True)
        return _empty()
    if df.empty:
        return _empty()

    lower = {c.lower(): c for c in df.columns}

    def pick(*names) -> pd.Series | None:
        for n in names:
            if n in lower:
                return pd.to_numeric(df[lower[n]], errors="coerce")
        return None

    # Timestamp is either a single DATE column or Year/Month/Day/Hour parts.
    if "date" in lower:
        vt = pd.to_datetime(df[lower["date"]], errors="coerce", utc=True).dt.tz_localize(None)
    elif all(k in lower for k in ("year", "month", "day", "hour")):
        vt = pd.to_datetime(
            {
                "year": pd.to_numeric(df[lower["year"]], errors="coerce"),
                "month": pd.to_numeric(df[lower["month"]], errors="coerce"),
                "day": pd.to_numeric(df[lower["day"]], errors="coerce"),
                "hour": pd.to_numeric(df[lower["hour"]], errors="coerce"),
            },
            errors="coerce",
        )
    else:
        return _empty()

    out = pd.DataFrame({"valid_time": vt})
    out["air_temp_c"] = pick("temperature")
    out["dewpoint_c"] = pick("dew_point_temperature")
    out["relative_humidity_pct"] = pick("relative_humidity")
    ws = pick("wind_speed")
    out["wind_speed_ms"] = ws
    out["wind_gust_ms"] = pick("wind_gust")
    out["wind_dir_deg"] = pick("wind_direction")
    precip = pick("precipitation", "precipitation_1hr", "precip")
    out["precip_1h_mm"] = precip

    out = out.dropna(subset=["valid_time"])
    mask = (out["valid_time"] >= pd.Timestamp(start)) & (
        out["valid_time"] <= pd.Timestamp(end) + pd.Timedelta(days=1)
    )
    out = out.loc[mask]
    if out.empty:
        return _empty()
    out = out.set_index("valid_time").resample("1h").mean(numeric_only=True).dropna(how="all")
    out = out.reset_index()
    return _finish(out, f"GHCNH:{ghcnh_id}", "GHCNH")


# --------------------------------------------------------------------------- SNOTEL


def fetch_snotel_hourly(triplet: str, start: date, end: date) -> pd.DataFrame:
    """Hourly SNOTEL obs (NRCS AWDB). TOBS -> air temp; PREC (cumulative) -> hourly increment."""
    params = {
        "stationTriplets": triplet,
        "elements": "TOBS,PREC",
        "duration": "HOURLY",
        "beginDate": start.isoformat(),
        "endDate": end.isoformat(),
    }
    raw = _http(f"{AWDB}/data?" + urlencode(params))
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        return _empty()

    series: dict[str, dict[str, float]] = {}
    for el in payload[0].get("data", []):
        code = el.get("stationElement", {}).get("elementCode")
        for v in el.get("values", []):
            ts, val = v.get("date"), v.get("value")
            if ts is None or val is None:
                continue
            series.setdefault(ts, {})[code] = float(val)
    if not series:
        return _empty()

    df = pd.DataFrame.from_dict(series, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    out = pd.DataFrame(index=df.index)
    out["air_temp_c"] = F_TO_C(df["TOBS"]) if "TOBS" in df else np.nan
    if "PREC" in df:
        # Cumulative for the water year; hourly increment = positive diff (negatives are
        # the sensor's end-of-season reset, not negative precipitation).
        out["precip_1h_mm"] = (df["PREC"].diff() * IN_TO_MM).clip(lower=0.0)
    else:
        out["precip_1h_mm"] = np.nan
    out = out.reset_index(names="valid_time")
    return _finish(out, triplet, "SNOTEL")


# --------------------------------------------------------------------------- IEM


def fetch_iem_hourly(iem_id: str, network: str, start: date, end: date) -> pd.DataFrame:
    """Hourly ASOS/AWOS/RAWS obs from the Iowa Environmental Mesonet bulk archive."""
    params = {
        "station": iem_id,
        "data": "tmpf,dwpf,relh,sknt,gust,drct,p01i",
        "network": network,
        "tz": "UTC",
        "format": "onlycomma",
        "missing": "empty",
        # The endpoint validates these as timezone-aware datetimes and 422s on bare dates.
        "sts": f"{start.isoformat()}T00:00Z",
        "ets": f"{(end + timedelta(days=1)).isoformat()}T00:00Z",
    }
    raw = _http(f"{IEM_URL}?" + urlencode(params))
    text = raw.decode("utf-8", errors="replace")
    if not text.strip() or "station,valid" not in text:
        return _empty()
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return _empty()

    out = pd.DataFrame()
    out["valid_time"] = pd.to_datetime(df["valid"], utc=True).dt.tz_localize(None)
    out["air_temp_c"] = F_TO_C(pd.to_numeric(df.get("tmpf"), errors="coerce"))
    out["dewpoint_c"] = F_TO_C(pd.to_numeric(df.get("dwpf"), errors="coerce"))
    out["relative_humidity_pct"] = pd.to_numeric(df.get("relh"), errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(df.get("sknt"), errors="coerce") * KT_TO_MS
    out["wind_gust_ms"] = pd.to_numeric(df.get("gust"), errors="coerce") * KT_TO_MS
    out["wind_dir_deg"] = pd.to_numeric(df.get("drct"), errors="coerce")
    out["precip_1h_mm"] = pd.to_numeric(df.get("p01i"), errors="coerce") * IN_TO_MM
    out = out.set_index("valid_time")
    out = (
        out.resample("1h")
        .agg(
            {
                "air_temp_c": "mean",
                "dewpoint_c": "mean",
                "relative_humidity_pct": "mean",
                "wind_speed_ms": "mean",
                "wind_gust_ms": "max",
                "wind_dir_deg": "median",
                "precip_1h_mm": "max",
            }
        )
        .dropna(how="all")
        .reset_index()
    )
    return _finish(out, f"IEM:{iem_id}", "IEM")


# --------------------------------------------------------------------------- Synoptic


def fetch_synoptic_hourly(stid: str, token: str, start: date, end: date) -> pd.DataFrame:
    """Hourly obs from the Synoptic timeseries API (mesonets the other sources lack)."""
    params = {
        "token": token,
        "stid": stid,
        "start": start.strftime("%Y%m%d0000"),
        "end": end.strftime("%Y%m%d2359"),
        "vars": "air_temp,dew_point_temperature,relative_humidity,wind_speed,wind_gust,wind_direction,precip_accum_one_hour",
        "units": "metric,speed|ms",
        "obtimezone": "utc",
        "output": "json",
    }
    raw = _http(f"{SYNOPTIC_TS}?" + urlencode(params))
    payload = json.loads(raw.decode("utf-8"))
    stations = payload.get("STATION", []) if isinstance(payload, dict) else []
    if not stations:
        return _empty()
    obs = stations[0].get("OBSERVATIONS", {})
    times = obs.get("date_time", [])
    if not times:
        return _empty()

    def col(key: str) -> pd.Series:
        # Synoptic suffixes variables with _set_1 etc.; take the first matching set.
        for k, v in obs.items():
            if k.startswith(key) and k != "date_time":
                return pd.Series(v, dtype="float64")
        return pd.Series([np.nan] * len(times))

    out = pd.DataFrame()
    out["valid_time"] = pd.to_datetime(times, utc=True, format="ISO8601").tz_localize(None)
    out["air_temp_c"] = col("air_temp")
    out["dewpoint_c"] = col("dew_point_temperature")
    out["relative_humidity_pct"] = col("relative_humidity")
    out["wind_speed_ms"] = col("wind_speed")
    out["wind_gust_ms"] = col("wind_gust")
    out["wind_dir_deg"] = col("wind_direction")
    out["precip_1h_mm"] = col("precip_accum_one_hour")
    out = out.set_index("valid_time").resample("1h").mean(numeric_only=True).dropna(how="all")
    out = out.reset_index()
    return _finish(out, f"SYN:{stid}", "SYNOPTIC")


# --------------------------------------------------------------------------- router


def fetch_obs(
    station_id: str,
    start: date,
    end: date,
    *,
    iem_network: str | None = None,
    ghcnh_id: str | None = None,
) -> pd.DataFrame:
    """Fetch observations for any station id, routing on its network prefix.

    ``ghcnh_id`` lets a station backed by a short-retention source (NWS) also pull deep
    history from GHCN-h under its own id; the caller merges the two.
    """
    if station_id.startswith("NWS:"):
        return fetch_nws_hourly(station_id[4:], start, end)
    if station_id.startswith("GHCNH:"):
        return fetch_ghcnh_hourly(station_id[6:], start, end)
    if station_id.startswith("IEM:"):
        if not iem_network:
            raise ValueError(f"IEM station {station_id} requires iem_network")
        return fetch_iem_hourly(station_id[4:], iem_network, start, end)
    if station_id.startswith("SYN:"):
        token = os.environ.get("SYNOPTIC_API_TOKEN")
        if not token:
            print(f"  WARN: {station_id} needs SYNOPTIC_API_TOKEN; skipping", flush=True)
            return _empty()
        return fetch_synoptic_hourly(station_id[4:], token, start, end)
    if station_id.startswith("MS:"):
        from wxfuser.data.meteostat import fetch_meteostat_hourly

        return fetch_meteostat_hourly(station_id[3:], start, end)
    # Bare id: SNOTEL triplet (e.g. "663:CO:SNTL").
    return fetch_snotel_hourly(station_id, start, end)


# --------------------------------------------------------------------------- QC

QC_BOUNDS = {
    "air_temp_c": (-70.0, 60.0),
    "dewpoint_c": (-80.0, 45.0),
    "relative_humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 110.0),
    "wind_gust_ms": (0.0, 150.0),
    "wind_dir_deg": (0.0, 360.0),
    "precip_1h_mm": (0.0, 305.0),  # above the world 1-hour record
}

QC_MAX_STEP = {
    "air_temp_c": 20.0,
    "dewpoint_c": 20.0,
    "wind_speed_ms": 60.0,
}


def qc(df: pd.DataFrame, *, persistence_hours: int = 24, copy: bool = True) -> pd.DataFrame:
    """Range, step, and flatline screens; dewpoint forced <= air temp.

    A stuck sensor reporting the same value for a day looks like a perfectly predictable
    station and would quietly poison the calibration, so runs of identical values are
    nulled rather than trusted. Rows with every measurement nulled are dropped.
    """
    if df.empty:
        return df
    src = df if not copy else df.copy()
    out = src.sort_values(["station_id", "valid_time"]).reset_index(drop=True)
    del src

    for col, (lo, hi) in QC_BOUNDS.items():
        if col in out:
            out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan

    if "dewpoint_c" in out and "air_temp_c" in out:
        out.loc[out["dewpoint_c"] > out["air_temp_c"] + 1.0, "dewpoint_c"] = np.nan

    gb = out.groupby("station_id", sort=False)
    for col, step in QC_MAX_STEP.items():
        if col in out:
            out.loc[(gb[col].diff().abs() > step).to_numpy(), col] = np.nan

    for col in ("air_temp_c", "wind_speed_ms", "relative_humidity_pct"):
        if col not in out:
            continue
        d = gb[col].diff()
        changed = d.ne(0) | d.isna()
        run_id = changed.cumsum()
        run_len = out.groupby(["station_id", run_id], sort=False).cumcount() + 1
        # Calm wind legitimately reads 0.0 for hours; only flatline non-zero runs.
        stuck = (run_len >= persistence_hours).to_numpy(copy=True)
        if col == "wind_speed_ms":
            stuck &= (out[col] > 0.05).to_numpy()
        out.loc[stuck, col] = np.nan

    meas = [c for c in OBS_COLUMNS if c not in ("station_id", "valid_time", "source")]
    return out.dropna(subset=meas, how="all").reset_index(drop=True)


def normalize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Snap valid_time to the top of the hour, dedupe, and fix column order."""
    if df.empty:
        return df
    out = df.copy()
    out["valid_time"] = pd.to_datetime(out["valid_time"]).dt.floor("h")
    out = out.drop_duplicates(["station_id", "valid_time"], keep="last").reset_index(drop=True)
    return out[OBS_COLUMNS]


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
