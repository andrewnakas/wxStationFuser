"""Meteostat — global raw-density surface stations (~22k worldwide, key-free bulk).

Meteostat aggregates NOAA ISD/GHCN, SYNOP/METAR, and national-service hourly data into a
free bulk archive: ~22,000 stations, genuinely global (EMEA ~9k, Asia/Oceania ~6k,
Americas ~7k) — 3× our ASOS+IEM catalogue and, unlike those, not US-skewed. No API key.

Two bulk endpoints:
  - stations:  https://bulk.meteostat.net/v2/stations/full.json.gz  (all stations + coords)
  - hourly:    https://bulk.meteostat.net/v2/hourly/{year}/{id}.csv.gz  (per station-year)

Station ids are ``MS:<id>``. The hourly CSV is column-positional (no header); mapping per
the Meteostat bulk spec. Wind is km/h -> m/s; temp/dewpoint already °C; precip already mm.
"""
from __future__ import annotations

import gzip
import io
import json
from datetime import date
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BULK = "https://bulk.meteostat.net/v2"
KMH_TO_MS = 1000.0 / 3600.0

# Positional columns of the hourly bulk CSV (Meteostat bulk data spec).
HOURLY_COLS = ["d", "h", "temp", "dwpt", "rhum", "prcp", "snow", "wdir",
               "wspd", "wpgt", "pres", "tsun", "coco"]


def _get_gz(url: str, *, timeout: int = 120, retries: int = 3) -> bytes | None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "wxfuser/0.1"})
            with urlopen(req, timeout=timeout) as r:
                return gzip.decompress(r.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            import time
            # 404 = no data for this station-year; don't retry, just skip.
            if "404" in str(exc):
                return None
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    if last:
        raise last
    return None


def select_stations(bounds: dict) -> pd.DataFrame:
    """All Meteostat stations inside the region bbox, with lat/lon/elevation."""
    raw = _get_gz(f"{BULK}/stations/full.json.gz")
    data = json.loads(raw)
    rows = []
    for s in data:
        loc = s.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is None or lon is None:
            continue
        name = s.get("name") or {}
        rows.append({
            "station_id": f"MS:{s.get('id')}",
            "ms_id": s.get("id"),
            "name": (name.get("en") if isinstance(name, dict) else name) or s.get("id"),
            "network": "METEOSTAT",
            "state": s.get("country"),
            "lat": float(lat),
            "lon": float(lon),
            "elevation_m": (float(loc["elevation"]) if loc.get("elevation") is not None else None),
        })
    df = pd.DataFrame(rows)
    b = bounds
    df = df[df["lon"].between(b["lon_min"], b["lon_max"])
            & df["lat"].between(b["lat_min"], b["lat_max"])].reset_index(drop=True)
    return df


def fetch_meteostat_hourly(ms_id: str, start: date, end: date) -> pd.DataFrame:
    """Hourly obs for one Meteostat station over [start, end], normalized to OBS_COLUMNS.

    Pulls the per-year bulk CSVs spanning the window, concatenates, filters to the range."""
    from wxfuser.data.obs import OBS_COLUMNS

    frames = []
    for year in range(start.year, end.year + 1):
        raw = _get_gz(f"{BULK}/hourly/{year}/{ms_id}.csv.gz")
        if not raw:
            continue
        try:
            df = pd.read_csv(io.BytesIO(raw), header=None, names=HOURLY_COLS)
        except Exception:  # noqa: BLE001 — malformed year file, skip
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=OBS_COLUMNS)
    df = pd.concat(frames, ignore_index=True)

    vt = pd.to_datetime(df["d"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(df["h"], errors="coerce").fillna(0), unit="h")
    out = pd.DataFrame({"valid_time": vt})
    out["air_temp_c"] = pd.to_numeric(df["temp"], errors="coerce")
    out["dewpoint_c"] = pd.to_numeric(df["dwpt"], errors="coerce")
    out["relative_humidity_pct"] = pd.to_numeric(df["rhum"], errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(df["wspd"], errors="coerce") * KMH_TO_MS
    out["wind_gust_ms"] = pd.to_numeric(df["wpgt"], errors="coerce") * KMH_TO_MS
    out["wind_dir_deg"] = pd.to_numeric(df["wdir"], errors="coerce")
    out["precip_1h_mm"] = pd.to_numeric(df["prcp"], errors="coerce").clip(lower=0)
    out["station_id"] = f"MS:{ms_id}"
    out["source"] = "METEOSTAT"
    out = out.dropna(subset=["valid_time"])
    mask = (out["valid_time"] >= pd.Timestamp(start)) & (out["valid_time"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    out = out[mask]
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[OBS_COLUMNS]
