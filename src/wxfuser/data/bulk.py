"""Bulk observation sources — what makes enrolling thousands of stations possible.

The per-station fetchers in ``obs.py`` are the right tool for one station: they hit an
API, get a week or a year of history, and stop. They are the wrong tool for five thousand
stations, because the cost is linear in stations and every source rate-limits.

These two sources are queryable in bulk instead, so the cost is linear in *time span*
rather than in station count:

  * **dynamical.org ASOS** — a year-partitioned GeoParquet of global airport observations
    from 1940 to now, served over HTTP and readable by DuckDB with predicate pushdown.
    One query returns hourly temperature, dewpoint, humidity, wind, gust and precipitation
    for every airport in a region. Roughly 3,800 stations report reliably.
  * **SNOTEL** (NRCS AWDB) — ~970 US snow-telemetry sites, nearly all in complex terrain
    at a median elevation above 2,200 m. These are the stations where a model's idea of
    the terrain is most wrong, and therefore where post-processing has the most to fix.

Together they are the enrollable universe: airports give wind and gust ground truth that
snow telemetry lacks, and SNOTEL gives mountains that airports mostly avoid.
"""
from __future__ import annotations

import json
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from wxfuser.data.obs import KT_TO_MS, OBS_COLUMNS, USER_AGENT

ASOS_BASE = "https://data.source.coop/dynamical/asos-parquet"
AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

FT_TO_M = 0.3048

# A station present in the archive but barely reporting cannot be calibrated. 4,000 hourly
# observations is roughly half a year of coverage.
MIN_OBS_PER_YEAR = 4000


def _duckdb():
    """A DuckDB connection able to read remote parquet."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false;")
    # The archive stores `valid` as a timestamptz, which DuckDB renders in the session's
    # timezone — inherited from the host. Left alone, the same query returns Mountain time
    # on a laptop in Colorado and UTC on a CI runner, and those observations would then be
    # paired against UTC forecast times with a station-dependent offset. Pinning the
    # session to UTC makes the result identical everywhere and aligned with the pipeline.
    con.execute("SET TimeZone='UTC';")
    return con


def _year_urls(start_year: int, end_year: int) -> list[str]:
    return [f"{ASOS_BASE}/year={y}/data.parquet" for y in range(start_year, end_year + 1)]


# --------------------------------------------------------------------------- ASOS


def asos_stations(year: int | None = None, min_obs: int = MIN_OBS_PER_YEAR) -> pd.DataFrame:
    """Every ASOS station reporting reliably in ``year``.

    Returns the catalogue columns the registry needs, filtered to stations with enough
    observations to be worth calibrating.
    """
    year = year or date.today().year - 1
    con = _duckdb()
    url = f"{ASOS_BASE}/year={year}/data.parquet"
    q = f"""
        SELECT station,
               any_value("name")      AS sname,
               any_value(latitude)    AS lat,
               any_value(longitude)   AS lon,
               any_value(elevation)   AS elev_m,
               any_value(country)     AS country,
               any_value(state)       AS state,
               count(*)               AS n_obs
        FROM read_parquet('{url}')
        GROUP BY station
        HAVING count(*) >= {int(min_obs)}
    """
    df = con.execute(q).fetchdf()
    df = df.dropna(subset=["lat", "lon"])
    df["id"] = "ASOS:" + df["station"].astype(str)
    df["name"] = df["sname"].fillna(df["station"])
    df["network"] = "ASOS"
    return df[["id", "station", "name", "lat", "lon", "elev_m", "country", "state", "n_obs"]]


def asos_observations(
    stations: list[str], start: date, end: date, *, chunk: int = 400
) -> pd.DataFrame:
    """Hourly observations for many ASOS stations at once, in the common schema.

    Station ids are pushed into the query so DuckDB can skip row groups. Splitting into
    chunks keeps the generated SQL and the returned frame to a sane size; the cost is
    dominated by the number of years scanned, not by the number of stations asked for.
    """
    if not stations:
        return pd.DataFrame(columns=OBS_COLUMNS)

    con = _duckdb()
    urls = _year_urls(start.year, end.year)
    url_list = ", ".join(f"'{u}'" for u in urls)
    frames = []

    for i in range(0, len(stations), chunk):
        batch = [s.split(":", 1)[-1] for s in stations[i : i + chunk]]
        in_list = ", ".join(f"'{s}'" for s in batch)
        q = f"""
            SELECT station, valid, tmpc, dwpc, relh, sknt, gust, drct, p01m
            FROM read_parquet([{url_list}])
            WHERE station IN ({in_list})
              AND valid >= TIMESTAMP '{start.isoformat()} 00:00:00'
              AND valid <  TIMESTAMP '{end.isoformat()} 00:00:00'
        """
        try:
            df = con.execute(q).fetchdf()
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: ASOS chunk {i // chunk} failed ({exc})", flush=True)
            continue
        if df.empty:
            continue
        frames.append(_normalise_asos(df))
        print(f"  ASOS: {min(i + chunk, len(stations))}/{len(stations)} stations, "
              f"{sum(len(f) for f in frames)} rows", flush=True)

    if not frames:
        return pd.DataFrame(columns=OBS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _normalise_asos(df: pd.DataFrame) -> pd.DataFrame:
    """Map the archive's columns onto the observation schema.

    Temperature and precipitation are already metric here (tmpc, p01m); wind is in knots.
    """
    out = pd.DataFrame()
    out["station_id"] = "ASOS:" + df["station"].astype(str)
    # Land on UTC-naive timestamps, the one representation the paired archive uses.
    # Belt and braces alongside the session timezone: if a future DuckDB returns these
    # tz-aware anyway, converting here still yields the correct instant rather than a
    # silently shifted one.
    valid = pd.to_datetime(df["valid"])
    if getattr(valid.dt, "tz", None) is not None:
        valid = valid.dt.tz_convert("UTC").dt.tz_localize(None)
    out["valid_time"] = valid.dt.floor("h")
    out["air_temp_c"] = pd.to_numeric(df["tmpc"], errors="coerce")
    out["dewpoint_c"] = pd.to_numeric(df["dwpc"], errors="coerce")
    out["relative_humidity_pct"] = pd.to_numeric(df["relh"], errors="coerce")
    out["wind_speed_ms"] = pd.to_numeric(df["sknt"], errors="coerce") * KT_TO_MS
    out["wind_gust_ms"] = pd.to_numeric(df["gust"], errors="coerce") * KT_TO_MS
    out["wind_dir_deg"] = pd.to_numeric(df["drct"], errors="coerce")
    out["precip_1h_mm"] = pd.to_numeric(df["p01m"], errors="coerce")
    out["source"] = "ASOS"

    # Sub-hourly METARs collapse to the top of the hour, matching the other sources:
    # mean for state variables, max for gust and precipitation.
    agg = {
        "air_temp_c": "mean", "dewpoint_c": "mean", "relative_humidity_pct": "mean",
        "wind_speed_ms": "mean", "wind_gust_ms": "max", "wind_dir_deg": "median",
        "precip_1h_mm": "max",
    }
    out = (
        out.groupby(["station_id", "valid_time"], as_index=False)
        .agg(agg)
        .assign(source="ASOS")
    )
    for c in OBS_COLUMNS:
        if c not in out:
            out[c] = np.nan
    return out[OBS_COLUMNS]


# --------------------------------------------------------------------------- SNOTEL


def _http_json(url: str, timeout: int = 120):
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def snotel_stations() -> pd.DataFrame:
    """Every SNOTEL site, from the NRCS station metadata service.

    The service returns neighbouring networks alongside SNOTEL, so the triplet is filtered
    explicitly rather than trusted from the query parameter.
    """
    params = {"networkCds": "SNTL", "returnStationElements": "false"}
    try:
        data = _http_json(f"{AWDB}/stations?" + urlencode(params))
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: SNOTEL station list failed ({exc})", flush=True)
        return pd.DataFrame(columns=["id", "name", "lat", "lon", "elev_m", "country", "state"])

    rows = []
    for s in data:
        triplet = s.get("stationTriplet") or ""
        if ":SNTL" not in triplet:
            continue
        lat, lon = s.get("latitude"), s.get("longitude")
        if lat is None or lon is None:
            continue
        elev = s.get("elevation")
        rows.append(
            {
                "id": triplet,          # bare triplet is the SNOTEL id the fetcher expects
                "station": triplet,
                "name": s.get("name") or triplet,
                "lat": float(lat),
                "lon": float(lon),
                # AWDB reports feet.
                "elev_m": float(elev) * FT_TO_M if elev is not None else None,
                "country": "US",
                "state": (triplet.split(":")[1] if triplet.count(":") >= 1 else None),
                "n_obs": None,
            }
        )
    return pd.DataFrame(rows)


def enrollable_universe(
    *,
    include_asos: bool = True,
    include_snotel: bool = True,
    min_elevation_m: float | None = None,
    countries: list[str] | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """The set of stations that can be enrolled, from both bulk sources.

    ``min_elevation_m`` is the mountain filter: SNOTEL sits above it almost by definition,
    and it selects the airports on plateaus and in mountain valleys where the model's
    terrain is furthest from the station's.
    """
    frames = []
    if include_asos:
        print("querying dynamical.org ASOS archive…", flush=True)
        a = asos_stations(year=year)
        a["network"] = "ASOS"
        frames.append(a)
        print(f"  {len(a)} ASOS stations reporting reliably", flush=True)
    if include_snotel:
        print("querying SNOTEL station list…", flush=True)
        s = snotel_stations()
        s["network"] = "SNOTEL"
        frames.append(s)
        print(f"  {len(s)} SNOTEL stations", flush=True)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    if min_elevation_m is not None:
        before = len(df)
        df = df[df["elev_m"].fillna(-1) >= min_elevation_m]
        print(f"  elevation >= {min_elevation_m:.0f} m: {len(df)} of {before}", flush=True)
    if countries:
        df = df[df["country"].isin(countries)]
        print(f"  in {countries}: {len(df)}", flush=True)

    return df.reset_index(drop=True)
