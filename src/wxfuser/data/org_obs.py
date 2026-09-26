"""Observations from an org's own stations: patrol study plots, avalanche-centre telemetry.

These stations are on no public network. An org gets data in two ways:

  * **CSV upload.** Usually the station's history, which is what lets a new station
    train tiers 2 and 3 from day one instead of waiting a season. Column names, units
    and time zone vary by logger, so the mapping is explicit (``ingest_csv``).
  * **Push API.** The logger, or a script beside it, posts recent readings to the Tree60
    Worker, which appends them as JSON lines to ``inbox/`` in the org's R2 prefix
    (``compact_inbox``). Fresh observations are what make short-lead corrections
    possible.

Both writers append *raw* records, at whatever interval the logger keeps, to one store
per station. Hourly aggregation happens on read (``fetch_org_hourly``): mean for
temperature, humidity and wind speed, maximum for gusts, sum for precipitation. An
hour that arrives in pieces, or a file ingested twice, therefore always aggregates the
same way. Aggregating on write would have averaged a partial hour and then frozen it.

Records are deduplicated on (station, time), last write wins, so a corrected upload
replaces what it corrects.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.data.obs import OBS_COLUMNS, gauge_increments, qc

# Where org stores live during a run. ``wxfuser org-run`` points this at the org's root.
ORG_OBS_DIR = Path(os.environ.get("WXFUSER_ORG_OBS_DIR", "observations"))

RAW_COLUMNS = ["station_id", "time", "air_temp_c", "relative_humidity_pct",
               "wind_speed_ms", "wind_gust_ms", "wind_dir_deg", "precip_mm",
               "snow_depth_cm", "swe_mm"]
AGG = {
    "air_temp_c": "mean",
    "relative_humidity_pct": "mean",
    "wind_speed_ms": "mean",
    "wind_gust_ms": "max",
    "wind_dir_deg": "median",
    "precip_mm": "sum",
    # Snowpack states, not amounts: the hour's value is its last reading.
    "snow_depth_cm": "last",
    "swe_mm": "last",
}

# Unit -> (target column family, converter to canonical units).
UNITS = {
    "C": lambda x: x,
    "F": lambda x: (x - 32.0) * 5.0 / 9.0,
    "K": lambda x: x - 273.15,
    "pct": lambda x: x,
    "m/s": lambda x: x,
    "mph": lambda x: x * 0.44704,
    "kt": lambda x: x * 0.514444,
    "km/h": lambda x: x / 3.6,
    "deg": lambda x: x,
    "mm": lambda x: x,
    "in": lambda x: x * 25.4,
    "cm": lambda x: x * 10.0,
}
# Unit converters above produce millimetres for lengths; snow depth is kept in cm.
DEPTH_IN_CM = {"snow_depth_cm"}
TARGETS = {
    "air_temp_c": {"C", "F", "K"},
    "relative_humidity_pct": {"pct"},
    "wind_speed_ms": {"m/s", "mph", "kt", "km/h"},
    "wind_gust_ms": {"m/s", "mph", "kt", "km/h"},
    "wind_dir_deg": {"deg"},
    "precip_mm": {"mm", "in", "cm"},
    "snow_depth_cm": {"cm", "in", "mm"},
    "swe_mm": {"mm", "in", "cm"},
}


class IngestError(ValueError):
    pass


def store_path(station_id: str, root: Path | None = None) -> Path:
    safe = station_id.replace(":", "_").replace("/", "_").replace(" ", "_")
    return (root or ORG_OBS_DIR) / f"{safe}.parquet"


def append_raw(records: pd.DataFrame, root: Path | None = None) -> dict[str, int]:
    """Merge raw records into each station's store. Returns rows now held per station."""
    held = {}
    for sid, block in records.groupby("station_id"):
        path = store_path(str(sid), root)
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=RAW_COLUMNS)
        merged = pd.concat([old, block[RAW_COLUMNS]], ignore_index=True)
        merged["time"] = pd.to_datetime(merged["time"])
        for col in RAW_COLUMNS[2:]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
        merged = (merged.sort_values("time", kind="stable")
                  .drop_duplicates(["station_id", "time"], keep="last")
                  .reset_index(drop=True))
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False, compression="zstd")
        held[str(sid)] = len(merged)
    return held


def fetch_org_hourly(station_id: str, start, end, root: Path | None = None) -> pd.DataFrame:
    """Hourly observations in the standard schema, aggregated from the raw store.

    Each hour is labelled by its *end*, as METARs and SNOTEL are: 10:00 covers
    09:00-10:00, so precipitation "in the hour ending 10:00" pairs with the model's
    precipitation valid at 10:00.
    """
    path = store_path(station_id, root)
    if not path.exists():
        return pd.DataFrame(columns=OBS_COLUMNS)
    raw = pd.read_parquet(path)
    raw["time"] = pd.to_datetime(raw["time"])
    lo = pd.Timestamp(start) - pd.Timedelta(hours=1)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    raw = raw[(raw["time"] > lo) & (raw["time"] <= hi)]
    if raw.empty:
        return pd.DataFrame(columns=OBS_COLUMNS)
    hourly = (raw.set_index("time")[list(AGG)]
              .resample("1h", label="right", closed="right")
              .agg(AGG))
    # sum() of an hour with no precipitation reports is 0, which would read as "dry".
    counts = raw.set_index("time")["precip_mm"].resample("1h", label="right", closed="right").count()
    hourly.loc[counts.reindex(hourly.index, fill_value=0) == 0, "precip_mm"] = np.nan
    hourly = hourly.dropna(how="all").reset_index(names="valid_time")
    out = pd.DataFrame({"station_id": station_id, "valid_time": hourly["valid_time"]})
    for col in OBS_COLUMNS:
        if col in ("station_id", "valid_time"):
            continue
        out[col] = np.nan
    for col in ("air_temp_c", "relative_humidity_pct", "wind_speed_ms", "wind_gust_ms",
                "wind_dir_deg", "snow_depth_cm", "swe_mm"):
        out[col] = hourly[col].to_numpy()
    out["precip_1h_mm"] = hourly["precip_mm"].to_numpy()
    out["source"] = "ORG"
    return qc(out[OBS_COLUMNS])


# --------------------------------------------------------------------------- CSV


def parse_mapping(specs: list[str]) -> dict[str, tuple[str, str, bool]]:
    """``target=column:unit[:cumulative]`` strings into {target: (column, unit, cumulative)}.

    ``cumulative`` marks an accumulating gauge (water-year precipitation, a tipping
    bucket's running total), whose increments are the precipitation.
    """
    out = {}
    for s in specs:
        try:
            target, rest = s.split("=", 1)
            parts = rest.split(":")
            column, unit = parts[0], parts[1]
        except (ValueError, IndexError):
            raise IngestError(f"mapping {s!r} should look like air_temp_c=TempF:F") from None
        cumulative = len(parts) > 2 and parts[2] == "cumulative"
        if target not in TARGETS:
            raise IngestError(f"unknown target {target!r}; one of {sorted(TARGETS)}")
        if unit not in TARGETS[target]:
            raise IngestError(f"{target} takes units {sorted(TARGETS[target])}, not {unit!r}")
        if cumulative and target != "precip_mm":
            raise IngestError("only precipitation can be cumulative")
        out[target] = (column, unit, cumulative)
    return out


def read_csv(
    path: str | Path,
    station_id: str,
    mapping: dict[str, tuple[str, str, bool]],
    *,
    time_col: str,
    tz: str,
) -> pd.DataFrame:
    """A logger CSV as raw records in canonical units and naive UTC.

    ``tz`` is an IANA zone ("America/Denver") or a fixed offset ("-07:00") for loggers
    kept on standard time all year, which many are. A timestamp column that already
    carries an offset is honoured as is.
    """
    df = pd.read_csv(path)
    if time_col not in df:
        raise IngestError(f"no time column {time_col!r}; columns are {list(df.columns)}")
    missing = [c for c, _, _ in mapping.values() if c not in df]
    if missing:
        raise IngestError(f"mapped columns not in file: {missing}")

    t = pd.to_datetime(df[time_col], errors="coerce")
    if t.dt.tz is None:
        if tz.upper() == "UTC":
            t = t.dt.tz_localize("UTC")
        elif tz[:1] in "+-":
            sign = 1 if tz[0] == "+" else -1
            hh, _, mm = tz[1:].partition(":")
            offset = pd.Timedelta(hours=int(hh), minutes=int(mm or 0))
            t = (t - sign * offset).dt.tz_localize("UTC")
        else:
            # Ambiguous or missing local times (DST changes) become NaT, not guesses.
            t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    t = t.dt.tz_convert("UTC").dt.tz_localize(None)

    out = pd.DataFrame({"station_id": station_id, "time": t})
    for col in RAW_COLUMNS[2:]:
        out[col] = np.nan
    for target, (column, unit, cumulative) in mapping.items():
        vals = UNITS[unit](pd.to_numeric(df[column], errors="coerce").astype(float))
        if target in DEPTH_IN_CM:
            vals = vals / 10.0  # the length converters give mm
        if cumulative:
            order = np.argsort(t.to_numpy())
            inc = gauge_increments(pd.Series(vals.to_numpy()[order]))
            vals = pd.Series(np.empty(len(inc)))
            vals.iloc[order] = inc.to_numpy()
        out[target] = vals.to_numpy()
    return out.dropna(subset=["time"]).reset_index(drop=True)


def diurnal_peak_solar_hour(records: pd.DataFrame, lon: float) -> float | None:
    """Local solar hour at which mean temperature peaks, for a time-zone sanity check.

    Temperature peaks in the afternoon almost everywhere, so a peak near dawn means the
    timestamps are in the wrong zone. SNOTEL's were, silently, for years.
    """
    temp = records.dropna(subset=["air_temp_c"])
    if len(temp) < 72:
        return None
    solar = (temp["time"].dt.hour + temp["time"].dt.minute / 60.0 + lon / 15.0) % 24
    by_hour = temp.groupby(solar.round().astype(int) % 24)["air_temp_c"].mean()
    return float(by_hour.idxmax()) if len(by_hour) >= 12 else None


# --------------------------------------------------------------------------- push inbox


def compact_inbox(inbox: Path, org: str, root: Path | None = None) -> dict[str, int]:
    """Fold pushed JSON-lines records into the stores.

    Each line is one reading in canonical units::

        {"station": "ORG:wasatch-patrol:collins", "time": "2026-01-11T14:10:00-07:00",
         "air_temp_c": -6.1, "wind_speed_ms": 7.2, "wind_gust_ms": 13.0, "precip_mm": 0.2}

    Re-reading a file is harmless (records dedupe on station and time), so files are left
    in place. A malformed line is skipped and counted rather than aborting the batch. So
    is a record for a station outside ``org``: the Worker already checks each
    station's key, and this makes sure one org's inbox can never write another org's
    station.
    """
    prefix = f"ORG:{org}:"
    rows, bad, foreign = [], 0, 0
    for f in sorted(inbox.rglob("*.jsonl")) if inbox.exists() else []:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not str(rec["station"]).startswith(prefix):
                    foreign += 1
                    continue
                t = pd.Timestamp(rec["time"])
                t = (t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")).tz_localize(None)
                row = {"station_id": str(rec["station"]), "time": t}
                for col in RAW_COLUMNS[2:]:
                    v = rec.get(col)
                    row[col] = float(v) if v is not None else np.nan
                rows.append(row)
            except (ValueError, KeyError, TypeError):
                bad += 1
    if bad:
        print(f"  inbox: skipped {bad} malformed lines", flush=True)
    if foreign:
        print(f"  inbox: refused {foreign} records for stations outside {org}", flush=True)
    if not rows:
        return {}
    return append_raw(pd.DataFrame(rows, columns=RAW_COLUMNS), root)


# --------------------------------------------------------------------------- uploads

UPLOAD_LEDGER = "uploads-ingested.json"


def ingest_uploads(uploads: Path, org: str, root: Path | None = None,
                   ledger: Path | None = None) -> list[str]:
    """Ingest uploaded CSVs that have not been ingested before.

    Each ``name.csv`` needs a sidecar ``name.json`` saying how to read it::

        {"station": "ORG:demo-patrol:plot", "time_col": "Date Time", "tz": "-07:00",
         "columns": ["air_temp_c=Air Temp (F):F", "precip_mm=Precip (in):in:cumulative"]}

    The upload form writes both. The ledger records each file's content hash, so a file
    is read once however many runs see it, while a corrected re-upload under the same
    name is read again. A file that fails is reported and retried next run, not
    recorded.
    """
    import hashlib

    ledger = ledger or (uploads.parent / "state" / UPLOAD_LEDGER)
    done = set(json.loads(ledger.read_text())) if ledger.exists() else set()
    ingested = []
    for csv_path in sorted(uploads.glob("*.csv")) if uploads.exists() else []:
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest in done:
            continue
        side = csv_path.with_suffix(".json")
        try:
            meta = json.loads(side.read_text())
            station = str(meta["station"])
            if not station.startswith(f"ORG:{org}:"):
                raise IngestError(f"{station} is not a station of {org}")
            records = read_csv(csv_path, station, parse_mapping(list(meta["columns"])),
                               time_col=meta["time_col"], tz=meta["tz"])
            append_raw(records, root)
        except (OSError, KeyError, ValueError) as exc:
            print(f"  upload {csv_path.name} not ingested: {exc}", flush=True)
            continue
        done.add(digest)
        ingested.append(csv_path.name)
    if ingested:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps(sorted(done)))
    return ingested
