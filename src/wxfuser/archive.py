"""The full SNOTEL observation archive: every station, its whole hourly record.

The per-station pipeline fetches what one calibration needs. Training one model across
the whole network needs every station's complete history in one place, fetched once:
hourly temperature, precipitation, snow depth and SWE, back to when each station's
hourly record began (Snowbird's starts in December 2001).

Everything goes through the same normalisation as the live pipeline: station-local
standard time converted to UTC, gauge increments without jitter. So the archive and
the live observations are the same quantity.

Layout: ``{out}/obs/{triplet as slug}.parquet`` in the standard observation schema,
plus ``{out}/stations.parquet`` with each station's metadata. A station is written
whole, and only after all its windows arrive, so an interrupted build resumes by
skipping stations that already have a file.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from wxfuser.data import bulk

HOURLY_FROM = date(2000, 1, 1)   # AWDB returns nothing earlier; each station starts later
# Stations per request. Small on purpose: a batch's whole record is merged in memory
# before normalising (gauge increments need it continuous), and 25 years of four
# elements for 40 stations is tens of millions of values per worker. With 5, AWDB's
# size cap still allows 600-day windows.
BATCH = 5


def slug(triplet: str) -> str:
    return triplet.replace(":", "_")


def build_snotel_archive(out: str | Path, *, start: date = HOURLY_FROM,
                         end: date | None = None, workers: int = 4,
                         limit: int | None = None, states: list[str] | None = None) -> dict:
    """Fetch every SNOTEL station's hourly history into ``out``. Resumable.

    ``workers`` batches run concurrently. AWDB is a public service, and four keeps the
    request rate modest while making a multi-hour build finish in hours rather than a
    day.
    """
    out = Path(out)
    (out / "obs").mkdir(parents=True, exist_ok=True)
    end = end or date.today()
    meta = bulk.snotel_stations()
    if meta.empty:
        raise RuntimeError("the SNOTEL station list is unavailable")
    if states:
        meta = meta[meta["state"].isin([st.upper() for st in states])].reset_index(drop=True)
    meta.to_parquet(out / "stations.parquet", index=False)
    triplets = [t for t in meta["id"] if not (out / "obs" / f"{slug(t)}.parquet").exists()]
    if limit:
        triplets = triplets[:limit]
    print(f"SNOTEL archive: {len(meta)} stations, {len(meta) - len(triplets)} already done, "
          f"{len(triplets)} to fetch, {start}..{end}", flush=True)

    batches = [triplets[i : i + BATCH] for i in range(0, len(triplets), BATCH)]
    done = rows = 0

    def one(batch: list[str]) -> tuple[int, int]:
        frame = bulk.snotel_observations(batch, start, end, chunk=BATCH)
        n = 0
        if not frame.empty:
            for sid, g in frame.groupby("station_id"):
                g = g.sort_values("valid_time").reset_index(drop=True)
                g.to_parquet(out / "obs" / f"{slug(str(sid))}.parquet", index=False,
                             compression="zstd")
                n += len(g)
        return len(batch), n

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, b) for b in batches]
        for f in as_completed(futures):
            try:
                k, n = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: batch failed ({exc}); rerun to resume", flush=True)
                continue
            done += k
            rows += n
            print(f"  {done}/{len(triplets)} stations, {rows:,} hourly rows", flush=True)
    have = len(list((out / "obs").glob("*.parquet")))
    return {"stations": len(meta), "archived": have, "rows_this_run": rows}


def summary(out: str | Path) -> pd.DataFrame:
    """Per-station span and completeness of an archive, for a quick look."""
    rows = []
    for p in sorted(Path(out, "obs").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["valid_time", "air_temp_c", "precip_1h_mm",
                                         "snow_depth_cm", "swe_mm"])
        rows.append({
            "station": p.stem, "first": df["valid_time"].min(), "last": df["valid_time"].max(),
            "hours": len(df), **{f"{c}_frac": float(df[c].notna().mean())
                                 for c in ("air_temp_c", "precip_1h_mm", "snow_depth_cm", "swe_mm")},
        })
    return pd.DataFrame(rows)


HUB_REPO = "nakas/wxfuser-archive"

CARD = """---
license: other
pretty_name: wxfuser station archive
tags: [weather, snotel, forecast-verification, post-processing, mountain-weather]
---

# wxfuser station archive

Paired material for calibrating weather forecasts at mountain stations: station
observations, and archived model forecasts extracted at the same points. It is built
by [wxStationFuser](https://github.com/andrewnakas/wxStationFuser) for Tree60 Weather.

## `snotel/`

The hourly record of NRCS SNOTEL stations (currently Montana), from each station's
first hourly report.

- `stations.parquet`: triplet, name, latitude, longitude, elevation (m), state.
- `obs/<triplet>.parquet`: `valid_time` (UTC), `air_temp_c`, `precip_1h_mm`,
  `snow_depth_cm`, `swe_mm`, and the other columns of the wxfuser observation schema.

Processing, relative to the raw AWDB service:

- AWDB stamps hourly data in station **standard** time (PST for western SNOTEL). It
  is converted to UTC here.
- `precip_1h_mm` is the increment of the cumulative gauge after removing sensor
  jitter (a future-minimum filter). Plain positive differences overcount summer
  precipitation roughly tenfold.
- Values are range- and step-checked, and stuck sensors are nulled.

Source: USDA NRCS National Water and Climate Center, AWDB REST API (public domain).

## `points/<model>/<YYYY-MM>.parquet`

Every archived run of a model initialised in that month, at every station point
(nearest grid cell): `station_id`, `model`, `init_time`, `lead_h`, `valid_time`, and
the forecast columns `fc_air_temp_c`, `fc_rh_pct`, `fc_wind_speed_ms`,
`fc_wind_gust_ms`, `fc_precip_1h_mm` (mean hourly rate over the step, mm).
Ensemble models are the member mean.

| model | runs | source store |
|---|---|---|
| hrrr | 00z and 12z, 2018-07 on, 48 h | NOAA HRRR via dynamical.org |
| gefs | 00z, 2020-10 on, to 168 h | NOAA GEFS via dynamical.org |
| ecmwf_ens | 00z, 2024-04 on, to 168 h | ECMWF IFS ENS via dynamical.org |

Forecast data comes from [dynamical.org](https://dynamical.org) archives of NOAA and
ECMWF open data. See dynamical.org and each originating centre for licensing and
attribution terms. ECMWF open data is CC BY 4.0.
"""


def upload_to_hub(out: str | Path, repo: str = HUB_REPO) -> None:
    """Publish an observation archive under ``snotel/``, with the dataset card."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset", commit_message="dataset card")
    api.upload_folder(folder_path=str(out), path_in_repo="snotel", repo_id=repo,
                      repo_type="dataset", allow_patterns=["stations.parquet", "obs/*.parquet"],
                      commit_message="SNOTEL hourly observation archive")
