"""Model forecast archives at station points: every run, extracted once.

Training one model across a whole network pairs years of observations with years of
forecasts at every station. Reading the zarr stores per training run would re-download
the same chunks every time: for HRRR at Montana's SNOTEL sites that is about 20 MB
per run per variable, over 700 GB for 2018 onward. So each run is read once, reduced
to the station points, and kept.

What is stored is the raw run: station, init time, lead, and each canonical variable.
Issue-time alignment (dynamical.align_to_issues) happens at training time, so choices
like issue hours or latency can change without re-extracting anything.

Layout: ``{out}/{model}/{YYYY-MM}.parquet`` locally and ``points/{model}/{YYYY-MM}.parquet``
in the hub dataset (nakas/wxfuser-archive). A month is written only once complete, so
a build resumes by skipping months that already exist here or on the hub.
"""
from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.data import derived, dynamical

VARIABLES = ["air_temp_c", "rh_pct", "wind_speed_ms", "wind_gust_ms", "precip_1h_mm"]


def snotel_points(states: list[str] | None) -> list[tuple[str, float, float]]:
    from wxfuser.data.bulk import snotel_stations

    meta = snotel_stations()
    if states:
        meta = meta[meta["state"].isin([s.upper() for s in states])]
    return [(r.id, float(r.lat), float(r.lon)) for r in meta.itertuples()]


def months(start: date, end: date) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def month_positions(ds, month: str, init_hours: list[int] | None) -> list[int]:
    inits = pd.DatetimeIndex(pd.to_datetime(ds["init_time"].values))
    period = pd.Period(month, freq="M")
    ok = (inits >= period.start_time) & (inits <= period.end_time)
    if init_hours is not None:
        ok &= np.isin(inits.hour, init_hours)
    return [int(i) for i in np.nonzero(ok)[0]]


def extract_month(model: str, coords, month: str, *, init_hours=None, ds=None) -> pd.DataFrame:
    """Every run of ``model`` initialised in ``month``, at every point."""
    ds = ds if ds is not None else dynamical.open_store(model)
    positions = month_positions(ds, month, init_hours)
    if not positions:
        return pd.DataFrame()
    fetch = derived.fetch_variables(VARIABLES)
    runs = dynamical.extract_runs(model, coords, positions, fetch, ds=ds)
    if runs.empty:
        return runs
    runs["lead_h"] = ((runs["valid_time"] - runs["init_time"]) / pd.Timedelta(hours=1)).round().astype("int16")
    runs["model"] = model
    for c in [c for c in runs.columns if c.startswith("fc_")]:
        runs[c] = runs[c].astype("float32")
    return runs[["station_id", "model", "init_time", "lead_h", "valid_time",
                 *[c for c in runs.columns if c.startswith("fc_")]]]


def _hub_existing(repo: str, model: str) -> set[str]:
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception:  # noqa: BLE001
        return set()
    return {Path(f).stem for f in files if f.startswith(f"points/{model}/") and f.endswith(".parquet")}


def _upload(repo: str, path: Path, model: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    for attempt in range(1, 4):
        try:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"points/{model}/{path.name}",
                            repo_id=repo, repo_type="dataset",
                            commit_message=f"{model} {path.stem} at station points")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  upload attempt {attempt} failed ({exc})", flush=True)
            time.sleep(60 * attempt)
    raise RuntimeError(f"could not upload {path}")


def build(model: str, *, states: list[str] | None, out: str | Path, start: date | None = None,
          end: date | None = None, init_hours: list[int] | None = None,
          shard: int = 0, of: int = 1, hub_repo: str | None = None,
          budget_s: float | None = None) -> dict:
    """Extract every month of ``model`` runs for this shard, skipping finished months.

    Months are dealt to shards round-robin. ``budget_s`` stops starting new months when
    time runs short, so a CI job ends cleanly and the next run carries on.
    """
    t0 = time.monotonic()
    out = Path(out) / model
    out.mkdir(parents=True, exist_ok=True)
    coords = snotel_points(states)
    ds = dynamical.open_store(model)
    first = pd.Timestamp(ds["init_time"].values[0]).date()
    last = pd.Timestamp(ds["init_time"].values[-1]).date()
    start, end = max(start or first, first), min(end or last, last)
    # The current month is still filling; it is extracted once it has ended.
    all_months = [m for m in months(start, end) if pd.Period(m, freq="M").end_time.date() < date.today()]
    mine = [m for i, m in enumerate(all_months) if i % of == shard]
    done = {p.stem for p in out.glob("*.parquet")} | (_hub_existing(hub_repo, model) if hub_repo else set())
    todo = [m for m in mine if m not in done]
    print(f"{model}: {len(coords)} stations, {len(all_months)} months, shard {shard + 1}/{of}: "
          f"{len(mine)} mine, {len(todo)} to do", flush=True)
    written = []
    for month in todo:
        if budget_s and time.monotonic() - t0 > budget_s:
            print(f"  time budget reached; {len(todo) - len(written)} months left for the next run")
            break
        t = time.monotonic()
        frame = extract_month(model, coords, month, init_hours=init_hours, ds=ds)
        if frame.empty:
            print(f"  {month}: no runs", flush=True)
            continue
        path = out / f"{month}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        if hub_repo:
            _upload(hub_repo, path, model)
        written.append(month)
        print(f"  {month}: {frame['init_time'].nunique()} runs, {len(frame):,} rows, "
              f"{path.stat().st_size / 1e6:.1f} MB, {time.monotonic() - t:.0f}s", flush=True)
    return {"model": model, "written": written, "remaining": len(todo) - len(written)}
