"""dynamical.org zarr archives — the same model stores Tree60 serves its forecasts from.

Why a second model source, when Open-Meteo already covers most of these centres:

  * **True init/lead archives.** Each store keeps every run it has ingested (HRRR from
    2018, GFS from 2021, GEFS from 2020, IFS-ENS and AIFS from 2024) indexed by
    ``(init_time, lead_time)``. There is no seamless blend and no ``previous_dayN``
    approximation: a 30-hour forecast is exactly what the run issued 30 hours earlier
    said. That is what lets every lead bucket train on years of history on day one.
  * **The models a Tree60 user actually picks.** Calibrating a model the app does not
    show would be answering a question nobody asked. Each entry in ``configs/models.yaml``
    carries the ``tree60_id`` it corresponds to.
  * **Ensembles.** GEFS and IFS-ENS carry their members; the fused forecast reads the
    member mean here.

Issue-time alignment
--------------------
Models run on different cycles (HRRR every 6 h for its 48 h runs, GEFS and IFS-ENS once a
day) and arrive with different latencies. Keying rows on each model's own lead would put
HRRR and GEFS on different ``(valid_time, lead_h)`` rows, so ``pairs.to_wide`` would never
see them side by side. Every row is therefore re-expressed from an *issue time*: a
forecast is always issued at some moment ``T``, uses each model's newest run that had
actually arrived by ``T`` (init + latency ≤ T), and its lead is ``valid_time - T``.
Training issues at the configured hours, and live issues at "now". Both follow the
same rule, so what the tiers learn is what they are later asked to do.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from wxfuser.config import load_configs, model_catalogue, model_supports

FC_COLUMNS = ["model", "valid_time", "lead_h", "lead_source"]
LEAD_SOURCE = "dyn"

# Concurrent chunk reads per variable. Reads are latency-bound (roughly a second per
# HRRR chunk against a few MB of payload), so threads overlap the waiting rather than
# the decoding.
MAX_WORKERS = 16


class DynamicalError(RuntimeError):
    pass


def settings() -> dict:
    return dict(load_configs()["models"].get("dynamical") or {})


def is_dynamical(model_id: str) -> bool:
    return model_catalogue().get(model_id, {}).get("source") == "dynamical"


# --------------------------------------------------------------------------- store access


def _store_url(meta: dict) -> str:
    url = meta["url"]
    email = os.environ.get("WXFUSER_DYNAMICAL_EMAIL") or settings().get("email")
    # dynamical.org asks heavy users to identify themselves; it is a courtesy, not auth.
    return f"{url}?email={email}" if email else url


@lru_cache(maxsize=16)
def _open_url(url: str):
    import xarray as xr

    return xr.open_zarr(url, decode_timedelta=True, chunks=None)


def open_store(model_id: str):
    """The model's dataset, lazily opened once per process. Tests replace this."""
    meta = model_catalogue()[model_id]
    return _open_url(_store_url(meta))


# --------------------------------------------------------------------------- grid lookup


def grid_indices(ds, lats: np.ndarray, lons: np.ndarray) -> dict[str, np.ndarray]:
    """Nearest grid cell for each point, as positional indices on the spatial dims.

    Regular lat/lon grids index each axis on its own. Projected grids (HRRR's Lambert
    conformal ``y``/``x``) carry 2-D latitude/longitude and are searched directly, which
    is a few hundred milliseconds per point over 1.9 M cells and runs once per batch.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if "latitude" in ds.dims and "longitude" in ds.dims:
        glat = ds["latitude"].values
        glon = ds["longitude"].values
        lat_idx = np.abs(glat[None, :] - lats[:, None]).argmin(axis=1)
        dlon = (glon[None, :] - lons[:, None] + 180.0) % 360.0 - 180.0
        lon_idx = np.abs(dlon).argmin(axis=1)
        return {"latitude": lat_idx, "longitude": lon_idx}

    glat = ds["latitude"].values
    glon = ds["longitude"].values
    ydim, xdim = ds["latitude"].dims
    ys, xs = [], []
    for la, lo in zip(lats, lons):
        dlon = ((glon - lo + 180.0) % 360.0 - 180.0) * np.cos(np.radians(la))
        d2 = (glat - la) ** 2 + dlon**2
        j, i = np.unravel_index(np.nanargmin(d2), d2.shape)
        ys.append(j)
        xs.append(i)
    return {ydim: np.asarray(ys), xdim: np.asarray(xs)}


def _in_domain(ds, lats: np.ndarray, lons: np.ndarray, idx: dict[str, np.ndarray]) -> np.ndarray:
    """Whether each point actually lies inside the grid, not merely nearest to its edge.

    Nearest-cell lookup always succeeds, so a Montana station asked of ICON-EU would
    quietly receive a forecast for the Atlantic. Anything whose matched cell is more than
    about two grid spacings away is outside.
    """
    if "latitude" in ds.dims:
        glat = ds["latitude"].values
        step = float(np.nanmedian(np.abs(np.diff(glat)))) if glat.size > 1 else 1.0
        mlat = glat[idx["latitude"]]
        mlon = ds["longitude"].values[idx["longitude"]]
    else:
        ydim, xdim = ds["latitude"].dims
        glat = ds["latitude"].values
        step = float(np.nanmedian(np.abs(np.diff(glat[:, glat.shape[1] // 2])))) or 0.03
        mlat = glat[idx[ydim], idx[xdim]]
        mlon = ds["longitude"].values[idx[ydim], idx[xdim]]
    dlon = ((mlon - lons + 180.0) % 360.0 - 180.0) * np.cos(np.radians(lats))
    return np.hypot(mlat - lats, dlon) <= 2.5 * step


# --------------------------------------------------------------------------- variables


def _rh_from_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity from temperature and dewpoint (Magnus, over water)."""
    a, b = 17.625, 243.04
    with np.errstate(invalid="ignore", over="ignore"):
        rh = 100.0 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))
    return np.clip(rh, 0.0, 100.0)


def _raw_needed(meta: dict, variables: list[str]) -> list[str]:
    need: list[str] = []
    for var in variables:
        spec = meta.get("fields", {}).get(var)
        if not spec:
            continue
        for name in spec.get("from", [spec.get("name")]):
            if name and name not in need:
                need.append(name)
    return need


def _derive(meta: dict, var: str, raw: dict[str, np.ndarray]) -> np.ndarray | None:
    spec = meta.get("fields", {}).get(var)
    if not spec:
        return None
    kind = spec.get("kind", "direct")
    if kind == "direct":
        arr = raw.get(spec["name"])
        return None if arr is None else arr * float(spec.get("scale", 1.0))
    if kind == "speed":
        u, v = (raw.get(n) for n in spec["from"])
        return None if u is None or v is None else np.hypot(u, v)
    if kind == "rh_from_dewpoint":
        t, td = (raw.get(n) for n in spec["from"])
        return None if t is None or td is None else _rh_from_dewpoint(t, td)
    raise DynamicalError(f"unknown field kind {kind!r} for {var}")


# --------------------------------------------------------------------------- run extraction


def _read_init(ds, name: str, k: int, idx: dict[str, np.ndarray], n_leads: int) -> np.ndarray:
    """One run of one variable at every point: shape (points, leads).

    Vectorised point indexing makes zarr fetch each chunk the points fall in once,
    however many stations share it. Ensemble members are averaged here.
    """
    import xarray as xr

    da = ds[name].isel(init_time=k, lead_time=slice(0, n_leads))
    sel = {dim: xr.DataArray(ix, dims="point") for dim, ix in idx.items()}
    da = da.isel(sel)
    if "ensemble_member" in da.dims:
        da = da.mean("ensemble_member", skipna=True)
    return np.asarray(da.transpose("point", "lead_time").values, dtype="float64")


def extract_runs(
    model_id: str,
    coords: list[tuple[str, float, float]],
    init_positions: list[int],
    variables: list[str],
    *,
    ds=None,
) -> pd.DataFrame:
    """Every lead of the given runs at every station, before issue-time alignment.

    Columns: station_id, init_time, valid_time, fc_{var}. Stations outside the model's
    grid are dropped rather than handed the nearest edge cell.
    """
    meta = model_catalogue()[model_id]
    ds = ds if ds is not None else open_store(model_id)
    wanted = [v for v in variables if model_supports(model_id, v)]
    raw_names = [n for n in _raw_needed(meta, wanted) if n in ds.data_vars]
    if not coords or not init_positions or not raw_names:
        return pd.DataFrame()

    ids = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords], dtype=float)
    lons = np.array([c[2] for c in coords], dtype=float)
    idx = grid_indices(ds, lats, lons)
    inside = _in_domain(ds, lats, lons, idx)
    if not inside.any():
        return pd.DataFrame()
    ids, lats, lons = ids[inside], lats[inside], lons[inside]
    idx = {d: ix[inside] for d, ix in idx.items()}

    leads_td = pd.to_timedelta(ds["lead_time"].values)
    max_lead = pd.Timedelta(hours=int(meta.get("max_lead_h") or settings().get("max_lead_h", 168)))
    n_leads = int((leads_td <= max_lead).sum())
    leads_td = leads_td[:n_leads]
    inits = pd.to_datetime(ds["init_time"].values)

    def one(k: int) -> pd.DataFrame | None:
        raw = {}
        for name in raw_names:
            try:
                raw[name] = _read_init(ds, name, k, idx, n_leads)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: {model_id} {inits[k]} {name} unreadable ({exc})", flush=True)
        if not raw:
            return None
        init = inits[k]
        n_pts = len(ids)
        frame = pd.DataFrame({
            "station_id": np.repeat(ids, n_leads),
            "init_time": init,
            "valid_time": np.tile(init + leads_td, n_pts),
        })
        got = False
        for var in variables:
            arr = _derive(meta, var, raw) if var in wanted else None
            if arr is None:
                frame[f"fc_{var}"] = np.nan
                continue
            frame[f"fc_{var}"] = arr.reshape(-1)
            got = True
        return frame if got else None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        frames = [f for f in pool.map(one, init_positions) if f is not None]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    fc_cols = [f"fc_{v}" for v in variables]
    return out.dropna(subset=fc_cols, how="all").reset_index(drop=True)


# --------------------------------------------------------------------------- issue alignment


def latest_available(inits: pd.DatetimeIndex, issue: pd.Timestamp, latency_h: float) -> int | None:
    """Position of the newest run that had arrived by ``issue``, or None."""
    cutoff = issue - pd.Timedelta(hours=latency_h)
    pos = int(inits.searchsorted(cutoff, side="right")) - 1
    return pos if pos >= 0 else None


def issue_times(start: date, end: date, hours: list[int]) -> list[pd.Timestamp]:
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    return [d + pd.Timedelta(hours=h) for d in days for h in sorted(hours)]


def align_to_issues(
    runs: pd.DataFrame,
    model_id: str,
    issues: list[pd.Timestamp],
    run_for_issue: dict[pd.Timestamp, pd.Timestamp],
    variables: list[str],
    *,
    max_lead_h: int,
    lead_source: str = LEAD_SOURCE,
) -> pd.DataFrame:
    """Re-express run rows from each issue time: lead is ``valid_time - issue``.

    A run can serve several issue times (GEFS's single daily run serves all four), so
    rows are duplicated per issue with their lead recomputed. Only strictly future hours
    within the lead horizon are kept: lead 0 is an analysis, not a forecast.
    """
    if runs.empty:
        return pd.DataFrame(columns=[*FC_COLUMNS, "station_id", *(f"fc_{v}" for v in variables)])
    by_init = dict(tuple(runs.groupby("init_time")))
    frames = []
    for issue in issues:
        init = run_for_issue.get(issue)
        block = by_init.get(init) if init is not None else None
        if block is None:
            continue
        lead = (block["valid_time"] - issue) / pd.Timedelta(hours=1)
        keep = (lead >= 1) & (lead <= max_lead_h)
        if not keep.any():
            continue
        b = block.loc[keep].copy()
        b["lead_h"] = lead[keep].round().astype(int)
        b["issue_time"] = issue
        frames.append(b)
    if not frames:
        return pd.DataFrame(columns=[*FC_COLUMNS, "station_id", *(f"fc_{v}" for v in variables)])
    out = pd.concat(frames, ignore_index=True)
    out["model"] = model_id
    out["lead_source"] = lead_source
    cols = ["station_id", *FC_COLUMNS, *(f"fc_{v}" for v in variables)]
    return out[cols].reset_index(drop=True)


def _plan_runs(
    ds, issues: list[pd.Timestamp], latency_h: float, max_lead_h: int, run_lead_h: float
) -> tuple[list[int], dict[pd.Timestamp, pd.Timestamp]]:
    """Which runs to read, and which run serves each issue time.

    An issue whose newest arrived run is older than the run could still cover (its last
    lead minus the horizon we need, at least one hour) is left unserved rather than
    answered from a stale run — a gap in the store is a gap, not a longer lead.
    """
    inits = pd.DatetimeIndex(pd.to_datetime(ds["init_time"].values))
    chosen: dict[pd.Timestamp, pd.Timestamp] = {}
    positions: set[int] = set()
    for issue in issues:
        pos = latest_available(inits, issue, latency_h)
        if pos is None:
            continue
        age_h = (issue - inits[pos]) / pd.Timedelta(hours=1)
        if age_h >= run_lead_h:
            continue
        chosen[issue] = inits[pos]
        positions.add(pos)
    return sorted(positions), chosen


def _run_lead_h(ds, meta: dict) -> float:
    leads = pd.to_timedelta(ds["lead_time"].values)
    return float(leads.max() / pd.Timedelta(hours=1)) if len(leads) else 0.0


# --------------------------------------------------------------------------- public API


def fetch_archive_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    start: date,
    end: date,
    *,
    issue_hours: list[int] | None = None,
    open_fn=None,
) -> pd.DataFrame:
    """Issue-aligned archived forecasts for many stations.

    Output matches the Open-Meteo batch frames: station_id, model, valid_time, lead_h,
    lead_source (``dyn``), fc_{var}.
    """
    cfg = settings()
    hours = issue_hours if issue_hours is not None else list(cfg.get("issue_hours_utc", [0, 6, 12, 18]))
    max_lead_h = int(cfg.get("max_lead_h", 168))
    issues = issue_times(start, end, hours)
    catalogue = model_catalogue()
    opener = open_fn or open_store

    frames = []
    for model in models:
        meta = catalogue[model]
        ds = opener(model)
        positions, chosen = _plan_runs(
            ds, issues, float(meta.get("latency_h", 3)), max_lead_h, _run_lead_h(ds, meta)
        )
        if not positions:
            print(f"  dynamical {model}: no runs in {start}..{end}", flush=True)
            continue
        print(f"  dynamical {model}: reading {len(positions)} runs for {len(coords)} stations",
              flush=True)
        runs = extract_runs(model, coords, positions, variables, ds=ds)
        aligned = align_to_issues(
            runs, model, issues, chosen, variables,
            max_lead_h=min(max_lead_h, int(meta.get("max_lead_h") or max_lead_h)),
        )
        print(f"  dynamical {model}: {len(aligned)} rows", flush=True)
        if not aligned.empty:
            frames.append(aligned)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_live_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    *,
    now: datetime | pd.Timestamp | None = None,
    open_fn=None,
) -> pd.DataFrame:
    """The forecast as it can be issued right now, in the same issue-aligned form.

    Leads are measured from the current hour, and each model contributes its newest run
    that has arrived. That is exactly the rule the archive was aligned by.
    """
    issue = pd.Timestamp(now or pd.Timestamp.utcnow()).tz_localize(None).floor("h")
    cfg = settings()
    max_lead_h = int(cfg.get("max_lead_h", 168))
    catalogue = model_catalogue()
    opener = open_fn or open_store

    frames = []
    for model in models:
        meta = catalogue[model]
        ds = opener(model)
        positions, chosen = _plan_runs(
            ds, [issue], float(meta.get("latency_h", 3)), max_lead_h, _run_lead_h(ds, meta)
        )
        # The store publishes a run's metadata before every lead has landed, so the newest
        # run can be partly empty. Step back to the previous one when it is.
        runs = extract_runs(model, coords, positions, variables, ds=ds) if positions else pd.DataFrame()
        if runs.empty and positions and positions[0] > 0:
            prev = positions[0] - 1
            inits = pd.to_datetime(ds["init_time"].values)
            chosen = {issue: inits[prev]}
            runs = extract_runs(model, coords, [prev], variables, ds=ds)
        aligned = align_to_issues(runs, model, [issue], chosen, variables, max_lead_h=max_lead_h,
                                  lead_source="live")
        if not aligned.empty:
            frames.append(aligned)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

