"""Forecasts read straight from dynamical.org's gridded archives.

The point of this module is *lead-resolved history*, which is the thing the calibration
has been starved of.

Open-Meteo serves two archives and neither is what a multi-day calibration wants. The
seamless historical archive reaches back years but stores the best-available forecast, so
every row of it is short-lead — ``fetch_historical`` tags them lead 3 h and says so. The
previous-runs archive is genuinely lead-resolved and is therefore the only honest basis
for day-2-to-day-7 skill, and it reaches back **92 days**, at daily lead offsets. So the
coefficients for the 97-168 h bucket have been fitted on about 92 samples per station.

dynamical.org publishes the raw forecast archives as Zarr: every model run since
2021-05, every lead hour, on the native grid. The same station gets roughly 730 samples
per lead from two years of daily initialisations, at hourly lead resolution rather than
daily — 8x the data at the leads where the fitted spread matters most, and the lead axis
is real rather than nominal.

**Why this is affordable, which is not obvious.** These are 4-D arrays of the whole globe
and the naive reading is that extracting one station means downloading the planet. The
chunking is what makes it work: GFS is chunked (1 init, 105 leads, 121 lat, 121 lon), so
one chunk read covers a 30-degree tile — and 2,422 of our stations sit inside the busiest
one. Reading per tile and indexing the stations out of the slab in memory turns "one
request per station" into "one request per tile". Measured against the live archive: 0.23 s
and 3.1 MB per (init, variable, tile), 59 occupied tiles for 9,040 stations, so a two-year
backfill of five variables is about 14 hours of transfer for the entire fleet — sharded,
under an hour per worker.

Read it per station instead and the same backfill is 50,000 times the traffic for the same
numbers. The tile grouping is not an optimisation here; it is the whole reason this is
possible.

The ensembles were measured too and are not affordable: GEFS (31 members) chunks 17x16
cells, so the fleet spans 811 tiles and the same backfill costs 4.3 TB; IFS ENS is 5.7 TB.
Those numbers are why this module carries the deterministic models only.
"""
from __future__ import annotations

import functools
from datetime import date, datetime

import numpy as np
import pandas as pd

CATALOG_URL = "https://stac.dynamical.org/catalog.json"

# Our model ids -> the STAC collection that serves them. Deliberately prefixed: a station
# fused from `dyn_gfs` is calibrated against dynamical's grid-cell values, and Open-Meteo's
# `gfs_seamless` is the same model interpolated to the point by somebody else's code. The
# two are close but not identical, and coefficients fitted on one must not be applied to
# the other — so they are different models as far as this system is concerned.
COLLECTIONS = {
    "dyn_gfs": "noaa-gfs-forecast",
    "dyn_aifs": "ecmwf-aifs-single-forecast",
    "dyn_hrrr": "noaa-hrrr-forecast-48-hour",
    "dyn_icon_eu": "dwd-icon-eu-forecast-5-day",
}

# Canonical variable -> the recipes that can produce it, in order of preference. Each
# recipe is the set of arrays it needs, and the first one a model can satisfy wins.
#
# Alternatives are not tidiness. The centres do not agree on names — gusts are
# `wind_gust_surface` in HRRR and `wind_gust_10m` in ICON-EU — and they do not agree on
# what to publish: GFS and AIFS carry no gusts at all, and AIFS gives dew point where the
# others give relative humidity. A single fixed name per variable silently produces a
# column of nulls for half the catalogue.
RECIPES: dict[str, tuple[tuple[str, ...], ...]] = {
    "air_temp_c": (("temperature_2m",),),
    "rh_pct": (("relative_humidity_2m",), ("temperature_2m", "dew_point_temperature_2m")),
    "wind_speed_ms": (("wind_u_10m", "wind_v_10m"),),
    "wind_gust_ms": (("wind_gust_surface",), ("wind_gust_10m",)),
    "precip_1h_mm": (("precipitation_surface",),),
}

# Precipitation is stored as an average mass flux over the step, kg m-2 s-1, which is
# millimetres per second. An hour of it is the millimetres the pipeline expects.
PRECIP_SCALE = 3600.0

# Inits are 6-hourly but a daily one is plenty: two years of daily runs already gives ~730
# samples per lead hour, against the 92 the previous-runs archive can offer, and taking all
# four multiplies the traffic and the archive size by four for a fit that is already
# well determined.
DEFAULT_INIT_STRIDE_H = 24

# Every third lead hour. The published forecast is bucketed by lead (1-6, 7-12, ... 97-168)
# and nothing downstream resolves finer, so storing every hour triples the paired archive
# to sharpen an axis that is then averaged over.
DEFAULT_LEAD_STRIDE_H = 3

# How many init times to hold in memory at once per tile. A GFS slab is
# leads x 121 x 121 x 4 bytes, about 10 MB per init per variable, so this bounds a worker
# at a few hundred megabytes while still amortising the per-request overhead.
INIT_BATCH = 16


class DynamicalError(RuntimeError):
    """Raised when the archive cannot be reached or does not carry what was asked for."""


# --------------------------------------------------------------------------- store access


@functools.lru_cache(maxsize=1)
def _catalog():
    try:
        import pystac
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DynamicalError(
            "reading dynamical.org needs the grid extras: pip install -e '.[grid]'"
        ) from exc
    return pystac.Catalog.from_file(CATALOG_URL)


@functools.lru_cache(maxsize=8)
def open_store(model: str):
    """The zarr group behind a model id.

    Cached per process because opening resolves the STAC catalogue and the repository
    manifest, which is wasted work once per tile.
    """
    try:
        import icechunk
        import zarr
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DynamicalError(
            "reading dynamical.org needs the grid extras: pip install -e '.[grid]'"
        ) from exc

    cid = COLLECTIONS.get(model)
    if cid is None:
        raise DynamicalError(f"{model!r} is not a dynamical.org model")
    collection = _catalog().get_child(cid)
    if collection is None:
        raise DynamicalError(f"collection {cid!r} is not in the catalogue")
    href = collection.assets["icechunk-https"].href
    repo = icechunk.Repository.open(icechunk.http_storage(href))
    return zarr.open_group(repo.readonly_session("main").store, mode="r")


@functools.lru_cache(maxsize=8)
def _grid(model: str):
    """Coordinate arrays, and whether the grid is regular.

    HRRR is on a Lambert conformal projection, so its latitude and longitude are 2-D
    fields rather than axes; a nearest-cell lookup there is a search over the whole grid
    instead of two independent bisections.
    """
    g = open_store(model)
    lat = np.asarray(g["latitude"][:])
    lon = np.asarray(g["longitude"][:])
    return lat, lon, lat.ndim == 1


@functools.lru_cache(maxsize=4)
def _projected_tree(model: str):
    from scipy.spatial import cKDTree

    lat, lon, _ = _grid(model)
    lat_r, lon_r = np.radians(lat.ravel()), np.radians(lon.ravel())
    cos = np.cos(lat_r)
    xyz = np.column_stack([cos * np.cos(lon_r), cos * np.sin(lon_r), np.sin(lat_r)])
    return cKDTree(xyz), lat.shape


def nearest_cell(model: str, lat: float, lon: float) -> tuple[int, int]:
    """Grid indices of the cell containing a station."""
    glat, glon, regular = _grid(model)
    if regular:
        iy = int(np.abs(glat - lat).argmin())
        # Longitudes are stored on -180..180 here; a station given as 0..360 would
        # otherwise land at the wrong edge of the world rather than fail.
        target = ((lon + 180.0) % 360.0) - 180.0
        ix = int(np.abs(glon - target).argmin())
        return iy, ix

    tree, shape = _projected_tree(model)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    point = [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    idx = int(tree.query(point)[1])
    return int(idx // shape[1]), int(idx % shape[1])


def _init_times(model: str) -> pd.DatetimeIndex:
    g = open_store(model)
    return pd.to_datetime(np.asarray(g["init_time"][:]), unit="s")


def _lead_hours(model: str) -> np.ndarray:
    g = open_store(model)
    return np.asarray(g["lead_time"][:]) / 3600.0


# --------------------------------------------------------------------------- extraction


@functools.lru_cache(maxsize=8)
def _array_names(model: str) -> frozenset[str]:
    return frozenset(n for n, _ in open_store(model).arrays())


@functools.lru_cache(maxsize=8)
def _tile_shape(model: str) -> tuple[int, int]:
    """The spatial chunk size, which is the unit a single read covers."""
    g = open_store(model)
    names = _array_names(model)
    name = "temperature_2m" if "temperature_2m" in names else sorted(names)[0]
    chunks = g[name].chunks
    return int(chunks[-2]), int(chunks[-1])


@functools.lru_cache(maxsize=64)
def _recipe_for(model: str, variable: str) -> tuple[str, ...] | None:
    """The first recipe this model can satisfy, or None if it carries the variable in no
    form at all."""
    have = _array_names(model)
    for recipe in RECIPES.get(variable, ()):
        if set(recipe) <= have:
            return recipe
    return None


def carries(model: str, variable: str) -> bool:
    """Whether the archive can produce this variable for this model.

    Worth asking before a fit rather than after: a model that returns nothing for gusts
    must be absent from the gust fit, not present with a column of nulls.
    """
    return _recipe_for(model, variable) is not None


def _tile_of(model: str, iy: int, ix: int) -> tuple[int, int]:
    clat, clon = _tile_shape(model)
    return iy // clat, ix // clon


def _relative_humidity(temp_c: np.ndarray, dew_c: np.ndarray) -> np.ndarray:
    """RH from temperature and dew point, for the models that publish only the latter.

    August-Roche-Magnus. Accurate to better than half a percent over the range surface
    stations live in, which is far inside the error the calibration is there to remove.
    """
    a, b = 17.625, 243.04
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = 100.0 * np.exp(a * dew_c / (b + dew_c) - a * temp_c / (b + temp_c))
    return np.clip(rh, 0.0, 100.0)


def _read_variable(g, model, name, init_idx, lead_sel, ys, xs):
    """One variable's slab for a batch of inits over one tile."""
    arr = g[name]
    # init chunks are one wide, so a list of indices costs exactly the chunks wanted.
    return np.asarray(arr.oindex[init_idx, lead_sel, ys, xs])


def _slab(g, model, variable, init_idx, lead_sel, ys, xs):
    """A canonical variable's values for a batch of inits over one tile.

    Returns None when the model does not carry it, rather than a column of nulls: a null
    column is indistinguishable from a missing observation once it reaches training, and
    that mistake silently degrades every fit that includes the model.
    """
    recipe = _recipe_for(model, variable)
    if recipe is None:
        return None
    read = [_read_variable(g, model, n, init_idx, lead_sel, ys, xs) for n in recipe]

    if variable == "wind_speed_ms":
        return np.hypot(read[0], read[1])
    if variable == "rh_pct" and len(recipe) == 2:
        return _relative_humidity(read[0], read[1])
    if variable == "precip_1h_mm":
        return read[0] * PRECIP_SCALE
    return read[0]


def fetch_history_batch(
    coords: list[tuple[str, float, float]],
    model: str,
    variables: list[str],
    start: date,
    end: date,
    *,
    init_stride_h: int = DEFAULT_INIT_STRIDE_H,
    max_lead_h: int = 168,
    lead_stride_h: int = DEFAULT_LEAD_STRIDE_H,
    init_batch: int = INIT_BATCH,
    only_inits: list[int] | None = None,
) -> pd.DataFrame:
    """Lead-resolved archived forecasts for many stations, grouped by chunk tile.

    The returned frame is the same long form the Open-Meteo fetchers produce — one row
    per (station, model, valid_time, lead) — so the pairing and training code downstream
    needs to know nothing about where it came from.
    """
    if not coords:
        return pd.DataFrame()

    g = open_store(model)
    inits = _init_times(model)
    leads = _lead_hours(model)

    if only_inits is not None:
        keep_init = np.asarray(only_inits, dtype=int)
    else:
        # The window is inclusive of the end *date*, not of midnight on it. Comparing
        # against the bare timestamp drops every run initialised after 00:00 that day —
        # which for a live fetch is the only run there is.
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
        keep_init = np.where(
            (inits >= pd.Timestamp(start))
            & (inits < end_ts)
            & (inits.hour % max(1, init_stride_h) == 0)
        )[0]
    if keep_init.size == 0:
        print(f"  dynamical {model}: no initialisations in {start}..{end}", flush=True)
        return pd.DataFrame()

    lead_idx = np.where((leads <= max_lead_h) & (leads % max(1, lead_stride_h) == 0))[0]
    if lead_idx.size == 0:
        return pd.DataFrame()
    lead_sel = slice(int(lead_idx[0]), int(lead_idx[-1]) + 1)
    lead_take = lead_idx - lead_idx[0]
    lead_hours = leads[lead_idx]
    # Rounded to whole seconds before becoming an offset: a float hour multiplied out
    # lands a few hundred nanoseconds off the hour, and every valid_time would then miss
    # the observation it should pair with.
    lead_offsets = np.round(lead_hours * 3600).astype("int64").astype("timedelta64[s]")

    # Group the stations by the chunk that holds them. This is the whole trick: every
    # station in a tile is served by the same read.
    tiles: dict[tuple[int, int], list[tuple[str, int, int]]] = {}
    for sid, lat, lon in coords:
        iy, ix = nearest_cell(model, lat, lon)
        tiles.setdefault(_tile_of(model, iy, ix), []).append((sid, iy, ix))

    clat, clon = _tile_shape(model)
    print(
        f"  dynamical {model}: {len(coords)} stations in {len(tiles)} tiles, "
        f"{keep_init.size} inits x {lead_idx.size} leads",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    for n_tile, ((ty, tx), members) in enumerate(sorted(tiles.items()), start=1):
        ys = slice(ty * clat, (ty + 1) * clat)
        xs = slice(tx * clon, (tx + 1) * clon)
        for i in range(0, keep_init.size, init_batch):
            batch = keep_init[i : i + init_batch].tolist()
            columns: dict[str, np.ndarray] = {}
            for variable in variables:
                try:
                    values = _slab(g, model, variable, batch, lead_sel, ys, xs)
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARN: {model} {variable} tile {ty},{tx} failed ({exc})",
                          flush=True)
                    values = None
                if values is not None:
                    columns[variable] = values[:, lead_take, :, :]
            if not columns:
                continue

            init_stamps = inits[batch]
            for sid, iy, ix in members:
                py, px = iy - ys.start, ix - xs.start
                block = pd.DataFrame(
                    {
                        "valid_time": np.repeat(init_stamps.values, len(lead_hours))
                        + np.tile(lead_offsets, len(batch)),
                        "lead_h": np.tile(lead_hours, len(batch)).astype(int),
                    }
                )
                for variable, values in columns.items():
                    block[f"fc_{variable}"] = values[:, :, py, px].reshape(-1)
                block["station_id"] = sid
                block["model"] = model
                block["lead_source"] = "dynamical"
                frames.append(block)
        if n_tile % 10 == 0 or n_tile == len(tiles):
            print(f"    tiles {n_tile}/{len(tiles)}", flush=True)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for v in variables:
        if f"fc_{v}" not in out:
            out[f"fc_{v}"] = np.nan
    return out.dropna(subset=[f"fc_{v}" for v in variables], how="all").reset_index(drop=True)


def fetch_forecast_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    *,
    max_lead_h: int = 168,
    lead_stride_h: int = 1,
) -> pd.DataFrame:
    """The current run, for many stations.

    Separate from the history path only in which initialisation it takes: the live
    forecast must come from the same archive the coefficients were fitted on, because a
    correction learned against one provider's grid-cell values does not transfer to
    another's interpolation of the same model.
    """
    frames = []
    for model in models:
        inits = _init_times(model)
        if len(inits) == 0:
            continue
        latest = inits[-1].to_pydatetime()
        # Exactly the newest run, addressed by index rather than filtered out of the
        # day's runs afterwards. Asking for the day and discarding the rest downloaded
        # three initialisations to publish one, every cycle, for every tile.
        frame = fetch_history_batch(
            coords,
            model,
            variables,
            start=latest.date(),
            end=latest.date(),
            max_lead_h=max_lead_h,
            lead_stride_h=lead_stride_h,
            only_inits=[len(inits) - 1],
        )
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def latest_init(model: str) -> datetime:
    """When the newest run in the archive was initialised."""
    return _init_times(model)[-1].to_pydatetime()
