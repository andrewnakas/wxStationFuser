"""How many people a station's forecast would serve.

The fleet is limited by forecast requests, not by the number of stations that exist — so
the order stations are enrolled, trained, and refreshed in decides which ones are actually
useful. Alphabetical order is the worst possible answer: it fills the fleet with whichever
airport happens to start with an A.

This ranks a station by the population living near it, decayed with distance. Cities come
from the GeoNames ``cities15000`` gazetteer — every settlement above 15,000 people, with
coordinates and a population count, in one 3 MB download and no API key. A station's score
is

    served_pop = sum over cities within RADIUS_KM of  population * exp(-distance / DECAY_KM)

which is a deliberate simplification of "who would read this forecast". The decay exists
because a city with its own station nearby does not need a distant one; the cutoff exists
because a station 200 km away is a different forecast, not a worse one.

It is a proxy and behaves like one. It ranks a major hub airport above a rural strip, and
Heathrow above both, which is the point. It says nothing about a station's *quality* — a
SNOTEL site in an empty mountain range scores near zero and is still one of the most
valuable stations in the fleet, because that is exactly where the model terrain is wrong.
So elevation stays the tie-break rather than being replaced: population decides which
populated station comes first, and among the unpopulated ones the mountains still win.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from wxfuser.data.obs import USER_AGENT

# The 15,000-and-above list: 34k cities, 3 MB zipped. The 1,000-and-above one is 12x
# larger and moves nothing near the top of the ranking, where the decisions are made.
GEONAMES_URL = "https://download.geonames.org/export/dump/cities15000.zip"

# Cached outside the hub-synced folders on purpose: it is a rebuildable download, not
# state, and every runner would otherwise push a copy of it back to the dataset repo.
CACHE_DIR = Path(os.environ.get("WXFUSER_STATE_DIR", "state")) / "cache"
CACHE_NAME = "cities15000.txt"

RADIUS_KM = 150.0
DECAY_KM = 30.0
EARTH_R_KM = 6371.0

# GeoNames' dump is a headerless TSV; these are the only columns this needs.
_COL_NAME = 1
_COL_LAT = 4
_COL_LON = 5
_COL_COUNTRY = 8
_COL_POP = 14


def cache_path() -> Path:
    return CACHE_DIR / CACHE_NAME


def download_cities(dest: Path | None = None, *, timeout: int = 120) -> Path:
    """Fetch and unpack the gazetteer, returning the path to the TSV."""
    dest = Path(dest or cache_path())
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(GEONAMES_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as fh:  # noqa: S310 - fixed https URL
        blob = fh.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        dest.write_bytes(zf.read(name))
    return dest


def load_cities(path: Path | None = None, *, download: bool = True) -> pd.DataFrame:
    """The gazetteer as ``name, lat, lon, population, country``.

    Cached on disk between runs. A refresh job downloads it once and every shard in the
    same job reads the copy.
    """
    path = Path(path or cache_path())
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        print(f"downloading city gazetteer -> {path}", flush=True)
        download_cities(path)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[_COL_NAME, _COL_LAT, _COL_LON, _COL_COUNTRY, _COL_POP],
        names=["name", "lat", "lon", "country", "population"],
        dtype={"name": str, "country": str},
        # GeoNames carries every alternate spelling of every place name; quoting is not
        # part of the format, so a lone double quote in a name would otherwise swallow
        # the rest of the file.
        quoting=3,
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["lat", "lon", "population"])
    return df[df["population"] > 0].reset_index(drop=True)


def _xyz(lat, lon) -> np.ndarray:
    """Cartesian coordinates on a sphere of Earth's radius, in kilometres.

    Distances are then straight-line rather than great-circle. Over 150 km the two differ
    by about 0.02%, which is far below the precision this ranking claims, and it lets a
    KD-tree answer "every city within 150 km" for 30,000 cities and 10,000 stations in
    under a second.
    """
    lat_r = np.radians(np.asarray(lat, dtype=float))
    lon_r = np.radians(np.asarray(lon, dtype=float))
    cos_lat = np.cos(lat_r)
    return np.column_stack(
        [
            EARTH_R_KM * cos_lat * np.cos(lon_r),
            EARTH_R_KM * cos_lat * np.sin(lon_r),
            EARTH_R_KM * np.sin(lat_r),
        ]
    )


def served_population(
    lat,
    lon,
    cities: pd.DataFrame | None = None,
    *,
    radius_km: float = RADIUS_KM,
    decay_km: float = DECAY_KM,
) -> np.ndarray:
    """Distance-decayed population near each of the given points."""
    from scipy.spatial import cKDTree

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if cities is None:
        cities = load_cities()
    if cities.empty or lat.size == 0:
        return np.zeros(lat.shape, dtype=float)

    tree = cKDTree(_xyz(cities["lat"], cities["lon"]))
    pop = cities["population"].to_numpy(dtype=float)
    points = _xyz(lat, lon)

    out = np.zeros(len(points), dtype=float)
    for i, neighbours in enumerate(tree.query_ball_point(points, r=radius_km)):
        if not neighbours:
            continue
        idx = np.asarray(neighbours, dtype=int)
        d = np.linalg.norm(points[i] - tree.data[idx], axis=1)
        out[i] = float(np.sum(pop[idx] * np.exp(-d / decay_km)))
    return out


def rank(df: pd.DataFrame, cities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add a ``served_pop`` column and sort by it, elevation breaking ties.

    Returns the frame unchanged but for the new column and the order, so a caller can
    still apply its own filters afterwards.
    """
    if df.empty:
        return df
    out = df.copy()
    out["served_pop"] = served_population(out["lat"].to_numpy(), out["lon"].to_numpy(), cities)
    elev = out["elev_m"] if "elev_m" in out.columns else pd.Series(0.0, index=out.index)
    out["_elev_key"] = pd.to_numeric(elev, errors="coerce").fillna(-1.0)
    out = out.sort_values(["served_pop", "_elev_key"], ascending=False, kind="mergesort")
    return out.drop(columns="_elev_key").reset_index(drop=True)


def sort_stations(stations: list, cities: pd.DataFrame | None = None) -> list:
    """Registry entries, most-served first.

    Used to decide what a capped or interrupted run spends its budget on. A shard that
    dies at its timeout has then done the stations most people would have looked at,
    rather than the first few hundred slugs in the alphabet.
    """
    if len(stations) < 2:
        return list(stations)
    scores = served_population(
        [s.lat for s in stations], [s.lon for s in stations], cities
    )
    order = sorted(
        range(len(stations)),
        key=lambda i: (-scores[i], -(stations[i].elev_m or -1.0), stations[i].id),
    )
    return [stations[i] for i in order]
