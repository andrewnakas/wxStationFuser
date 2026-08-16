"""Build the searchable catalogue of stations a user can pick from.

"Any weather station" needs a list to pick from, and that list has to be small enough for
a static page to load. Three sources give near-global coverage of hourly-reporting surface
stations:

  * **NWS** — every US station the National Weather Service serves, and the only network a
    browser can fetch observations from directly, so these are the stations that support
    instant in-browser calibration.
  * **IEM** — the global ASOS/AWOS network, mostly airports, with the network code each
    station needs for its observations to be fetchable.
  * **Meteostat** — the long tail, roughly 22k stations worldwide, aggregating national
    networks that the other two miss.

Entries are deduplicated by rounded position, preferring whichever source supports the
richest observation access, and written as a compact array-of-arrays: with tens of
thousands of stations, JSON object keys would dominate the file size.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import time
from urllib.request import Request, urlopen

import pandas as pd

from wxfuser.data.obs import USER_AGENT

NWS_STATIONS = "https://api.weather.gov/stations"
IEM_NETWORKS = "https://mesonet.agron.iastate.edu/api/1/networks.json"
# The /api/1/network/{net}/stations.json path this originally used now 404s for every
# network, which silently contributed zero IEM stations to the catalogue rather than
# failing. The geojson endpoint is the live one, and carries more: whether a station is
# still online, when its archive ends, and its GHCN-hourly id for deep history.
IEM_NETWORK_STATIONS = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
# "full" rather than "lite": lite is a ~16k subset of the ~22k station list.
METEOSTAT_STATIONS = "https://bulk.meteostat.net/v2/stations/full.json.gz"

CATALOGUE_COLUMNS = [
    "id", "name", "lat", "lon", "elev_m", "network", "country",
    "iem_network", "nws_id", "ghcnh_id",
]

# Source priority when two entries describe the same physical station. NWS wins because it
# is the only one whose observations a browser can fetch, then IEM for its long archive.
SOURCE_RANK = {"NWS": 0, "IEM": 1, "MS": 2}


def _get(url: str, *, timeout: int = 120, retries: int = 3) -> bytes | None:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  WARN: {url} failed ({exc})", flush=True)
                return None
            time.sleep(2**attempt + 1)
    return None


def fetch_nws_stations(limit: int = 500, max_pages: int = 200) -> pd.DataFrame:
    """US stations from api.weather.gov, paginated.

    The page cap needs to be generous. NWS serves tens of thousands of stations and the
    ordering is not by importance, so stopping early silently drops major airports —
    an earlier 40-page limit cut the list off before KASE, KBOS, and KSEA.
    """
    rows: list[dict] = []
    url = f"{NWS_STATIONS}?limit={limit}"
    for _ in range(max_pages):
        raw = _get(url)
        if not raw:
            break
        payload = json.loads(raw.decode("utf-8"))
        features = payload.get("features") or []
        if not features:
            break
        for f in features:
            p = f.get("properties") or {}
            geom = f.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2 or not p.get("stationIdentifier"):
                continue
            elev = (p.get("elevation") or {}).get("value")
            rows.append(
                {
                    "id": f"NWS:{p['stationIdentifier']}",
                    "name": p.get("name") or p["stationIdentifier"],
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "elev_m": float(elev) if elev is not None else None,
                    "network": "NWS",
                    "country": "US",
                    "iem_network": None,
                    "nws_id": p["stationIdentifier"],
                    "ghcnh_id": None,
                }
            )
        nxt = (payload.get("pagination") or {}).get("next")
        if not nxt:
            break
        url = nxt
        print(f"  NWS: {len(rows)} stations", flush=True)
    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)


def fetch_iem_stations(max_networks: int | None = None) -> pd.DataFrame:
    """Global ASOS/AWOS stations, carrying the network code needed to fetch their obs."""
    raw = _get(IEM_NETWORKS)
    if not raw:
        return pd.DataFrame(columns=CATALOGUE_COLUMNS)
    networks = json.loads(raw.decode("utf-8")).get("data") or []
    codes = [n.get("id") for n in networks if n.get("id", "").endswith("ASOS")]
    if max_networks:
        codes = codes[:max_networks]

    rows: list[dict] = []
    skipped_offline = 0
    for i, code in enumerate(codes):
        raw = _get(IEM_NETWORK_STATIONS.format(network=code), timeout=60, retries=2)
        if not raw:
            continue
        try:
            features = json.loads(raw.decode("utf-8")).get("features") or []
        except json.JSONDecodeError:
            continue
        for feat in features:
            p = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or []
            sid = p.get("sid") or feat.get("id")
            if not sid or len(coords) < 2:
                continue
            # A station whose archive has closed still appears in the network listing but
            # will never return a recent observation, so it cannot be calibrated.
            if p.get("online") is False or p.get("archive_end"):
                skipped_offline += 1
                continue
            attrs = p.get("attributes") or {}
            rows.append(
                {
                    "id": f"IEM:{sid}",
                    "name": p.get("sname") or sid,
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "elev_m": float(p["elevation"]) if p.get("elevation") not in (None, "") else None,
                    "network": "IEM",
                    "country": p.get("country") or (code[:2] if not code.startswith("_") else None),
                    "iem_network": code,
                    # US ASOS ids map to NWS ids by prefixing K, which gives these stations
                    # fresh observations and instant-mode support for free.
                    "nws_id": f"K{sid}" if len(sid) == 3 and p.get("country") == "US" else None,
                    # IEM knows the GHCN-hourly id, which unlocks decades of deep history.
                    "ghcnh_id": attrs.get("GHCNH_ID") or None,
                }
            )
        if (i + 1) % 40 == 0:
            print(f"  IEM: {i + 1}/{len(codes)} networks, {len(rows)} stations", flush=True)
    if skipped_offline:
        print(f"  IEM: skipped {skipped_offline} stations with a closed archive", flush=True)
    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)


def fetch_meteostat_stations() -> pd.DataFrame:
    """The global long tail from the Meteostat bulk station list."""
    raw = _get(METEOSTAT_STATIONS, timeout=180)
    if not raw:
        return pd.DataFrame(columns=CATALOGUE_COLUMNS)
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, EOFError):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
            text = fh.read().decode("utf-8")
    data = json.loads(text)

    rows = []
    for s in data:
        loc = s.get("location") or {}
        sid = s.get("id")
        if not sid or loc.get("latitude") is None:
            continue
        name = s.get("name") or {}
        rows.append(
            {
                "id": f"MS:{sid}",
                "name": name.get("en") or next(iter(name.values()), sid),
                "lat": float(loc["latitude"]),
                "lon": float(loc["longitude"]),
                "elev_m": float(loc["elevation"]) if loc.get("elevation") is not None else None,
                "network": "MS",
                "country": s.get("country"),
                "iem_network": None,
                "nws_id": None,
                # Meteostat exposes the WMO/ICAO identifiers it aggregates.
                "ghcnh_id": (s.get("identifiers") or {}).get("national") or None,
            }
        )
    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)


# NWS serves tens of thousands of "stations" that are river gauges, precipitation-only
# sites, and cooperative observers reporting daily totals. They appear in the same station
# list as airports but return nothing usable from the observations endpoint — clicking one
# is a dead end. ICAO-style identifiers (KDEN, PANC, TJSJ) are the airport/ASOS network
# that actually reports hourly temperature, wind, and humidity.
ICAO_PATTERN = re.compile(r"^[KPT][A-Z0-9]{3}$")


def is_reporting_station(station_id: str, network: str) -> bool:
    """Whether a station is likely to serve usable hourly weather observations.

    Meteostat and IEM entries are weather stations by construction. NWS needs the filter.
    """
    if network != "NWS":
        return True
    return bool(ICAO_PATTERN.match(station_id.split(":", 1)[-1]))


def deduplicate(df: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    """Collapse entries describing the same physical station.

    ~0.1 km resolution on the rounded position. Where several sources describe one
    station, keep the one whose network supports the richest observation access, and
    carry the others' identifiers across so the station can still use them.
    """
    if df.empty:
        return df
    out = df.dropna(subset=["lat", "lon"]).copy()
    out["_key"] = (
        out["lat"].round(precision).astype(str) + "," + out["lon"].round(precision).astype(str)
    )
    out["_rank"] = out["network"].map(SOURCE_RANK).fillna(9)
    out = out.sort_values(["_key", "_rank"])

    # Carry identifiers from the duplicates that are about to be dropped.
    nws = out.dropna(subset=["nws_id"]).drop_duplicates("_key").set_index("_key")["nws_id"]
    iem = out.dropna(subset=["iem_network"]).drop_duplicates("_key").set_index("_key")["iem_network"]
    ghc = out.dropna(subset=["ghcnh_id"]).drop_duplicates("_key").set_index("_key")["ghcnh_id"]

    kept = out.drop_duplicates("_key", keep="first").copy()
    kept["nws_id"] = kept["nws_id"].fillna(kept["_key"].map(nws))
    kept["iem_network"] = kept["iem_network"].fillna(kept["_key"].map(iem))
    kept["ghcnh_id"] = kept["ghcnh_id"].fillna(kept["_key"].map(ghc))
    return kept.drop(columns=["_key", "_rank"]).reset_index(drop=True)


def build(
    sources: tuple[str, ...] = ("nws", "iem", "meteostat"),
    *,
    reporting_only: bool = True,
) -> pd.DataFrame:
    frames = []
    if "nws" in sources:
        print("fetching NWS stations…", flush=True)
        frames.append(fetch_nws_stations())
    if "iem" in sources:
        print("fetching IEM stations…", flush=True)
        frames.append(fetch_iem_stations())
    if "meteostat" in sources:
        print("fetching Meteostat stations…", flush=True)
        frames.append(fetch_meteostat_stations())

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=CATALOGUE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    print(f"combined {len(combined)} entries", flush=True)

    if reporting_only:
        before = len(combined)
        keep = [
            is_reporting_station(r.id, r.network)
            for r in combined.itertuples(index=False)
        ]
        combined = combined[pd.Series(keep, index=combined.index)]
        print(f"kept {len(combined)} of {before} that report hourly weather", flush=True)

    out = deduplicate(combined)
    print(f"deduplicated to {len(out)} stations", flush=True)
    return out


def to_min_json(df: pd.DataFrame) -> dict:
    """Compact form for the browser's search box.

    Array-of-arrays with a declared column order: at tens of thousands of rows, repeating
    JSON keys would roughly triple the download for no benefit.
    """
    cols = ["id", "name", "lat", "lon", "elev_m", "network", "country", "iem_network",
            "nws_id", "ghcnh_id"]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                r["id"],
                r["name"],
                round(float(r["lat"]), 4),
                round(float(r["lon"]), 4),
                None if pd.isna(r["elev_m"]) else round(float(r["elev_m"]), 1),
                r["network"],
                None if pd.isna(r["country"]) else r["country"],
                None if pd.isna(r["iem_network"]) else r["iem_network"],
                None if pd.isna(r["nws_id"]) else r["nws_id"],
                None if pd.isna(r.get("ghcnh_id")) else r.get("ghcnh_id"),
            ]
        )
    return {"columns": cols, "stations": rows}
