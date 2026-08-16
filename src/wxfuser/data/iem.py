"""IEM (Iowa Environmental Mesonet) global METAR/ASOS station enumeration.

The dynamical.org asos-parquet dataset is US-heavy (~3,300 of ~4,000 stations in the
Americas). IEM publishes the *worldwide* ASOS/METAR network as ~200 per-country network
tables (``FR__ASOS``, ``DE__ASOS``, ``JP__ASOS``, …), together ~4,000 non-US stations —
dense in Europe and Asia where the dynamical set is thin. This module enumerates those
stations so ``build_catalogue`` can add them; their hourly obs are already fetchable via
``obs.fetch_iem_hourly`` (routed by ``collect`` on the ``IEM:`` prefix + ``iem_network``).

Station ids are ``IEM:<id>``. We carry ``iem_network`` (the country table) so the obs
fetcher can query the right network. No API key required.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pandas as pd

IEM_API = "https://mesonet.agron.iastate.edu/api/1"


def _get(url: str, *, timeout: int = 60, retries: int = 3):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "wxfuser/0.1"})
            with urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            import time
            if attempt < retries - 1:
                time.sleep(2**attempt + 1)
    assert last is not None
    raise last


def list_asos_networks(include_us_states: bool = False) -> list[str]:
    """All IEM ASOS-type network codes. Country tables are ``XX__ASOS`` (double
    underscore); US state tables are ``XX_ASOS`` (single). Default to non-US only, since
    the dynamical asos-parquet already covers the US densely."""
    data = _get(f"{IEM_API}/networks.json")
    rows = data.get("data", data) if isinstance(data, dict) else data
    nets = [str(r["id"]) for r in rows if str(r.get("id", "")).endswith("ASOS")]
    if include_us_states:
        return nets
    return [n for n in nets if "__" in n]  # country networks only


def _network_stations(network: str) -> list[dict]:
    data = _get(f"{IEM_API}/network/{network}.json")
    rows = data.get("data", data) if isinstance(data, dict) else data
    out = []
    for s in rows:
        # The `lat`/`lon` shortcut fields are often null; use the full names.
        lat = s.get("latitude", s.get("lat"))
        lon = s.get("longitude", s.get("lon"))
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        elev = s.get("elevation")
        try:
            elev = float(elev) if elev is not None else None
        except (TypeError, ValueError):
            elev = None
        out.append({
            "station_id": f"IEM:{s.get('id')}",
            "iem_network": network,
            "name": s.get("name"),
            "network": "IEM",
            "state": s.get("state") or s.get("country"),
            "lat": lat,
            "lon": lon,
            "elevation_m": elev,
        })
    return out


def select_stations(bounds: dict, *, include_us_states: bool = False,
                    max_workers: int = 12) -> pd.DataFrame:
    """Enumerate worldwide IEM ASOS stations inside the region bbox.

    Fetches the ~200 country network tables concurrently, flattens to stations, filters to
    the bounding box. Elevation/relief mountain filtering is applied later (stations.py /
    terrain.py), same as the other networks."""
    from concurrent.futures import ThreadPoolExecutor

    nets = list_asos_networks(include_us_states=include_us_states)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for got in ex.map(lambda n: _safe_net(n), nets):
            rows.extend(got)
    if not rows:
        return pd.DataFrame(columns=["station_id", "iem_network", "name", "network",
                                     "state", "lat", "lon", "elevation_m"])
    df = pd.DataFrame(rows)
    b = bounds
    df = df[df["lon"].between(b["lon_min"], b["lon_max"])
            & df["lat"].between(b["lat_min"], b["lat_max"])].reset_index(drop=True)
    return df


def _safe_net(network: str) -> list[dict]:
    try:
        return _network_stations(network)
    except Exception as exc:  # noqa: BLE001 — one bad network shouldn't kill the sweep
        print(f"WARN: IEM network {network} failed ({exc})")
        return []
