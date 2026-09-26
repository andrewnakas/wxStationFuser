"""`wxfuser calibrate <link>`: from a link to a station's data to a calibrated forecast.

The working loop with a new customer is "here is where our station's data lives", so
this takes that link and does the rest:

  * A **network station** (SNOTEL, an IEM/ASOS site, a Synoptic/MesoWest station) is
    recognised from its URL, and its location and history come from the network.
  * **Anything else** is read as a CSV, including a Google Sheets share link. Columns,
    units and time zone are detected, and every guess is printed with the flag that
    overrides it. The time zone is decided by which reading puts the temperature
    peak in the afternoon, the check that would have caught SNOTEL's eight-hour error.

The result is an ordinary spec run into a local root (never the public state), plus a
standalone report to send.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from wxfuser.data.obs import USER_AGENT
from wxfuser.data.registry import Station

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
FT_TO_M = 0.3048

# Decision thresholds shown by default: roughly 40 and 60 mph gusts, a freezing line,
# a wet hour, and the new-snow and loading amounts avalanche forecasters watch.
DEFAULT_THRESHOLDS = {
    "air_temp_c": [0.0],
    "wind_gust_ms": [18.0, 27.0],
    "precip_1h_mm": [1.0],
    "hn24_cm": [15.0, 30.0],
    "swe_24h_mm": [10.0, 25.0],
}


class LinkError(ValueError):
    pass


@dataclass
class Resolved:
    kind: str                      # snotel | iem | synoptic | csv
    station: Station
    csv_url: str | None = None
    notes: list[str] = field(default_factory=list)


def _get(url: str, timeout: int = 120) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as resp:
        return resp.read()


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "station")[:40]


# --------------------------------------------------------------------------- links

SNOTEL_RE = re.compile(r"(\d{1,5})[:_ -]([A-Za-z]{2})[:_ -](SNTL|SCAN|MSNT)", re.I)


def csv_download_url(url: str) -> str:
    """Rewrite share links to their raw download form."""
    m = re.match(r"https://docs\.google\.com/spreadsheets/d/([^/]+)", url)
    if m:
        gid = parse_qs(urlparse(url).query).get("gid", [None])[0]
        frag = re.search(r"gid=(\d+)", urlparse(url).fragment or "")
        gid = gid or (frag.group(1) if frag else "0")
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"
    if "dropbox.com" in url:
        return re.sub(r"[?&]dl=0", "", url) + ("&" if "?" in url else "?") + "dl=1"
    return url


def classify(link: str) -> tuple[str, dict]:
    """What a link points at, and the identifiers pulled from it. Pure; no network."""
    m = SNOTEL_RE.search(link)
    if m:
        return "snotel", {"triplet": f"{m.group(1)}:{m.group(2).upper()}:{m.group(3).upper()}"}
    u = urlparse(link)
    q = {k.lower(): v[0] for k, v in parse_qs(u.query).items()}
    host = u.netloc.lower()
    if "wcc.sc.egov.usda.gov" in host and ("sitenum" in q or "stationid" in q):
        return "snotel", {"site": q.get("sitenum") or q.get("stationid")}
    if "mesonet.agron.iastate.edu" in host and "station" in q and "network" in q:
        return "iem", {"station": q["station"].upper(), "network": q["network"].upper()}
    if "mesowest" in host or "synopticdata" in host:
        stid = q.get("stn") or q.get("stid") or q.get("station")
        if not stid:
            parts = [p for p in u.path.split("/") if p]
            stid = next((p for p in parts if re.fullmatch(r"[A-Za-z0-9]{3,10}", p)
                         and p.lower() not in {"cgi-bin", "droman", "station", "stations"}), None)
        if stid:
            return "synoptic", {"stid": stid.upper()}
    if u.scheme in ("http", "https"):
        return "csv", {"url": csv_download_url(link)}
    if Path(link).exists():
        return "csv", {"url": str(Path(link).resolve())}
    raise LinkError(f"cannot tell what {link!r} points at")


def resolve(link: str, *, name: str | None = None, lat: float | None = None,
            lon: float | None = None, elev_m: float | None = None) -> Resolved:
    kind, ids = classify(link)
    if kind == "snotel":
        key = ids.get("triplet") or f"{ids['site']}:*:SNTL"
        meta = json.loads(_get(f"{AWDB}/stations?stationTriplets={key}&returnStationElements=false"))
        if not meta:
            raise LinkError(f"no SNOTEL station matches {key}")
        m = meta[0]
        st = Station(id=m["stationTriplet"], name=name or m.get("name") or m["stationTriplet"],
                     lat=float(m["latitude"]), lon=float(m["longitude"]),
                     elev_m=float(m["elevation"]) * FT_TO_M if m.get("elevation") else None,
                     country="US")
        return Resolved("snotel", st)
    if kind == "iem":
        gj = json.loads(_get(f"https://mesonet.agron.iastate.edu/geojson/network/{ids['network']}.geojson"))
        f = next((f for f in gj.get("features", []) if f.get("id") == ids["station"]), None)
        if f is None:
            raise LinkError(f"{ids['station']} is not in IEM network {ids['network']}")
        lo, la = f["geometry"]["coordinates"][:2]
        st = Station(id=f"IEM:{ids['station']}", name=name or f["properties"].get("sname") or ids["station"],
                     lat=float(la), lon=float(lo), elev_m=f["properties"].get("elevation"),
                     iem_network=ids["network"])
        return Resolved("iem", st)
    if kind == "synoptic":
        import os

        token = os.environ.get("SYNOPTIC_API_TOKEN")
        if not token:
            raise LinkError("a Synoptic/MesoWest station needs SYNOPTIC_API_TOKEN set")
        meta = json.loads(_get(
            f"https://api.synopticdata.com/v2/stations/metadata?stid={ids['stid']}&token={token}"))
        s = (meta.get("STATION") or [None])[0]
        if not s:
            raise LinkError(f"Synoptic has no station {ids['stid']}")
        elev = s.get("ELEVATION")
        st = Station(id=f"SYN:{ids['stid']}", name=name or s.get("NAME") or ids["stid"],
                     lat=float(s["LATITUDE"]), lon=float(s["LONGITUDE"]),
                     elev_m=float(elev) * FT_TO_M if elev not in (None, "") else None)
        return Resolved("synoptic", st)

    if lat is None or lon is None:
        raise LinkError("a CSV link needs the station's --lat and --lon")
    label = name or Path(urlparse(ids["url"]).path).stem or "station"
    st = Station(id=f"ORG:adhoc:{slugify(label)}", name=label, lat=lat, lon=lon, elev_m=elev_m)
    return Resolved("csv", st, csv_url=ids["url"])


# --------------------------------------------------------------------------- CSV detection

# Header words -> target. Order matters: gust before wind, dew point never read as
# temperature, snow depth before depth.
PATTERNS = [
    ("time", r"^(date[ _]?time|timestamp|datetime|time[ _]?stamp|obs(ervation)?[ _]?time|valid|date|time|dt)\b"),
    ("wind_gust_ms", r"gust|peak[ _]?wind|wind[ _]?max|wsmax"),
    ("wind_dir_deg", r"(wind[ _]?)?dir(ection)?\b|wd\b"),
    ("wind_speed_ms", r"wind|wspd|\bws\b|speed"),
    ("relative_humidity_pct", r"\brh\b|humid"),
    ("swe_mm", r"\bswe\b|water[ _]?equiv|wteq|snow[ _]?water"),
    ("snow_depth_cm", r"snow[ _]?depth|\bhs\b|snwd|\bdepth\b"),
    ("precip_mm", r"precip|rain|\bppt\b|\bpcp\b|accum"),
    ("air_temp_c", r"^(?!.*dew)(?!.*(snow|soil|surface|water|road)).*(temp|\btair\b|\bta\b|\bt\b)"),
]

UNIT_HINTS = {
    "air_temp_c": [(r"°?\s*f\b|deg\s*f|degf|fahrenheit|\(f\)|_f$", "F"), (r"°?\s*c\b|degc|celsius|\(c\)|_c$", "C"), (r"\bk\b|kelvin", "K")],
    "wind_speed_ms": [(r"mph", "mph"), (r"\bkt|knot", "kt"), (r"km/?h|kph", "km/h"), (r"m/?s", "m/s")],
    "wind_gust_ms": [(r"mph", "mph"), (r"\bkt|knot", "kt"), (r"km/?h|kph", "km/h"), (r"m/?s", "m/s")],
    "precip_mm": [(r"\bin\b|inch|\(in\)|_in$|\"", "in"), (r"\bmm\b|\(mm\)|_mm$", "mm"), (r"\bcm\b", "cm")],
    "snow_depth_cm": [(r"\bin\b|inch|\(in\)|_in$|\"", "in"), (r"\bcm\b|\(cm\)|_cm$", "cm"), (r"\bmm\b", "mm")],
    "swe_mm": [(r"\bin\b|inch|\(in\)|_in$|\"", "in"), (r"\bmm\b|\(mm\)|_mm$", "mm"), (r"\bcm\b", "cm")],
    "relative_humidity_pct": [(r".", "pct")],
    "wind_dir_deg": [(r".", "deg")],
}


def _guess_unit(target: str, header: str, values: pd.Series) -> tuple[str, str]:
    """(unit, how it was decided)."""
    h = header.lower()
    for pat, unit in UNIT_HINTS.get(target, []):
        if re.search(pat, h):
            return unit, f"from the header '{header}'"
    v = pd.to_numeric(values, errors="coerce").dropna()
    if target == "air_temp_c":
        unit = "F" if len(v) and v.quantile(0.95) > 45 else "C"
        return unit, "from the value range (no unit in the header)"
    if target in ("wind_speed_ms", "wind_gust_ms"):
        return "mph", "assumed (no unit in the header)"
    if target in ("precip_mm", "swe_mm", "snow_depth_cm"):
        return "in", "assumed US inches (no unit in the header)"
    return "C", "assumed"


def detect_columns(df: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """(time column, mapping strings for org_obs.parse_mapping, human-readable notes)."""
    time_col, mapping, notes, used = None, [], [], set()
    for target, pattern in PATTERNS:
        for col in df.columns:
            if col in used or not re.search(pattern, str(col).strip().lower()):
                continue
            if target == "time":
                if pd.to_datetime(df[col].head(50), errors="coerce").notna().mean() > 0.8:
                    time_col = col
                    used.add(col)
                    notes.append(f"time: '{col}'")
                    break
                continue
            if pd.to_numeric(df[col], errors="coerce").notna().mean() < 0.5:
                continue
            unit, why = _guess_unit(target, str(col), df[col])
            spec = f"{target}={col}:{unit}"
            if target == "precip_mm":
                p = pd.to_numeric(df[col], errors="coerce").dropna()
                cumulative = len(p) > 10 and (p.diff().dropna() >= -0.001).mean() > 0.95 and p.iloc[-1] > p.iloc[0]
                if cumulative:
                    spec += ":cumulative"
                    why += "; a running total, so its increments are the precipitation"
            mapping.append(spec)
            notes.append(f"{target}: '{col}' in {unit} ({why})")
            used.add(col)
            break
    if time_col is None:
        raise LinkError(f"no timestamp column found among {list(df.columns)}; pass --time-col")
    return time_col, mapping, notes


def choose_time_zone(path: str, station_id: str, mapping: dict, time_col: str,
                     lon: float) -> tuple[str, str]:
    """Pick UTC or local standard time by where each puts the temperature peak.

    Temperature peaks in the early afternoon almost everywhere, so of the two readings a
    logger is usually in, the one landing nearest 14:30 local solar time is right.
    """
    from wxfuser.data import org_obs

    std = int(round(lon / 15.0))
    candidates = {"UTC": "UTC", f"{std:+03d}:00": f"{std:+03d}:00"}
    scores = {}
    for label, tz in candidates.items():
        rec = org_obs.read_csv(path, station_id, mapping, time_col=time_col, tz=tz)
        peak = org_obs.diurnal_peak_solar_hour(rec, lon)
        if peak is not None:
            scores[label] = abs(peak - 14.5)
    if not scores:
        raise LinkError("no temperature to check the time zone against; pass --tz")
    best = min(scores, key=scores.get)
    if scores[best] > 3.5:
        raise LinkError(f"neither UTC nor local standard time puts the temperature peak in the "
                        f"afternoon ({scores}); pass --tz explicitly")
    return candidates[best], f"time zone {best}: temperature then peaks in the afternoon"


def load_csv(url: str) -> pd.DataFrame:
    raw = Path(url).read_bytes() if Path(url).exists() else _get(url, timeout=300)
    text = raw.decode("utf-8-sig", errors="replace")
    # Logger files often carry a few lines of preamble before the header row.
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines[:30]):
        if line.count(",") >= 2 and re.search(r"[A-Za-z]", line) and i + 1 < len(lines) \
                and lines[i + 1].count(",") == line.count(","):
            start = i
            break
    return pd.read_csv(io.StringIO("\n".join(lines[start:])))


# --------------------------------------------------------------------------- models

def default_models(lat: float, lon: float) -> list[str]:
    in_conus = 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0
    return ["hrrr", "gefs"] if in_conus else ["gefs", "ecmwf_ens"]


def variables_for(columns: set[str]) -> list[str]:
    out = []
    for obs_col, var in [("air_temp_c", "air_temp_c"), ("relative_humidity_pct", "rh_pct"),
                         ("wind_speed_ms", "wind_speed_ms"), ("wind_gust_ms", "wind_gust_ms"),
                         ("precip_mm", "precip_1h_mm"), ("precip_1h_mm", "precip_1h_mm"),
                         ("snow_depth_cm", "hn24_cm"), ("swe_mm", "swe_24h_mm")]:
        if obs_col in columns and var not in out:
            out.append(var)
    return out


# A 24 h snow variable needs this many distinct snow days in its history before its
# skill means anything. Below it, "better than raw" mostly measures predicting zero
# through a snow-free summer, which is true and useless.
MIN_SNOW_DAYS = 15


def withhold_unverifiable_snow(payload: dict, pairs_path) -> dict:
    """Replace the skill of snow variables with too few observed snow days by a reason."""
    from wxfuser.data import pairs as pairs_mod

    archive = pairs_mod.read_archive(pairs_path)
    # A snow day is new snow on the depth sensor. A SWE gain alone is not one: rain on
    # the pillow weighs the same, and summer at Snowbird had enough of it to pass a
    # count of SWE-gain days while measuring nothing about snow.
    depth = "obs_hn24_cm" if "obs_hn24_cm" in archive else None
    for var in ("hn24_cm", "swe_24h_mm"):
        if var not in payload.get("skill", {}):
            continue
        col = f"obs_{var}"
        days = 0
        if not archive.empty and col in archive:
            snowing = archive[col] > 0
            if depth:
                snowing &= archive[depth] > 0
            days = int(pd.to_datetime(archive.loc[snowing, "valid_time"]).dt.date.nunique())
        if days < MIN_SNOW_DAYS:
            payload["skill"][var] = {
                "status": "warming_up",
                "reason": f"only {days} snow days in the training history; skill is verified "
                          f"after {MIN_SNOW_DAYS}",
            }
    return payload
