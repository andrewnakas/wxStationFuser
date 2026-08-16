"""Parse and validate a station-enrollment issue, then add it to the registry.

Run by ``.github/workflows/enroll.yml`` when someone submits the enrollment issue form.
GitHub renders issue forms into markdown, so the input here is the issue body text and
the job is to turn it back into structured fields.

Validation is strict and the errors are written for the person who filled in the form,
not for a log: a bad latitude or a missing IEM network code should come back as a comment
explaining what to fix, rather than as a stack trace in a workflow nobody reads.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date

from wxfuser.config import model_catalogue, variable_map
from wxfuser.data.registry import Station, find, upsert

NO_RESPONSE = {"_no response_", "_none_", "none", "n/a", ""}

KNOWN_PREFIXES = ("IEM:", "NWS:", "SYN:", "MS:", "GHCNH:")


def parse_issue_body(body: str) -> dict[str, object]:
    """Extract fields from a rendered GitHub issue form.

    The rendered shape is a level-3 heading per field followed by its value, and
    checkbox groups render as ``- [x] label`` lines.
    """
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for raw in body.replace("\r\n", "\n").split("\n"):
        heading = re.match(r"^###\s+(.*)$", raw.strip())
        if heading:
            if current:
                sections[current] = "\n".join(lines).strip()
            current = heading.group(1).strip().lower()
            lines = []
        elif current:
            lines.append(raw)
    if current:
        sections[current] = "\n".join(lines).strip()

    def scalar(key: str) -> str | None:
        val = (sections.get(key) or "").strip()
        return None if val.lower() in NO_RESPONSE else val

    def checked(key: str) -> list[str]:
        block = sections.get(key) or ""
        out = []
        for line in block.split("\n"):
            m = re.match(r"^\s*-\s*\[[xX]\]\s*(\S+)", line)
            if m:
                out.append(m.group(1).strip())
        return out

    return {
        "station_id": scalar("station id"),
        "name": scalar("display name"),
        "lat": scalar("latitude"),
        "lon": scalar("longitude"),
        "elev": scalar("elevation (metres)"),
        "iem_network": scalar("iem network code"),
        "nws_id": scalar("nws station id"),
        "ghcnh_id": scalar("ghcn-hourly station id"),
        "models": checked("weather models to fuse"),
        "variables": checked("variables to publish"),
    }


def validate(fields: dict) -> tuple[Station | None, list[str]]:
    errors: list[str] = []

    station_id = (fields.get("station_id") or "").strip()
    if not station_id:
        errors.append("Station ID is required.")
    elif not (
        station_id.startswith(KNOWN_PREFIXES) or re.match(r"^\d+:[A-Z]{2}:\w+$", station_id)
    ):
        errors.append(
            f"`{station_id}` does not look like a supported station ID. Use one of the "
            f"prefixes {', '.join(KNOWN_PREFIXES)} or a SNOTEL triplet like `663:CO:SNTL`."
        )

    name = (fields.get("name") or "").strip()
    if not name:
        errors.append("Display name is required.")

    def number(key: str, lo: float, hi: float, required: bool = True):
        raw = fields.get(key)
        if raw in (None, ""):
            if required:
                errors.append(f"{key.capitalize()} is required.")
            return None
        try:
            val = float(str(raw).strip())
        except ValueError:
            errors.append(f"{key.capitalize()} `{raw}` is not a number.")
            return None
        if not (lo <= val <= hi):
            errors.append(f"{key.capitalize()} {val} is outside the valid range {lo}..{hi}.")
            return None
        return val

    lat = number("lat", -90.0, 90.0)
    lon = number("lon", -180.0, 180.0)
    elev = number("elev", -500.0, 9000.0, required=False)

    if station_id.startswith("IEM:") and not fields.get("iem_network"):
        errors.append(
            "IEM stations need a network code (for example `CO_ASOS`) so their "
            "observations can be fetched."
        )

    catalogue = model_catalogue()
    models = [m for m in (fields.get("models") or []) if m]
    unknown = [m for m in models if m not in catalogue]
    if unknown:
        errors.append(f"Unknown model(s): {', '.join(unknown)}.")

    known_vars = set(variable_map())
    variables = [v for v in (fields.get("variables") or []) if v]
    bad_vars = [v for v in variables if v not in known_vars]
    if bad_vars:
        errors.append(f"Unknown variable(s): {', '.join(bad_vars)}.")

    # A regional model outside its domain would silently produce empty columns.
    if lat is not None and lon is not None:
        in_conus = 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0
        for m in models:
            if not catalogue.get(m, {}).get("global", True) and not in_conus:
                errors.append(
                    f"`{m}` only covers the contiguous United States, but this station is "
                    f"at {lat}, {lon}. Remove it or pick a global model."
                )

    if errors:
        return None, errors

    if find(station_id):
        errors.append(f"`{station_id}` is already enrolled.")
        return None, errors

    station = Station(
        id=station_id,
        name=name,
        lat=lat,
        lon=lon,
        elev_m=elev,
        models=models,
        variables=variables,
        iem_network=(fields.get("iem_network") or None),
        nws_id=(fields.get("nws_id") or None),
        ghcnh_id=(fields.get("ghcnh_id") or None),
        enrolled_at=date.today().isoformat(),
    )
    return station, []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--body-file", required=True, help="file containing the issue body")
    ap.add_argument("--output", help="write the resolved station id here")
    ap.add_argument("--errors-output", help="write validation errors here (markdown)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.body_file) as fh:
        body = fh.read()

    fields = parse_issue_body(body)
    station, errors = validate(fields)

    if errors:
        msg = "I could not enroll this station:\n\n" + "\n".join(f"- {e}" for e in errors)
        print(msg)
        if args.errors_output:
            with open(args.errors_output, "w") as fh:
                fh.write(msg)
        return 1

    print(f"validated {station.id} ({station.name}) at {station.lat}, {station.lon}")
    print(f"models: {station.resolved_models()}")
    print(f"variables: {station.resolved_variables()}")

    if not args.dry_run:
        upsert(station)
        print(f"added {station.id} to stations.yaml")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"id": station.id, "slug": station.slug, "name": station.name}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
