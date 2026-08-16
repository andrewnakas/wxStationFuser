"""Enrollment parsing and validation.

The issue form is the only way most people will ever add a station, so a parsing bug
here is a broken front door. GitHub renders forms into markdown, and these tests work
from that rendered shape rather than from the form definition.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_enrollment import parse_issue_body, validate  # noqa: E402

RENDERED = """### Station ID

IEM:DEN

### Display name

Denver International Airport

### Latitude

39.8467

### Longitude

-104.6564

### Elevation (metres)

1656

### IEM network code

CO_ASOS

### NWS station ID

KDEN

### GHCN-hourly station ID

_No response_

### Weather models to fuse

- [x] gfs_seamless (NOAA GFS, global)
- [ ] ecmwf_ifs025 (ECMWF IFS, global)
- [x] icon_seamless (DWD ICON)

### Variables to publish

- [x] air_temp_c (temperature)
- [ ] precip_1h_mm (precipitation)
"""


def test_parses_a_rendered_issue_form():
    f = parse_issue_body(RENDERED)
    assert f["station_id"] == "IEM:DEN"
    assert f["name"] == "Denver International Airport"
    assert f["lat"] == "39.8467"
    assert f["iem_network"] == "CO_ASOS"
    assert f["nws_id"] == "KDEN"
    # "_No response_" is GitHub's placeholder for a skipped optional field.
    assert f["ghcnh_id"] is None
    assert f["models"] == ["gfs_seamless", "icon_seamless"]
    assert f["variables"] == ["air_temp_c"]


def test_rejects_an_out_of_range_latitude():
    fields = parse_issue_body(RENDERED.replace("39.8467", "391.5"))
    station, errors = validate(fields)
    assert station is None
    assert any("lat" in e.lower() for e in errors)


def test_rejects_an_unknown_station_prefix():
    fields = parse_issue_body(RENDERED.replace("IEM:DEN", "WAT:DEN"))
    station, errors = validate(fields)
    assert station is None
    assert any("supported station ID" in e for e in errors)


def test_requires_a_network_code_for_iem_stations():
    fields = parse_issue_body(RENDERED.replace("CO_ASOS", "_No response_"))
    station, errors = validate(fields)
    assert station is None
    assert any("network code" in e for e in errors)


def test_rejects_an_unknown_model():
    fields = parse_issue_body(RENDERED.replace("gfs_seamless", "gfs_imaginary"))
    station, errors = validate(fields)
    assert station is None
    assert any("Unknown model" in e for e in errors)


def test_rejects_a_conus_only_model_outside_conus():
    """HRRR outside North America returns empty columns rather than an error, so the
    mistake has to be caught at enrollment or it becomes a silently untrained station."""
    body = (
        RENDERED.replace("IEM:DEN", "MS:12345")
        .replace("39.8467", "51.5")
        .replace("-104.6564", "-0.12")
        .replace("- [x] gfs_seamless (NOAA GFS, global)",
                 "- [x] ncep_hrrr_conus (NOAA HRRR, US only, short range)")
    )
    fields = parse_issue_body(body)
    station, errors = validate(fields)
    assert station is None
    assert any("contiguous United States" in e for e in errors)


def test_accepts_a_valid_submission(tmp_path, monkeypatch):
    import wxfuser.data.registry as registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "stations.yaml")
    fields = parse_issue_body(RENDERED)
    station, errors = validate(fields)
    assert errors == []
    assert station.id == "IEM:DEN"
    assert station.resolved_models() == ["gfs_seamless", "icon_seamless"]
    assert station.slug == "IEM_DEN"
