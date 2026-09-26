"""Running fusion specs into a private root: shared by `org-run` and `calibrate`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.config import quantiles
from wxfuser.pipeline import core, emit


def use_root(root: str | Path) -> Path:
    """Point the pipeline's state, site and org-observation stores at ``root``.

    Everything a run writes lands under it, never in the public state or site
    directories.
    """
    from wxfuser.data import org_obs

    root = Path(root)
    core.STATE_DIR = root / "state"
    core.SITE_DIR = root / "site"
    org_obs.ORG_OBS_DIR = root / "observations"
    return root


def run_spec(spec, *, bootstrap: bool, years: float, evaluate: bool | None = None) -> dict:
    """Calibrate one spec and publish its schema-v2 forecast. Returns its index entry."""
    from wxfuser.spec import exceedance

    levels = quantiles()
    qkeys = [f"q{int(q * 100):02d}" for q in levels]
    st = spec.to_station()
    try:
        entry = core.run_station(st, bootstrap=bootstrap, years=years, evaluate=evaluate)
    except Exception as exc:  # noqa: BLE001
        print(f"[{spec.slug}] FAILED: {exc}", flush=True)
        entry = {"status": "error"}
    path = forecast_path(spec)
    if entry.get("status") == "ok" and path.exists():
        payload = json.loads(path.read_text())
        probs = {}
        for var, ts in spec.thresholds.items():
            block = payload["hourly"].get(var) or {}
            if all(k in block for k in qkeys):
                vals = np.array([block[k] for k in qkeys], dtype=float).T
                probs[var] = exceedance(levels, vals, ts)
        obs_file = core.obs_path(st)
        latest = None
        if obs_file.exists():
            vt = pd.read_parquet(obs_file, columns=["valid_time"])["valid_time"]
            latest = pd.to_datetime(vt).max().strftime("%Y-%m-%dT%H:%MZ") if len(vt) else None
        emit.write_json(emit.spec_forecast_json(payload, spec.public_dict(), probs, latest), path)
    return {**spec.public_dict(), "slug": st.slug, "station_id": st.id,
            "station_name": st.name, "lat": st.lat, "lon": st.lon,
            "status": entry.get("status"),
            "crpss_vs_raw": entry.get("crpss_vs_raw"),
            "beats_raw": entry.get("beats_raw")}


def forecast_path(spec) -> Path:
    return core.SITE_DIR / "stations" / spec.slug / "forecast.json"


def write_index(org: str, entries: list[dict]) -> None:
    emit.write_json({"schema_version": emit.SCHEMA_VERSION_SPEC, "org": org, "specs": entries},
                    core.SITE_DIR / "index.json")
