"""The enrollment registry — which stations the scheduled jobs maintain.

``stations.yaml`` at the repo root is the single mutable piece of state kept in git.
Everything else that grows (paired archives, fitted coefficients, verification history)
lives on the Hugging Face dataset repo, so the git history stays reviewable: a diff shows
someone enrolled a station, not that a robot committed 40 MB of parquet.

A station entry carries where it is, which models to fuse, which variables to publish,
and any per-source identifiers needed to fetch its observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from wxfuser.config import REPO_ROOT, load_configs

REGISTRY_PATH = REPO_ROOT / "stations.yaml"


@dataclass
class Station:
    id: str
    name: str
    lat: float
    lon: float
    elev_m: float | None = None
    models: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    iem_network: str | None = None
    ghcnh_id: str | None = None
    nws_id: str | None = None
    country: str | None = None
    enrolled_at: str | None = None

    @property
    def slug(self) -> str:
        return self.id.replace(":", "_").replace("/", "_").replace(" ", "_")

    def resolved_models(self) -> list[str]:
        if self.models:
            return self.models
        defaults = load_configs()["models"]["defaults"]
        # HRRR/NBM are CONUS-only; offering them elsewhere would return empty columns.
        in_conus = 24.0 <= self.lat <= 50.0 and -125.0 <= self.lon <= -66.0
        return list(defaults["models_conus"] if in_conus else defaults["models"])

    def resolved_variables(self) -> list[str]:
        return self.variables or list(load_configs()["models"]["defaults"]["variables"])

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
        }
        for key in ("elev_m", "models", "variables", "iem_network", "ghcnh_id",
                    "nws_id", "country", "enrolled_at"):
            val = getattr(self, key)
            if val:
                out[key] = val
        return out

    def public_dict(self) -> dict:
        """Metadata safe to embed in published JSON."""
        return {
            "id": self.id,
            "name": self.name,
            "lat": round(float(self.lat), 4),
            "lon": round(float(self.lon), 4),
            "elev_m": float(self.elev_m) if self.elev_m is not None else None,
            "country": self.country,
        }


def load_registry(path: str | Path | None = None) -> list[Station]:
    p = Path(path or REGISTRY_PATH)
    if not p.exists():
        return []
    with open(p) as fh:
        data = yaml.safe_load(fh) or {}
    return [Station(**entry) for entry in (data.get("stations") or [])]


def save_registry(stations: list[Station], path: str | Path | None = None) -> Path:
    p = Path(path or REGISTRY_PATH)
    payload = {"stations": [s.to_dict() for s in stations]}
    with open(p, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return p


def find(station_id: str, path: str | Path | None = None) -> Station | None:
    for s in load_registry(path):
        if s.id == station_id:
            return s
    return None


def upsert(station: Station, path: str | Path | None = None) -> list[Station]:
    """Add or replace a station in the registry, keeping it sorted by id."""
    stations = [s for s in load_registry(path) if s.id != station.id]
    stations.append(station)
    stations.sort(key=lambda s: s.id)
    save_registry(stations, path)
    return stations
