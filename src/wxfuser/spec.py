"""FusionSpec: one customer's choice of station, models and variables.

The public pipeline has one calibration per station, defined by the station itself. A
customer (an avalanche centre, a ski patrol) picks the models to fuse and the variables
to forecast, and two customers, or one trying two model sets, can pick differently for
the same station. So everything a calibration owns (its paired archive, fitted state
and published forecast) is keyed by the spec rather than the station.

The key is a hash of the choice, so a spec that changes its model set becomes a new
calibration with its own history. That is intended: a champion fitted on HRRR+GEFS says
nothing about how HRRR+IFS-ENS should be weighted.

Org spec files are private. They live in the org's state root (``orgs/{org}/specs.yaml``
in R2), never in this repository, because they name customer stations and locations.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from wxfuser.config import known_variables, model_catalogue
from wxfuser.data.registry import Station
from wxfuser.data.sources import source_for
from wxfuser.verify.metrics import exceedance_probability

ORG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


class SpecError(ValueError):
    pass


class SpecStation(Station):
    """A Station whose storage slug is its spec's, so every path the pipeline derives
    from ``slug`` (pairs, obs, models, site) lands under the spec rather than the station.
    """

    def __init__(self, *args, spec_slug: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_slug = spec_slug

    @property
    def slug(self) -> str:  # type: ignore[override]
        return self._spec_slug


@dataclass
class FusionSpec:
    org: str
    station: Station
    models: list[str]
    variables: list[str]
    # Variable -> thresholds to publish exceedance probabilities for, e.g.
    # {"wind_gust_ms": [20, 25]} gives P(gust > 20 m/s) and P(gust > 25 m/s) each hour.
    thresholds: dict[str, list[float]] = field(default_factory=dict)
    label: str | None = None

    def key(self) -> str:
        """Stable hash of what the calibration depends on, and nothing else.

        Thresholds and labels are presentation, computed from the same quantiles, so
        changing them must not throw away a trained archive.
        """
        basis = {
            "org": self.org,
            "station": self.station.id,
            "models": sorted(self.models),
            "variables": sorted(self.variables),
        }
        return hashlib.sha1(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:10]

    @property
    def slug(self) -> str:
        return f"{self.station.slug}--{self.key()}"

    def to_station(self) -> SpecStation:
        """The Station the pipeline runs: this spec's models, variables and storage slug."""
        st = self.station
        return SpecStation(
            id=st.id, name=st.name, lat=st.lat, lon=st.lon, elev_m=st.elev_m,
            models=list(self.models), variables=list(self.variables),
            iem_network=st.iem_network, ghcnh_id=st.ghcnh_id, nws_id=st.nws_id,
            country=st.country, enrolled_at=st.enrolled_at, spec_slug=self.slug,
        )

    def validate(self) -> None:
        if not ORG_RE.match(self.org):
            raise SpecError(f"org id {self.org!r} must be lowercase letters, digits and dashes")
        catalogue = model_catalogue()
        unknown = [m for m in self.models if m not in catalogue]
        if unknown:
            raise SpecError(f"unknown models {unknown}")
        if not self.models:
            raise SpecError("a spec needs at least one model")
        source_for(self.models)  # raises on a mixed list
        bad_vars = [v for v in self.variables if v not in known_variables()]
        if bad_vars:
            raise SpecError(f"unknown variables {bad_vars}")
        for var in self.thresholds:
            if var not in self.variables:
                raise SpecError(f"threshold on {var}, which the spec does not forecast")

    def public_dict(self) -> dict:
        return {
            "key": self.key(),
            "org": self.org,
            "label": self.label,
            "models": self.models,
            "variables": self.variables,
            "thresholds": self.thresholds,
        }


def load_org(path: str | Path) -> tuple[str, list[FusionSpec]]:
    """Read an org spec file.

    Format::

        org: wasatch-patrol
        stations:                 # the org's own stations; public ones need no entry
          - {id: "ORG:wasatch-patrol:collins", name: Collins, lat: 40.58, lon: -111.64,
             elev_m: 2900}
        specs:
          - station: "766:UT:SNTL"
            models: [hrrr, gefs]
            variables: [air_temp_c, wind_gust_ms, precip_1h_mm]
            thresholds: {wind_gust_ms: [20, 25]}

    A spec naming a public station takes its location from the public registry, so an
    org cannot quietly move a shared station.
    """
    from wxfuser.data.registry import load_registry

    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    org = str(doc.get("org") or "")
    own = {s["id"]: Station(**s) for s in doc.get("stations") or []}
    for sid in own:
        if not sid.startswith(f"ORG:{org}:"):
            raise SpecError(f"org station {sid!r} must be named ORG:{org}:<id>")
    public = {s.id: s for s in load_registry()}

    specs = []
    for entry in doc.get("specs") or []:
        sid = entry["station"]
        station = own.get(sid) or public.get(sid)
        if station is None:
            raise SpecError(f"spec names station {sid!r}, which is neither the org's nor enrolled")
        spec = FusionSpec(
            org=org,
            station=station,
            models=list(entry["models"]),
            variables=list(entry.get("variables") or station.resolved_variables()),
            thresholds={k: [float(x) for x in v] for k, v in (entry.get("thresholds") or {}).items()},
            label=entry.get("label"),
        )
        spec.validate()
        specs.append(spec)
    return org, specs


def exceedance(levels: list[float], quantile_values: np.ndarray, thresholds: list[float]) -> dict:
    """P(X > t) for each hour and threshold. The estimator is the one verification scores
    (``metrics.exceedance_probability``), so the published number is the verified one."""
    return {f"{t:g}": exceedance_probability(levels, quantile_values, t) for t in thresholds}
