"""Config loading — every tunable constant lives in configs/*.yaml, not in code."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# repo root = three levels up from src/wxfuser/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=1)
def load_configs() -> dict[str, Any]:
    """Merged contents of every configs/*.yaml, keyed by file stem."""
    out: dict[str, Any] = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(path) as fh:
            out[path.stem] = yaml.safe_load(fh) or {}
    return out


def lead_buckets() -> list[tuple[int, int]]:
    return [tuple(b) for b in load_configs()["tiers"]["lead_buckets"]]


def bucket_for_lead(lead_h: int) -> str:
    """Label of the bucket a lead time falls in, e.g. ``13-24``.

    Leads beyond the last bucket clamp into it rather than being dropped: a 200 h
    forecast is still better served by the 97-168 h coefficients than by nothing.
    """
    buckets = lead_buckets()
    for lo, hi in buckets:
        if lo <= lead_h <= hi:
            return f"{lo}-{hi}"
    lo, hi = buckets[-1] if lead_h > buckets[-1][1] else buckets[0]
    return f"{lo}-{hi}"


def bucket_labels() -> list[str]:
    return [f"{lo}-{hi}" for lo, hi in lead_buckets()]


def quantiles() -> list[float]:
    return list(load_configs()["tiers"]["quantiles"])


def model_catalogue() -> dict[str, dict]:
    """Open-Meteo model id -> its metadata block from configs/models.yaml."""
    return {m["id"]: m for m in load_configs()["models"]["models"]}


def variable_map() -> dict[str, str]:
    """Canonical variable name -> Open-Meteo hourly variable name."""
    return dict(load_configs()["models"]["variables"])


def model_supports(model_id: str, variable: str) -> bool:
    """Whether a model actually carries a variable.

    Open-Meteo returns an all-null column rather than an error when it does not
    (ECMWF IFS/AIFS have no wind gusts), so this gate keeps null columns from being
    mistaken for missing observations during training.
    """
    meta = model_catalogue().get(model_id)
    return bool(meta) and variable in meta.get("vars", [])


def hf_state_repo() -> str:
    return os.environ.get("WXFUSER_HF_REPO") or load_configs()["hub"]["hub"]["state_repo"]


def hub_path(which: str) -> str:
    return load_configs()["hub"]["hub"]["paths"][which]
