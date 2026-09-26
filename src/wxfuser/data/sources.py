"""Which client serves a model, and the one place the pipeline asks for forecasts.

The pipeline used to call the Open-Meteo client directly. With dynamical.org alongside
it, every forecast request goes through here. The request is routed by the model's
``source`` in ``configs/models.yaml``, and the frames that come back have the same
shape either way:

    station_id, model, valid_time, lead_h, lead_source, fc_{var}...
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from wxfuser.config import model_catalogue
from wxfuser.data import dynamical, openmeteo


class MixedSourceError(ValueError):
    pass


def source_of(model_id: str) -> str:
    return model_catalogue().get(model_id, {}).get("source", "openmeteo")


def source_for(models: list[str]) -> str:
    """The single source a model list is served from.

    Open-Meteo rows carry approximate per-day leads while dynamical rows carry
    issue-time leads. The two never share a (valid_time, lead_h) row, so a mixed list
    would fit every model alone while claiming to fuse them. Refusing is better than
    quietly publishing that.
    """
    found = {source_of(m) for m in models}
    if len(found) > 1:
        raise MixedSourceError(
            f"models {models} span sources {sorted(found)}; a station's models must all "
            "come from one source"
        )
    return found.pop() if found else "openmeteo"


def archive_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
    start: date,
    end: date,
    *,
    include_long_history: bool,
    previous_runs_days: int = 92,
) -> pd.DataFrame:
    """Paired-to-be archived forecasts for many stations.

    ``include_long_history`` asks for the full ``start..end`` window. Without it, only
    the recent window a refresh needs is fetched: previous-runs for Open-Meteo, and the
    trailing days for dynamical, whose archive answers any window directly.
    """
    if source_for(models) == "dynamical":
        return dynamical.fetch_archive_batch(coords, models, variables, start, end)

    frames = []
    prev = openmeteo.fetch_previous_runs_batch(
        coords, models, variables, past_days=previous_runs_days
    )
    if not prev.empty:
        frames.append(prev)
    if include_long_history:
        hist = openmeteo.fetch_historical_batch(coords, models, variables, start, end)
        if not hist.empty:
            frames.append(hist)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def live_batch(
    coords: list[tuple[str, float, float]],
    models: list[str],
    variables: list[str],
) -> pd.DataFrame:
    """Current forecasts for many stations, long form with station_id."""
    if source_for(models) == "dynamical":
        return dynamical.fetch_live_batch(coords, models, variables)
    return openmeteo.fetch_forecast_batch(coords, models, variables)
