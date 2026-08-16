"""Per-station fitted state: champion choice, coefficients, and the last verification.

Split out because the two scheduled jobs have very different budgets. Deciding *which*
method to use requires walk-forward verification — dozens of refits per variable — and
that is a weekly job. Producing tomorrow's forecast with an already-chosen method is
seconds of work, and that is the job that runs every few hours.

So the champion and its scorecard are persisted here and reused between retrains. The
forecast still gets a fresh fit on data up to the current hour every refresh; what is
cached is the *decision*, not the coefficients, because a decision made from a year of
walk-forward evidence does not change in three hours.
"""
from __future__ import annotations

import json
from pathlib import Path

from wxfuser.pipeline.emit import _clean

STATE_VERSION = 1


def state_path(state_dir: Path, slug: str) -> Path:
    return Path(state_dir) / "models" / f"{slug}.json"


def load(state_dir: Path, slug: str) -> dict:
    p = state_path(state_dir, slug)
    if not p.exists():
        return {"version": STATE_VERSION, "variables": {}}
    try:
        with open(p) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt cache must not wedge the pipeline; a lost decision is recoverable
        # on the next retrain, a crashed refresh is not.
        return {"version": STATE_VERSION, "variables": {}}


def save(state_dir: Path, slug: str, payload: dict) -> Path:
    p = state_path(state_dir, slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(_clean(payload), fh, separators=(",", ":"), allow_nan=False)
    return p


def champion_for(state: dict, variable: str) -> str | None:
    return (state.get("variables", {}).get(variable) or {}).get("champion")


def scorecard_for(state: dict, variable: str) -> dict | None:
    return (state.get("variables", {}).get(variable) or {}).get("scorecard")


def record(
    state: dict,
    variable: str,
    *,
    champion: str,
    scorecard: dict | None = None,
    selection: dict | None = None,
    evaluated_at: str | None = None,
) -> dict:
    """Merge a variable's outcome into the station state, keeping prior verification.

    A refresh that only refits does not overwrite the stored scorecard: the numbers on
    the site should keep describing the evaluation that produced them.
    """
    variables = state.setdefault("variables", {})
    entry = variables.setdefault(variable, {})
    entry["champion"] = champion
    if selection is not None:
        entry["selection"] = selection
    if scorecard is not None:
        entry["scorecard"] = scorecard
        entry["evaluated_at"] = evaluated_at
    return state
