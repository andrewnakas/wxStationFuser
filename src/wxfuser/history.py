"""Replay a station's forecasts through its history, as they would have been issued.

For each variable, the per-station calibration is walked forward through the archive
(rolling.walk_forward): refit every couple of weeks on what was known then, predict the
next block, repeat. So every calibrated forecast here is one the system could actually
have issued at that time. Beside it sit each raw model's forecast and the observation,
which is what makes the discrepancy, and the calibration's value, visible on any date.

The champion method per variable is chosen on the first 60% of the history, as the
scorecard does, and its predictions are kept for the whole period.

Output is compact columnar JSON for a page to load, sampled to what a phone can hold:

  * temperature: valid every 6 h, leads to 48 h
  * 24 h new snow and SWE: the 08:00 MST morning totals (15 UTC), leads to 72 h.
    15 UTC, not 14: GEFS and ECMWF ENS step every 3 hours, so a 14 UTC total exists
    only for HRRR, and a replay at 14 UTC compared nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wxfuser.models import select
from wxfuser.pipeline.core import FIT_FUNCS
from wxfuser.verify import metrics, rolling

SAMPLING = {
    "air_temp_c": {"hours": (0, 6, 12, 18), "max_lead": 48},
    "hn24_cm": {"hours": (15,), "max_lead": 72},
    "swe_24h_mm": {"hours": (15,), "max_lead": 72},
}
QKEYS = ["q05", "q25", "q50", "q75", "q95"]


def replay(wide: pd.DataFrame, variable: str, models: list[str], *,
           refit_every_days: int = 14) -> tuple[pd.DataFrame, str]:
    """Out-of-sample calibrated forecasts for the whole history, with raw and observed."""
    tiers = tuple(t for t in select.eligible_tiers(wide) if t in FIT_FUNCS)
    oos = rolling.walk_forward(wide, variable, models, tiers, refit_every_days=refit_every_days,
                               max_refits=400)
    if not oos:
        return pd.DataFrame(), ""
    # The method is chosen on the earlier 60% only, so choosing does not flatter it.
    times = np.sort(np.unique(np.concatenate([f["valid_time"].to_numpy() for f in oos.values()])))
    split = times[int(len(times) * 0.6)]
    score = {}
    for t, f in oos.items():
        early = f[f["valid_time"] < split]
        if len(early) > 50:
            score[t] = float(np.mean(metrics.crps_from_quantiles(
                early["obs"].to_numpy(float), {k: early[k].to_numpy() for k in QKEYS})))
    champion = min(score, key=score.get)
    out = oos[champion].merge(
        wide[["valid_time", "lead_h", *[f"fc_{m}" for m in models]]].drop_duplicates(["valid_time", "lead_h"]),
        on=["valid_time", "lead_h"], how="left")
    out["issue_time"] = pd.to_datetime(out["valid_time"]) - pd.to_timedelta(out["lead_h"], unit="h")
    return out, champion


def sample(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    rule = SAMPLING[variable]
    vt = pd.to_datetime(frame["valid_time"])
    keep = vt.dt.hour.isin(rule["hours"]) & (frame["lead_h"] <= rule["max_lead"])
    return frame[keep].sort_values(["issue_time", "lead_h"]).reset_index(drop=True)


def columnar(frame: pd.DataFrame, models: list[str], decimals: int = 1) -> dict:
    """Compact columns: issue and valid as epoch hours, values rounded, NaN as null."""
    def col(v):
        a = np.round(np.asarray(v, dtype=float), decimals)
        return [None if np.isnan(x) else float(x) for x in a]

    epoch_h = lambda s: (pd.to_datetime(s).astype("int64") // 3_600_000_000_000).astype(int).tolist()  # noqa: E731
    out = {"issue": epoch_h(frame["issue_time"]), "lead": frame["lead_h"].astype(int).tolist(),
           "obs": col(frame["obs"])}
    for k in QKEYS:
        out[k] = col(frame[k])
    for m in models:
        out[f"raw_{m}"] = col(frame[f"fc_{m}"])
    return out


def events(frame: pd.DataFrame, models: list[str], n: int = 15) -> list[dict]:
    """The largest observed 24 h snowfalls, each with its day-ahead forecasts.

    For each event day the forecast issued the morning before (the lead closest to 24 h)
    is used: a day's notice, what a briefing would have had.
    """
    day1 = frame[(frame["lead_h"] >= 11) & (frame["lead_h"] <= 30)].copy()
    if day1.empty:
        return []
    day1["pick"] = (day1["lead_h"] - 24).abs()
    day1 = day1.sort_values("pick").drop_duplicates("valid_time")
    top = day1.dropna(subset=["obs"]).nlargest(n, "obs")
    rows = []
    for r in top.itertuples():
        raws = {m: getattr(r, f"fc_{m}") for m in models}
        rows.append({
            "valid": int(pd.Timestamp(r.valid_time).value // 3_600_000_000_000),
            "issue": int(pd.Timestamp(r.issue_time).value // 3_600_000_000_000),
            "lead": int(r.lead_h), "obs": round(float(r.obs), 1),
            "q50": round(float(r.q50), 1), "q05": round(float(r.q05), 1), "q95": round(float(r.q95), 1),
            "raw": {m: (None if pd.isna(v) else round(float(v), 1)) for m, v in raws.items()},
        })
    return rows


def summary(frame: pd.DataFrame, models: list[str]) -> dict:
    """MAE of the calibrated median and of each raw model on identical rows."""
    cols = ["obs", "q50", *[f"fc_{m}" for m in models]]
    same = frame.dropna(subset=cols)
    # No shared rows means nothing to compare, published as null: NaN is not JSON, and
    # one NaN makes the whole page's data unreadable to the browser.
    mae = (lambda c: float((same[c] - same["obs"]).abs().mean()) if len(same) else None)  # noqa: E731
    out = {"rows": int(len(same)), "mae_calibrated": mae("q50")}
    for m in models:
        out[f"mae_{m}"] = mae(f"fc_{m}")
    return out


def network_snow(root, stations: list[str], models: list[str], variable: str, *,
                 test_from: str, storm_at: float = 10.0, cut: float = 0.24,
                 issue_fraction: float = 0.35, rounds: int = 300) -> dict:
    """Snow replay from the network model, with the replayed stations held out entirely.

    One station's two winters are too few storms to calibrate amounts, so snow is
    replayed from the model pooled across the network. That model is trained on every
    other station and only on data before ``test_from``, then replayed on the held-out
    stations from ``test_from`` on: unseen stations in an unseen season, as the network
    scores are. Each block adds the chance of a storm (``storm_at``) and the day-ahead
    storm calls, where calibrated means that chance reaching ``cut``.
    """
    from wxfuser import network

    cutoff = pd.Timestamp(test_from)
    pool = network.build_dataset(root, models, variable, issue_fraction=issue_fraction)
    vt = pd.to_datetime(pool["valid_time"])
    train = pool[(~pool["station_id"].isin(stations)) & (vt < cutoff)]
    del pool
    model = network.fit(train, variable, models, rounds=rounds)
    del train
    blocks = {}
    for sid in stations:
        d = network.build_dataset(root, models, variable, station_ids=[sid])
        d = d[pd.to_datetime(d["valid_time"]) >= cutoff].reset_index(drop=True)
        if d.empty:
            continue
        q = model.predict(d)
        for k, v in q.items():
            d[k] = v
        mat = np.column_stack([d[k] for k in QKEYS])
        d["pstorm"] = metrics.exceedance_probability([0.05, 0.25, 0.5, 0.75, 0.95], mat, storm_at)
        d["issue_time"] = pd.to_datetime(d["valid_time"]) - pd.to_timedelta(d["lead_h"], unit="h")
        f = sample(d, variable)
        cols = columnar(f, models)
        cols["pstorm"] = [round(float(x), 2) for x in f["pstorm"]]
        ev = events(f, models)
        chance = {(int(pd.Timestamp(r.valid_time).value // 3_600_000_000_000), int(r.lead_h)): float(r.pstorm)
                  for r in f.itertuples()}
        for e in ev:
            e["pstorm"] = round(chance.get((e["valid"], e["lead"]), float("nan")), 2)
        day1 = f[f["lead_h"].between(18, 30)]
        day1 = day1.assign(_p=(day1["lead_h"] - 24).abs()).sort_values("_p").drop_duplicates("valid_time")
        y = day1["obs"] >= storm_at
        calls = {"storm_days": int(y.sum()), "days": int(len(day1)), "cut": cut, "storm_at": storm_at}
        for name, flag in [("calibrated", day1["pstorm"] >= cut),
                           *[(m, day1[f"fc_{m}"] >= storm_at) for m in models]]:
            calls[name] = {"caught": int((flag & y).sum()), "false_alarms": int((flag & ~y).sum())}
        blocks[sid] = {"champion": "network model, this station held out", "summary": summary(f, models),
                       "rows": cols, "events": ev, "calls": calls, "since": test_from}
    return blocks
