"""One calibration model trained across a whole station network.

Per-station calibration needs each station's own history, and a new customer station
has none. A model trained across many stations learns how the forecasts err in
mountain terrain in general: at elevation, in cold pools, under the storm types the
region gets. It can calibrate a station it has never seen, and per-station tiers
refine it as that station's history grows.

Inputs are the two archives (archive.py and points.py):

  * ``{archive}/snotel/stations.parquet`` and ``snotel/obs/*.parquet``
  * ``{archive}/points/{model}/{YYYY-MM}.parquet``

Runs are issue-aligned exactly as live forecasts are (dynamical.align_to_issues), and
paired with observations by the same code the per-station pipeline uses.

The honest test for "a station it has never seen" is **leave-stations-out**: stations
are split into folds, and each fold is scored by a model that never saw them.
Baselines are scored on identical rows (fused and raw alike), per the lesson that
averaging each model over its own coverage misstates the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.config import model_catalogue, quantiles
from wxfuser.data import derived, dynamical
from wxfuser.data import pairs as pairs_mod
from wxfuser.verify import metrics

STATION_FEATURES = ["elev_m", "lat", "lon"]
CALENDAR = ["sin_doy", "cos_doy", "sin_hod", "cos_hod"]


# --------------------------------------------------------------------------- loading


def load_stations(root: Path) -> pd.DataFrame:
    meta = pd.read_parquet(root / "snotel" / "stations.parquet")
    return meta.rename(columns={"id": "station_id"})[["station_id", "name", "lat", "lon", "elev_m", "state"]]


def load_obs(root: Path, station_ids: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sid in station_ids:
        p = root / "snotel" / "obs" / f"{sid.replace(':', '_')}.parquet"
        if p.exists():
            out[sid] = pd.read_parquet(p)
    return out


def load_runs(root: Path, model: str, station_ids: set[str] | None = None,
              columns: list[str] | None = None) -> pd.DataFrame:
    """A model's stored runs, filtered to stations and columns as they are read.

    Filtering in the reader rather than after it is the difference between holding
    ten stations' runs and the whole network's: HRRR for Montana alone is 27 M rows.
    """
    import pyarrow.dataset as pads

    files = sorted((root / "points" / model).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    ds = pads.dataset([str(f) for f in files], format="parquet")
    filt = pads.field("station_id").isin(sorted(station_ids)) if station_ids is not None else None
    return ds.to_table(filter=filt, columns=columns).to_pandas()


def align_runs(runs: pd.DataFrame, model: str, issue_hours: list[int], variables: list[str],
               max_lead_h: int = 168) -> pd.DataFrame:
    """Stored raw runs -> issue-aligned rows, by the live system's rule.

    Each issue uses the newest run that had arrived (init + latency <= issue) and has
    not gone stale, and lead is measured from the issue.
    """
    if runs.empty:
        return pd.DataFrame()
    meta = model_catalogue()[model]
    latency = float(meta.get("latency_h", 3))
    run_lead = float(runs["lead_h"].max())
    inits = pd.DatetimeIndex(sorted(runs["init_time"].unique()))
    first, last = inits[0].normalize(), inits[-1].normalize() + pd.Timedelta(days=2)
    issues = [d + pd.Timedelta(hours=h) for d in pd.date_range(first, last, freq="D")
              for h in sorted(issue_hours)]
    chosen = {}
    for issue in issues:
        pos = dynamical.latest_available(inits, issue, latency)
        if pos is None or (issue - inits[pos]) / pd.Timedelta(hours=1) >= run_lead:
            continue
        chosen[issue] = inits[pos]
    cap = min(max_lead_h, int(meta.get("max_lead_h") or max_lead_h))
    return dynamical.align_to_issues(runs, model, list(chosen), chosen, variables, max_lead_h=cap)


# --------------------------------------------------------------------------- dataset


def build_dataset(root: str | Path, models: list[str], variable: str, *,
                  issue_hours: tuple[int, ...] = (3, 15), states: list[str] | None = None,
                  station_batch: int = 10, issue_fraction: float = 1.0,
                  seed: int = 0) -> pd.DataFrame:
    """The pooled training table for one variable: one row per (station, issue, lead).

    Built a few stations at a time, keeping float32, so peak memory is one batch's runs
    plus the finished table rather than every run in the network. ``issue_fraction``
    keeps a random share of issue times (the same ones at every station), a way to
    trade rows for memory without biasing toward any station or season.

    Columns: station_id, valid_time, lead_h, obs, fc_{model}..., fc_mean, fc_spread,
    nowcast_err, the station descriptors and calendar harmonics.
    """
    root = Path(root)
    stations = load_stations(root)
    if states:
        stations = stations[stations["state"].isin(states)]
    ids = sorted(stations["station_id"])
    inputs = derived.fetch_variables([variable])
    cols = ["station_id", "init_time", "lead_h", "valid_time", *[f"fc_{v}" for v in inputs]]
    rng = np.random.default_rng(seed)
    keep_issue: dict = {}

    tables = []
    for i in range(0, len(ids), station_batch):
        batch = set(ids[i : i + station_batch])
        obs = load_obs(root, sorted(batch))
        aligned = []
        for model in models:
            runs = load_runs(root, model, batch, columns=cols)
            a = align_runs(runs, model, list(issue_hours), inputs)
            del runs
            if a.empty:
                continue
            if issue_fraction < 1.0:
                issue = a["valid_time"] - pd.to_timedelta(a["lead_h"], unit="h")
                for t in issue.unique():
                    if t not in keep_issue:
                        keep_issue[t] = rng.random() < issue_fraction
                a = a[issue.map(keep_issue).to_numpy()]
            aligned.append(a)
        if not aligned:
            continue
        fc_all = pd.concat(aligned, ignore_index=True)
        for sid, fc in fc_all.groupby("station_id"):
            ob = obs.get(sid)
            if ob is None or ob.empty:
                continue
            built = pairs_mod.build_pairs(fc.drop(columns="station_id"), ob, sid, [variable])
            wide = pairs_mod.to_wide(built, variable, models)
            if wide.empty:
                continue
            wide["station_id"] = sid
            for c in wide.columns:
                if wide[c].dtype == "float64":
                    wide[c] = wide[c].astype("float32")
            tables.append(wide)
        print(f"  stations {min(i + station_batch, len(ids))}/{len(ids)}: "
              f"{sum(len(t) for t in tables):,} rows", flush=True)
    if not tables:
        return pd.DataFrame()
    data = pd.concat(tables, ignore_index=True).merge(stations, on="station_id", how="left")
    vt = pd.to_datetime(data["valid_time"])
    doy, hod = vt.dt.dayofyear.to_numpy(np.float32), vt.dt.hour.to_numpy(np.float32)
    data["sin_doy"], data["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    data["sin_hod"], data["cos_hod"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    return data


# --------------------------------------------------------------------------- model


def feature_names(models: list[str], data: pd.DataFrame) -> list[str]:
    feats = [f"fc_{m}" for m in models] + ["fc_mean", "fc_spread", "lead_h", *STATION_FEATURES, *CALENDAR]
    if "nowcast_err" in data and data["nowcast_err"].notna().mean() > 0.2:
        feats.append("nowcast_err")
    return feats


@dataclass
class NetworkModel:
    variable: str
    models: list[str]
    features: list[str]
    boosters: dict[str, str]

    def predict(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        import lightgbm as lgb

        X = data.reindex(columns=self.features).astype(float)
        out = {k: lgb.Booster(model_str=s).predict(X) for k, s in self.boosters.items()}
        keys = sorted(out)
        mat = np.sort(np.column_stack([out[k] for k in keys]), axis=1)  # no crossing
        if self.variable in ("precip_1h_mm", "wind_speed_ms", "wind_gust_ms", "hn24_cm", "swe_24h_mm", "rh_pct"):
            mat = np.maximum(mat, 0.0)
        if self.variable == "rh_pct":
            mat = np.minimum(mat, 100.0)
        return {k: mat[:, i] for i, k in enumerate(keys)}


def fit(data: pd.DataFrame, variable: str, models: list[str], *, rounds: int = 400) -> NetworkModel:
    """Pooled quantile gradient boosting, one booster per quantile level."""
    import lightgbm as lgb

    feats = feature_names(models, data)
    X = data[feats].astype(float)
    y = data["obs"].to_numpy(float)
    boosters = {}
    for q in quantiles():
        b = lgb.train(
            {"objective": "quantile", "alpha": q, "learning_rate": 0.05, "num_leaves": 63,
             "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8,
             "bagging_freq": 1, "verbosity": -1},
            lgb.Dataset(X, label=y, feature_name=feats), num_boost_round=rounds)
        boosters[f"q{int(q * 100):02d}"] = b.model_to_string()
    return NetworkModel(variable, models, feats, boosters)


# --------------------------------------------------------------------------- evaluation


def leave_stations_out(data: pd.DataFrame, variable: str, models: list[str], *,
                       folds: int = 5, seed: int = 0, rounds: int = 400,
                       test_from: str | pd.Timestamp | None = None) -> dict:
    """Score the pooled model on stations it never saw, against raw models on the same rows.

    With ``test_from``, each fold also trains only on hours before it and is scored only
    on hours from it on: unseen stations *and* an unseen period. Without it, neighbouring
    stations share the test period's weather in training, which flatters the score.

    Returns overall CRPS for the network model and for each raw model on identical rows
    (rows every compared model covers), with CRPSS and a per-station breakdown.
    """
    stations = np.array(sorted(data["station_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)
    fold_of = {s: i % folds for i, s in enumerate(stations)}
    data = data.assign(_fold=data["station_id"].map(fold_of))
    levels = np.array(quantiles())
    scored = []
    cut = pd.Timestamp(test_from) if test_from is not None else None
    vt = pd.to_datetime(data["valid_time"])
    for k in range(folds):
        train, test = data[data["_fold"] != k], data[data["_fold"] == k]
        if cut is not None:
            train, test = train[vt[train.index] < cut], test[vt[test.index] >= cut]
        if train.empty or test.empty:
            continue
        model = fit(train, variable, models, rounds=rounds)
        q = model.predict(test)
        y = test["obs"].to_numpy(float)
        crps = metrics.crps_from_quantiles(y, {kk: v for kk, v in q.items()})
        part = test[["station_id", "valid_time", "lead_h", "obs"]].copy()
        part["crps_network"] = crps
        for m in models:
            part[f"crps_{m}"] = np.abs(test[f"fc_{m}"].to_numpy(float) - y)
        part["q50"] = q["q50"]
        scored.append(part)
        print(f"  fold {k + 1}/{folds}: {test['station_id'].nunique()} unseen stations, "
              f"{len(test):,} rows", flush=True)
    res = pd.concat(scored, ignore_index=True)
    raw_cols = [f"crps_{m}" for m in models]
    same = res.dropna(subset=raw_cols + ["crps_network"])
    out = {"variable": variable, "models": models, "rows": int(len(same)),
           "scheme": ("unseen stations, from " + str(cut.date())) if cut is not None else "unseen stations",
           "stations": int(same["station_id"].nunique()),
           "crps_network": float(same["crps_network"].mean()),
           "levels": levels.tolist()}
    for m in models:
        raw = float(same[f"crps_{m}"].mean())
        out[f"crps_{m}"] = raw
        out[f"crpss_vs_{m}"] = float(1 - out["crps_network"] / raw) if raw > 0 else None
    best = min(models, key=lambda m: out[f"crps_{m}"])
    out["raw_best_model"] = best
    out["crpss_vs_raw_best"] = out[f"crpss_vs_{best}"]
    per = same.groupby("station_id").agg(network=("crps_network", "mean"), raw=(f"crps_{best}", "mean"))
    per["crpss"] = 1 - per["network"] / per["raw"]
    out["stations_improved"] = float((per["crpss"] > 0).mean())
    by_lead = same.assign(bucket=pd.cut(same["lead_h"], [0, 6, 12, 24, 48, 96, 168]))
    lead = by_lead.groupby("bucket", observed=True).agg(network=("crps_network", "mean"),
                                                        raw=(f"crps_{best}", "mean"))
    out["by_lead"] = {str(k): float(1 - r["network"] / r["raw"]) for k, r in lead.iterrows() if r["raw"] > 0}
    return out


def subset(data: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """The table as if only ``models`` had been fused: fc_mean and fc_spread recomputed.

    Lets two model sets be trained from one table and scored on identical rows, which
    is the only fair way to ask whether adding models helps.
    """
    cols = [f"fc_{m}" for m in models]
    out = data.copy()
    with np.errstate(invalid="ignore"):
        out["fc_mean"] = out[cols].mean(axis=1, skipna=True).astype("float32")
        out["fc_spread"] = out[cols].std(axis=1, skipna=True, ddof=0).fillna(0).astype("float32")
    return out


WINTER_MONTHS = (11, 12, 1, 2, 3, 4)


def compare(data: pd.DataFrame, variable: str, sets: dict[str, list[str]], *,
            test_from: str, folds: int = 5, rounds: int = 300, seed: int = 0,
            thresholds: tuple[float, ...] = ()) -> dict:
    """Train each model set on the same table and score all of them on identical rows.

    Rows are unseen stations after ``test_from`` where every model in every set has a
    forecast. Beside the overall CRPS, it reports winter (Nov-Apr) and, for amounts,
    event rows (observed or forecast above zero), so snow-free hours cannot carry the
    score, plus Brier scores at ``thresholds``.
    """
    all_models = sorted({m for ms in sets.values() for m in ms})
    data = data.dropna(subset=[f"fc_{m}" for m in all_models]).reset_index(drop=True)
    stations = np.array(sorted(data["station_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)
    fold_of = {st: i % folds for i, st in enumerate(stations)}
    fold = data["station_id"].map(fold_of).to_numpy()
    vt = pd.to_datetime(data["valid_time"])
    cut = pd.Timestamp(test_from)
    parts = []
    for k in range(folds):
        tr = (fold != k) & (vt < cut).to_numpy()
        te = (fold == k) & (vt >= cut).to_numpy()
        if not tr.any() or not te.any():
            continue
        test = data[te]
        part = test[["station_id", "valid_time", "lead_h", "obs"]].copy()
        for name, ms in sets.items():
            model = fit(subset(data[tr], ms), variable, ms, rounds=rounds)
            q = model.predict(subset(test, ms))
            part[f"crps_{name}"] = metrics.crps_from_quantiles(test["obs"].to_numpy(float), q)
            for t in thresholds:
                levels = [float(kk[1:]) / 100 for kk in sorted(q)]
                mat = np.column_stack([q[kk] for kk in sorted(q)])
                part[f"p{t:g}_{name}"] = metrics.exceedance_probability(levels, mat, t)
        for m in all_models:
            part[f"crps_raw_{m}"] = np.abs(test[f"fc_{m}"].to_numpy(float) - test["obs"].to_numpy(float))
            for t in thresholds:
                part[f"p{t:g}_raw_{m}"] = (test[f"fc_{m}"].to_numpy(float) > t).astype(float)
        part["raw_any"] = test[[f"fc_{m}" for m in all_models]].max(axis=1).to_numpy()
        parts.append(part)
        print(f"  fold {k + 1}/{folds}: {test['station_id'].nunique()} unseen stations, {len(test):,} rows",
              flush=True)
    res = pd.concat(parts, ignore_index=True)
    months = pd.to_datetime(res["valid_time"]).dt.month
    views = {"all": np.ones(len(res), bool), "winter": months.isin(WINTER_MONTHS).to_numpy()}
    if variable in ("precip_1h_mm", "hn24_cm", "swe_24h_mm"):
        views["winter_events"] = views["winter"] & ((res["obs"] > 0) | (res["raw_any"] > 0)).to_numpy()
    names = list(sets) + [f"raw_{m}" for m in all_models]
    out = {"variable": variable, "sets": sets, "scheme": f"unseen stations, from {cut.date()}",
           "stations": int(res["station_id"].nunique())}
    for view, mask in views.items():
        r = res[mask]
        block = {"rows": int(len(r))}
        for n in names:
            block[f"crps_{n}"] = float(r[f"crps_{n}"].mean()) if len(r) else None
        best_raw = min((f"raw_{m}" for m in all_models), key=lambda n: block[f"crps_{n}"] or np.inf)
        block["raw_best"] = best_raw
        for n in sets:
            if block[f"crps_{best_raw}"]:
                block[f"crpss_{n}_vs_raw_best"] = 1 - block[f"crps_{n}"] / block[f"crps_{best_raw}"]
        for t in thresholds:
            y = (r["obs"] > t).astype(float)
            block[f"events_over_{t:g}"] = int(y.sum())
            for n in names:
                block[f"brier_{t:g}_{n}"] = float(((r[f"p{t:g}_{n}"] - y) ** 2).mean()) if len(r) else None
        out[view] = block
    return out


def fetch_archive(dest: str | Path, models: list[str], repo: str = "nakas/wxfuser-archive") -> Path:
    """Download the archive's SNOTEL files and the chosen models' point files."""
    from huggingface_hub import snapshot_download

    patterns = ["snotel/stations.parquet", "snotel/obs/*.parquet",
                *[f"points/{m}/*.parquet" for m in models]]
    path = snapshot_download(repo, repo_type="dataset", local_dir=str(dest), allow_patterns=patterns)
    return Path(path)


def save(model: NetworkModel, path: str | Path, evaluation: dict | None = None) -> Path:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"variable": model.variable, "models": model.models,
                             "features": model.features, "boosters": model.boosters,
                             "evaluation": evaluation}))
    return p


def load(path: str | Path) -> NetworkModel:
    import json

    d = json.loads(Path(path).read_text())
    return NetworkModel(d["variable"], d["models"], d["features"], d["boosters"])
