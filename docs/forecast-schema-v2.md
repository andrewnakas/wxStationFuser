# Forecast schema v2: the contract with Tree60

This is what Tree60 reads. The fuser writes it (`src/wxfuser/pipeline/emit.py`), and
the Tree60 Worker serves it to an org's signed-in members. Version 2 is version 1 plus
four fields. Every v1 field keeps its meaning, so a v1 reader works unchanged on v2.

## Where it lives

Everything for an org sits under one R2 prefix in the `tree60-data` bucket:

| Key | What |
|---|---|
| `orgs/{org}/specs.yaml` | the org's specs (private: names stations and locations) |
| `orgs/{org}/site/index.json` | every spec with its status and headline skill |
| `orgs/{org}/site/stations/{slug}/forecast.json` | the forecast (this document) |
| `orgs/{org}/site/stations/{slug}/verify.json` | the full walk-forward verification |
| `orgs/{org}/inbox/*.jsonl` | observations pushed by the org's loggers (Worker writes) |
| `orgs/{org}/uploads/*.csv` | uploaded logger files awaiting ingest (Worker writes) |
| `orgs/{org}/state/…`, `orgs/{org}/observations/…` | calibration state; never served |

`{slug}` is `{station slug}--{spec key}`, for example
`ORG_demo-patrol_snowbird-plot--6021d0ff26`. The spec key is a hash of (org, station,
models, variables). Changing any of those creates a new calibration with its own
history. Thresholds and the label are presentation, so changing them keeps the key.

## `index.json`

```json
{
  "schema_version": 2,
  "org": "demo-patrol",
  "specs": [{
    "key": "6021d0ff26", "slug": "ORG_demo-patrol_snowbird-plot--6021d0ff26",
    "label": "Study plot, HRRR + GEFS",
    "station_id": "ORG:demo-patrol:snowbird-plot", "station_name": "Snowbird study plot (demo)",
    "lat": 40.5691, "lon": -111.6585,
    "models": ["hrrr", "gefs"], "variables": ["air_temp_c", "precip_1h_mm"],
    "thresholds": {"air_temp_c": [0.0], "precip_1h_mm": [0.5, 2.0]},
    "status": "ok", "crpss_vs_raw": 0.7747, "beats_raw": true
  }]
}
```

`status` is one of:

| Value | Meaning |
|---|---|
| `ok` | published |
| `warming_up` | too little paired history to calibrate yet |
| `no_data` | no pairs at all |
| `no_forecast` | the models returned nothing this run |
| `error` | the run failed; the previous `forecast.json` stays in place |

## `forecast.json`

```jsonc
{
  "schema_version": 2,
  "generated_at": "2026-09-26T03:12:40+00:00",
  "station": {"id": "...", "name": "...", "lat": 40.5691, "lon": -111.6585, "elev_m": 2795.0},
  "models_used": ["hrrr", "gefs"],            // fuser model ids; see "Model ids" below
  "method": {                                  // per variable: which tier produced it
    "air_temp_c": {"tier": "tier2", "label": "EMOS + seasonal/diurnal harmonics",
                   "train_days": 92.0, "n_pairs": 27676, "calibrated": true,
                   "models": ["hrrr", "gefs"]}
  },
  "hourly": {
    "time": ["2026-09-26T03:00Z", "..."],      // UTC, hourly; every array below aligns to it
    "air_temp_c": {
      "q05": [...], "q25": [...], "q50": [...], "q75": [...], "q95": [...],
      "calibrated": true,
      "p_exceed": {"0": [0.99, ...]}           // v2: P(value > threshold) per hour
    },
    "precip_1h_mm": {
      "q05": [...], "...": "...",
      "p_occ": [...],                          // P(precip > 0.1 mm); zero-inflated vars only
      "p_exceed": {"0.5": [...], "2": [...]}
    }
  },
  "raw": {"hrrr": {"air_temp_c": [...]}, "gefs": {"...": "..."}},   // each model, uncalibrated
  "skill": {"air_temp_c": { /* see below */ }},
  "spec": {"key": "6021d0ff26", "org": "demo-patrol", "label": "...",   // v2
           "models": [...], "variables": [...], "thresholds": {...}},
  "obs_latest": "2026-09-25T00:00Z"            // v2: newest observation the run saw
}
```

**Nulls.** Any value can be `null` where it is undefined: an hour a model does not
cover, or a statistic with no data behind it. Render a gap, never a zero.

**Quantiles.** `q05`–`q95` are the 5th, 25th, 50th, 75th and 95th percentiles. `q50` is
the headline value. The q25–q75 band holds the observation half the time, and q05–q95
nine times in ten. `skill.coverage50` and `skill.coverage90` report how well that held
in verification.

**Exceedance (`p_exceed`).** Keyed by threshold as text (`"0.5"`, `"20"`). It is read
off the same quantiles by the estimator verification scores
(`metrics.exceedance_probability`). Its resolution stops at the 1% and 99% tails, so
show `0.01` as "≤1%" and `0.99` as "≥99%", never as exact.

**`obs_latest`.** Short-lead skill depends on fresh observations. If `obs_latest` is
more than about 3 hours before `generated_at`, say so next to the 1–6 h forecast.

### `skill`, per variable

| Field | Meaning |
|---|---|
| `status` | `verified`, or `warming_up` (no other fields then) |
| `crpss_vs_raw` | skill gain over the best raw model, on the rows that model covers. **The headline.** |
| `crpss_vs_raw_ci90` | 90% block-bootstrap interval on it |
| `beats_raw` | true only when that whole interval is above zero. Claim "better than raw" only then. |
| `raw_best_model` | the baseline it was measured against |
| `crps_same_rows`, `crps_raw_best` | the fused and raw CRPS behind `crpss_vs_raw`, on identical rows. Compare only these two. |
| `crps` | fused CRPS over *all* rows. Not comparable with `crps_raw_best` (different rows). |
| `mae_median`, `mae_raw_best`, `mae_gain_vs_raw` | the same comparison for the median alone |
| `coverage50`, `coverage90` | how often observations fell inside the 50% and 90% bands |
| `crpss_vs_climo` | skill against climatology. Negative means worse than the historical average for that hour and day of year, which honestly happens for summer hourly precipitation. |
| `evaluation_scheme`, `n_eval` | walk-forward, refit every 14 d; how many hours were scored |

## Variables

| Id | Unit | Kind | Notes |
|---|---|---|---|
| `air_temp_c` | °C | continuous | |
| `rh_pct` | % | bounded 0–100 | |
| `wind_speed_ms` | m/s | ≥0 | 10 m, or the station's sensor height |
| `wind_gust_ms` | m/s | ≥0 | not every model carries gusts |
| `precip_1h_mm` | mm | zero-inflated | has `p_occ` |
| `swe_24h_mm` | mm | zero-inflated | 24 h SWE gain ending at each hour; has `p_occ` |
| `hn24_cm` | cm | zero-inflated | 24 h new snow ending at each hour; has `p_occ` |

The 24 h variables are published for every hour: `hn24_cm` at `07:00` is 07:00 yesterday
to 07:00 today. To show the morning HN24 an avalanche forecaster uses, pick the hour of
their study-plot reading.

## Model ids

A spec names fuser model ids. Each dynamical.org entry in `configs/models.yaml` carries
the Tree60 `ModelId` it matches (`tree60_id`):

| Fuser id | Tree60 `ModelId` |
|---|---|
| `hrrr` | `hrrr` |
| `gfs16` | `gfs16` |
| `gefs` | `gefs` |
| `ecmwf_ens` | `ecmwf-ens` |
| `aifs_single` | `aifs-single` |

A spec's models must all come from one source. The UI's model picker should offer the
dynamical.org set together, and not mix them with the Open-Meteo ids.

## Push API (Worker → inbox)

The Worker authenticates a logger by a per-station key and appends one JSON line per
reading to `orgs/{org}/inbox/{yyyy-mm-dd}.jsonl`:

```json
{"station": "ORG:demo-patrol:snowbird-plot", "time": "2026-01-11T07:10:00-07:00",
 "air_temp_c": -6.1, "wind_speed_ms": 7.2, "wind_gust_ms": 13.0, "precip_mm": 0.2,
 "snow_depth_cm": 142.0, "swe_mm": 410.0}
```

- Units are canonical. `precip_mm` is the amount since the previous reading, and
  `snow_depth_cm` and `swe_mm` are the current state.
- Times need an offset or `Z`.
- The runner ignores any record for a station outside the inbox's org, so a leaked
  key cannot write another org's data.
