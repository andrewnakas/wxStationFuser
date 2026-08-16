# wxStationFuser

Pick any surface weather station. Pick any weather models. Get a calibrated probabilistic
forecast for that exact station that is measurably better than the raw models — with the
measurement published next to the forecast.

Everything runs on GitHub Actions and is served as static files from GitHub Pages. There
is no server.

## Why this exists

A global weather model predicts a grid cell several kilometres across, at the model's idea
of the terrain height. A station sits at one point, in a valley or on a ridge, with its own
cold pools, sea breezes, and sensor quirks. The difference between the two is not random —
it is systematic, and it repeats. Learning it from the station's own history is the single
cheapest accuracy gain available for point forecasts, and it has been standard practice in
operational meteorology since Model Output Statistics in 1972.

This project does that learning per station, for any station, and refuses to claim a gain
it has not measured.

## How it works

### 1. Paired history

For each enrolled station the system builds an archive of *what the models predicted* next
to *what the station actually measured*, hour by hour.

Observations come from whichever sources cover the station — NWS, GHCN-hourly, the Iowa
Environmental Mesonet, Meteostat, Synoptic, SNOTEL — merged and quality-controlled
(range, step, and stuck-sensor screens). Forecasts come from Open-Meteo, which archives
what each model predicted on past dates. That archive is the reason a station enrolled
today is useful today: it starts with a year or more of paired history instead of waiting
months to accumulate one.

### 2. A tiered correction, chosen by measurement

Different amounts of history support different methods, so the system fits several and
publishes whichever actually wins out-of-sample.

| Tier | Needs | Method |
|---|---|---|
| 0 | days | Decaying-average bias correction per lead time and hour of day (Delle Monache et al. 2011) |
| 1 | ~2 weeks | **EMOS** — Gaussian predictive distribution fitted by minimum CRPS (Gneiting et al. 2005), exponentially time-weighted (Lang et al. 2020) |
| 2 | ~3 months | EMOS plus seasonal and diurnal harmonics |
| 3 | ~1 year | Gradient-boosted quantile regression |

The mean is a learned combination of every model you selected, and the spread is driven by
how much those models disagree — hours where GFS and ECMWF diverge are genuinely less
predictable, so the intervals widen where they should. Combining models this way is worth
roughly 5–15% on its own, because model errors are only partly correlated.

Wind uses a normal truncated at zero rather than a Gaussian, so it cannot forecast negative
wind. Precipitation is a two-part model — probability of occurrence, then amount — because
it is a mixed variable with a point mass at zero that no single distribution describes.

### 3. Verification that can say no

Every station page publishes the skill gain **and its confidence interval**, and the site
only says a forecast beats the raw model when the 90% bootstrap interval lies entirely
above zero. Where there is no gain, it says so.

The evaluation is walk-forward: fit on the past, predict the next block, refit, repeat.
No fit is ever scored on data it saw. Both the probabilistic score (CRPS) and the
deterministic one (absolute error of the published median) are reported, because a CRPS win
next to an MAE loss means the gain is in expressing uncertainty rather than in a better
central estimate, and that distinction belongs to the reader.

Baselines are compared on identical hours. This matters more than it sounds: HRRR stops at
48 hours, so averaging its error over the hours it covers compares it only on short leads —
the easy ones — while the fused forecast is averaged over everything out to seven days.
Getting that wrong made a winning configuration look 15% *worse* than a model it in fact
beat in every lead bucket. Each model is now scored on its own coverage, with the fused
score recomputed on the same rows, and the coverage is shown.

The method is chosen on rows it is not then scored on. Picking the best of four candidates
and quoting that candidate's score on the same data is optimistically biased — the
selection is not itself out-of-sample — so the walk-forward output is split in time, the
earlier part choosing the method and the later part scoring it.

CRPS is integrated over a dense grid of quantile levels rather than the five published
ones. That sounds like a detail and is not: the coarse version is *exact* for a point
forecast and understates a dispersed one by 12%, and the raw model enters the comparison
as a point forecast — so it inflated every gain on this page by roughly 12 points before
it was fixed. The current estimator errs about 1% in the opposite direction, against our
own claim.

## Results at the first enrolled station

Denver International (`IEM:DEN`), fusing GFS, ECMWF IFS, ICON, and HRRR against one year of
paired history, evaluated walk-forward:

| Variable | Method published | CRPS gain vs best raw model | 90% CI | Absolute-error gain | vs climatology |
|---|---|---|---|---|---|
| Temperature | Tier 2 | 42% | 39–45% | 21% | +34% |
| Relative humidity | Tier 2 | 37% | 36–43% | 16% | +36% |
| Wind speed | Tier 1 | 44% | 44–47% | 23% | +7% |
| Wind gusts | Tier 1 | 70% | 68–73% | 58% | −4% |
| Precipitation | Tier 3 | 46% | 39–60% | 45% | −4% |

Every interval excludes zero, so all five are genuine gains rather than noise. The
climatology column is the uncomfortable one and is published for that reason:
temperature and humidity beat "just use the seasonal average for this hour" by a wide
margin, but gusts and precipitation do not. Beating the physics model at those variables
while barely matching a calendar lookup is worth knowing, and averaging it away would be
the dishonest choice.

The methods differ by variable, and that is the system working rather than an
inconsistency. Temperature and humidity are well described by a Gaussian whose mean is
linear in the model forecasts, so EMOS with harmonics wins and gradient boosting actually
*loses* on them — trees cannot extrapolate beyond the temperatures they were trained on.
Precipitation is skewed with a point mass at zero, which is precisely where the
nonparametric quantile method pulls ahead.

Wind speed and gusts are cases where a different tier scored marginally better (Tier 2 by
0.2%, Tier 3 by 0.9%) but did not clear the 2% promotion margin, so the incumbent stayed.
That margin exists so the published method does not flip between near-equivalent options
every time the job runs — a user should not see the method change without the accuracy
changing. Each station page shows the full per-tier comparison, so the runner-up is
visible rather than hidden.

## Using it

### Enroll a station

Open an [enrollment issue](../../issues/new?template=enroll-station.yml) with the station's
ID, name, and coordinates. An automated job validates it, pulls the history, trains the
models, verifies them, and comments back with a link and the measured skill.

### Run it locally

```bash
pip install -e ".[train,dev]"

wxfuser list                                    # enrolled stations
wxfuser bootstrap IEM:DEN --years 2             # deep backfill + train
wxfuser refresh                                 # update forecasts, write site JSON
wxfuser retrain                                 # re-verify and re-choose methods
python -m http.server -d site                   # view at localhost:8000
```

### Station IDs

| Prefix | Network | Example |
|---|---|---|
| `IEM:` | Airport/ASOS via Iowa Environmental Mesonet (needs `iem_network`) | `IEM:DEN` |
| `NWS:` | US National Weather Service | `NWS:KDEN` |
| `GHCNH:` | GHCN-hourly (deep history) | `GHCNH:USW00003017` |
| `MS:` | Meteostat (~22k global stations) | `MS:72565` |
| `SYN:` | Synoptic/MesoWest (needs `SYNOPTIC_API_TOKEN`) | `SYN:KDEN` |
| *(bare)* | SNOTEL triplet | `663:CO:SNTL` |

## Scheduled jobs

| Workflow | Cadence | Does |
|---|---|---|
| `refresh` | every 3 h | New observations and model runs, refit the cheap tiers, republish |
| `retrain` | weekly | Walk-forward verification, re-choose the published method |
| `enroll` | on issue | Validate, backfill, train, publish, comment back |
| `catalogue` | weekly | Rebuild the searchable global station list |
| `ci` | on push | Lint and tests |

State that grows — paired archives, fitted coefficients, verification history — lives on a
Hugging Face dataset repo rather than in git, so the repository history stays readable.
Set `HF_TOKEN` (and `SYNOPTIC_API_TOKEN` if you use Synoptic) as repository secrets.

## References

Glahn & Lowry (1972), *Model Output Statistics*, J. Appl. Meteor. ·
Gneiting, Raftery, Westveld & Goldman (2005), *Calibrated Probabilistic Forecasting Using
Ensemble Model Output Statistics and Minimum CRPS Estimation*, MWR 133 ·
Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA ·
Thorarinsdottir & Gneiting (2010), *Probabilistic forecasts of wind speed*, JRSS-A ·
Delle Monache et al. (2011), *Kalman Filter and Analog Schemes to Postprocess NWP*, MWR 139 ·
Delle Monache et al. (2013), *Probabilistic Weather Prediction with an Analog Ensemble*, MWR 141 ·
Gneiting & Ranjan (2013), *Combining predictive distributions*, EJS ·
Taillardat et al. (2016), *Calibrated Ensemble Forecasts Using Quantile Regression Forests*, MWR 144 ·
Rasp & Lerch (2018), *Neural Networks for Postprocessing Ensemble Forecasts*, MWR 146 ·
Lang et al. (2020), *Remember the past: time-adaptive training schemes*, NPG 27 ·
Vannitsem et al. (2021), *Statistical Postprocessing for Weather Forecasts*, BAMS ·
Demaeyer et al. (2023), *The EUPPBench postprocessing benchmark dataset*, ESSD 15

## License

MIT
