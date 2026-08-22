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
Environmental Mesonet, Meteostat, SNOTEL, and Synoptic where licensed — merged and quality-controlled
(range, step, and stuck-sensor screens). Forecasts come from an archive of what each model
predicted on past dates, which is the reason a station enrolled today is useful today: it
starts with a year or more of paired history instead of waiting months to accumulate one.

#### Where the archived forecasts come from, and why it matters

There are two providers, and they are not interchangeable.

**Open-Meteo** serves forecasts by point, and offers two archives. The *seamless* one
reaches back years but stores the best-available forecast for each hour, so every row of
it is short-lead by construction — the code tags those rows lead 3 h and they legitimately
train the 1-6 h bucket and nothing longer. The *previous-runs* one is genuinely
lead-resolved and is therefore the only honest basis for day-2-to-day-7 skill, and it
reaches back **92 days**, at daily lead offsets. The coefficients for the 97-168 h bucket
have been fitted on about 92 samples per station.

**dynamical.org** publishes the raw model archives as Zarr — every run, every lead hour,
on the native grid. GFS back to 2021-05, HRRR to 2018-07 (and the only archive here that
carries gusts at every lead), ECMWF's AIFS to 2024-04, ICON-EU to 2026-02. Two years of
daily initialisations gives roughly **730 samples per lead hour** instead of 92, and the
lead axis is real rather than nominal.

Reading a global 4-D array to get one station sounds absurd, and would be, done naively.
The chunking is what makes it work: GFS is chunked (1 init, 105 leads, 121 lat, 121 lon),
so one read covers a 30-degree tile — and 2,422 of the enrolled stations sit inside the
busiest one. Grouping stations by tile and indexing them out of the slab in memory turns
one request per station into one request per tile. Measured against the live archive:

| | measured |
|---|---|
| per initialisation, variable and tile | 0.23 s, 3.1 MB |
| tiles occupied by 9,040 stations | 59 (busiest holds 2,422) |
| two-year backfill, daily inits, five variables | ~14 h of transfer, 0.67 TB — under an hour per worker across 16 shards |

Read it per station instead and the same backfill is 50,000 times the traffic for the same
numbers.

The ensembles were measured too and deliberately left out. GEFS carries 31 members but
chunks 17x16 cells, so the fleet spans 811 tiles and the same backfill costs 4.3 TB; IFS
ENS costs 5.7 TB. A true ensemble spread would be the better uncertainty predictor, and it
is not affordable at this scale — so the spread still comes from disagreement between
models.

What it costs on the other side is storage, and the factor is about ten. A station's
paired archive today averages 88 KB across all of its models; two years of grid history
measures **465 KB per station per model** (39,949 paired rows at Denver), so a two-model
station is roughly 930 KB and a fleet of 9,040 is about 8.6 GB. The dataset repo holds
that comfortably; what it changes is the restore, which is per-shard traffic on every job
— the 14 MB a worker currently pulls becomes closer to 500 MB. That is the price of the
depth, and it is why the lead axis is thinned to every third hour: the published forecast
buckets leads anyway, so storing every hour would triple the archive to sharpen an axis
that is then averaged over.

A station's models must all come from one provider. The training matrix is keyed on lead
source as well as valid time, so a point row and a grid row for the same hour stay separate
rows, and each would become a one-model ensemble with no spread at all. Nothing raises; the
station simply calibrates on half the evidence. The default sets are provider-homogeneous
and a test keeps them that way.

#### The depth does not improve the forecast, and that is worth saying

The grid archives were added on the expectation that eight times the samples per lead
would calibrate better. Measured, they do not.

Three stations (Denver, Boston, Seattle), leads beyond 48 h, `dyn_gfs` on both sides,
walked forward with the same block length and scored **only on the rows both arms
predicted** — so the method, the station and the hours are identical and the sole
difference is how much history each fit could see:

| | tier 1 | tier 2 |
|---|---|---|
| temperature | −1.1% to −3.1% | −0.5% to −19.0% |
| wind speed | −0.8% to +1.7% | −0.3% to +1.5% |

Negative means two years of history scored *worse* than 92 days. Wind is noise; temperature
is consistently a little worse. That direction is not a surprise on reflection — the tiers
are exponentially time-weighted precisely because old data dilutes a fit (Lang et al. 2020,
in the references below), and EMOS has about four parameters per lead bucket against
thousands of samples per bucket already. It was not short of data.

The obvious follow-up hypothesis was seasonal: a station enrolled in July has an archive
that has never seen a winter, and the walk-forward can never expose that because it always
trains on the block immediately before the one it scores. So: fix an evaluation block, fit
once, and vary only whether the training window reaches back to the same season a year
earlier. Winter, spring and summer blocks, identical test rows.

That is negative too. Tier 1 moves between −3.0% and +1.4% with no pattern — the winter
block, where the hypothesis predicted the gain, comes in at +1.3%, −1.1% and −1.6% across
the three stations. Having last winter in the archive does not measurably help this winter.

**Tier 2 is a different matter, and the finding is about the method rather than the data.**
Given two years it degrades badly and repeatably: −48.8% at Boston in summer, −34.6% at
Seattle, −15.1% on Boston's wind.

The cause is a gate that measures the wrong thing. Tier 2 adds annual and diurnal
harmonics, and it already refuses the annual ones unless the training window spans enough
of the year to identify them — 180 days, measured as the calendar distance from the oldest
retained row to the newest. A deep archive passes that test while the exponential time
weighting quietly ignores most of what it counted. Measured per lead bucket at Boston and
Seattle:

| lead bucket | calendar span | span holding 90% of the weight | annual harmonic |
|---|---|---|---|
| 25-48 h | 299 d | 69 d | on |
| 49-96 h | 406-412 d | 132-135 d | on |
| 97-168 h | 253-281 d | 128-131 d | on |

So four annual coefficients per bucket are fitted from roughly one season of effective
data and then applied across the rest of the year. Against a 92-day archive, where the gate
correctly refuses them, that is the whole of the loss.

Nothing published today is wrong: the walk-forward selection rejects Tier 2 wherever it
loses, which is exactly what "verification that can say no" is for. But the latent version
of this exists on the current data too — the seamless archive gives the 1-6 h bucket two
years of hourly rows, so that bucket passes the calendar gate now. `configs/tiers.yaml`
therefore carries `harmonic_gate`, which can measure the span on the weight instead. It is
left on `calendar` deliberately: the evidence is three stations on one source, and the
change would alter what existing stations publish at short leads. It is written down and
switchable rather than quietly applied.

So the case for reading these archives is **throughput, not skill**. What it buys is the
ability to backfill a fleet at all: the Open-Meteo arm of this very experiment was killed
by its own 40-minute timeout having banked nothing for three stations, and the reduced
version only completed after two rate-limit waits, while the grid archive delivered three
stations of two-year lead-resolved history in 23 minutes. Reading the grid cell instead of
an interpolated point costs nothing measurable — six comparisons on the same window split
three each way within ±4 points — so the switch is available whenever throughput is the
binding constraint. It is opt-in per station (`wxfuser set-models`) and no station uses it
by default.

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

Denver International (`IEM:DEN`), fusing GFS, ECMWF IFS, ICON, and HRRR against two years
of paired history, evaluated walk-forward. These are the numbers the live page shows:

| Variable | Method chosen | CRPS gain vs best raw model | Absolute-error gain |
|---|---|---|---|
| Temperature | Tier 2 — EMOS + harmonics | 43% | 22% |
| Relative humidity | Tier 3 — boosted quantiles | 40% | 18% |
| Wind speed | Tier 1 — EMOS | 44% | 23% |
| Wind gusts | Tier 3 — boosted quantiles | 69% | 57% |
| Precipitation | Tier 3 — boosted quantiles | 46% | 44% |

Every confidence interval excludes zero, so all five are genuine gains rather than noise.
The live page carries the intervals, the per-method comparison, and the skill against
climatology — which is the uncomfortable number and is published for that reason. At this
station temperature and humidity beat "just use the seasonal average for this hour" by a
wide margin while gusts and precipitation barely match it. Beating the physics model at a
variable where a calendar lookup does nearly as well is worth knowing, and averaging it
away would be the dishonest choice.

The method differs by variable, and that is the system working rather than an
inconsistency. Temperature is well described by a Gaussian whose mean is linear in the
model forecasts, so EMOS with harmonics wins and gradient boosting actually *loses* on it
— trees cannot extrapolate beyond the temperatures they were trained on. Gusts and
precipitation are skewed and heteroscedastic, which is exactly where the nonparametric
quantile method pulls ahead.

A challenger must beat the incumbent by 2% to take over. That margin exists so the
published method does not flip between near-equivalent options every time the job runs — a
user should not see the method change without the accuracy changing. Each station page
shows the full per-tier comparison, so the runner-up is visible rather than hidden.

## Using it

### Enroll a station

Open an [enrollment issue](../../issues/new?template=enroll-station.yml) with the station's
ID, name, and coordinates. An automated job validates it, pulls the history, trains the
models, verifies them, and comments back with a link and the measured skill.

You do not have to. The `grow` job enrolls stations on its own — see
[How the fleet fills itself](#how-the-fleet-fills-itself) — so the form is for asking for a
station sooner than the queue would reach it.

### Run it locally

```bash
pip install -e ".[train,dev]"

wxfuser list                                    # enrolled stations
wxfuser bootstrap IEM:DEN --years 2             # deep backfill + train
wxfuser refresh                                 # update forecasts, write site JSON
wxfuser retrain                                 # re-verify and re-choose methods
python -m http.server -d site                   # view at localhost:8000

# What the scheduled jobs do, if you want to watch them decide
wxfuser enroll-bulk --meteostat --limit 20 --dry-run    # next 20 stations to enroll
wxfuser refresh --bootstrap --only-untrained \
                --order prominence --limit 20           # train 20 of the backlog
```

### Station IDs

| Prefix | Network | Example |
|---|---|---|
| `ASOS:` | Global airport observations, in bulk from dynamical.org (~4,100 stations) | `ASOS:DEN` |
| `IEM:` | Airport/ASOS via Iowa Environmental Mesonet (needs `iem_network`) | `IEM:DEN` |
| `NWS:` | US National Weather Service | `NWS:KDEN` |
| `GHCNH:` | GHCN-hourly (deep history) | `GHCNH:USW00003017` |
| `MS:` | Meteostat (~22k global stations) | `MS:72565` |
| `SYN:` | Synoptic/MesoWest — see note below | `SYN:KDEN` |
| *(bare)* | SNOTEL triplet | `663:CO:SNTL` |

## Scheduled jobs

| Workflow | Cadence | Does |
|---|---|---|
| `refresh` | every 6 h | New observations and model runs, refit the cheap tiers, republish |
| `bootstrap` | nightly | Backfill and train the stations that are registered but have no archive yet |
| `grow` | weekly | Enroll new stations from the bulk sources, most-populated first |
| `retrain` | weekly | Walk-forward verification, re-choose the published method |
| `catalogue` | weekly | Rebuild the searchable global station list |
| `enroll` | on issue | Validate, backfill, train, publish, comment back |
| `ci` | on push | Lint and tests |

Nothing in that table needs a person. `grow` selects and registers stations, `bootstrap`
gives them their history, `refresh` keeps their forecasts current, and `retrain` decides
what each one publishes — so the fleet enrolls, trains, and verifies itself, and the issue
form is how someone asks for one *particular* station rather than how stations get added.

State that grows — paired archives, fitted coefficients, verification history — lives on a
Hugging Face dataset repo rather than in git, so the repository history stays readable.
Set `HF_TOKEN` as a repository secret to use it; without one the state falls back to the
Actions cache, which works but is evicted after a week of inactivity.

Synoptic is supported but effectively unavailable for this project: as of 2026 their free
Open Access tier requires a `.edu` address from an accredited US institution and
explicitly excludes personal and hobbyist use. `SYN:` stations therefore need a paid plan
and `SYNOPTIC_API_TOKEN`. Nothing depends on it — Synoptic's unique contribution was some
US mesonet coverage (RAWS, state networks), and the other five sources cover the rest.

## How the fleet fills itself

The registry can be grown faster than it can be trained, and for a long time it was: as of
this writing 9,040 stations are registered and 1,326 have a paired archive. Backfilling two
years of model history is the expensive step, so the useful question is not how many
stations exist but **which** ones get the budget. Registry order answers that badly — it is
alphabetical by station id, so it spends the night on whichever ICAO code sorts first.

So every job that cannot finish everything now works in order of the population a station
serves:

```
served_pop = sum over cities within 150 km of  population * exp(-distance / 30 km)
```

Cities come from the GeoNames `cities15000` gazetteer — every settlement above 15,000
people, one 3 MB download, no API key. The decay is there because a city with its own
station nearby has no use for a distant one, and the cutoff because a station 200 km away
is a different forecast rather than a worse one. Ranking the whole registry takes 0.4 s.

It puts São Paulo, Shanghai, Osaka, Rio de Janeiro and London at the front, which is the
point: those are the stations somebody actually searches for. Three jobs use it.

* **`grow`** picks what to enroll next, so the fleet fills with the stations people look up
  rather than with an arbitrary slice of the alphabet.
* **`bootstrap`** works the backlog most-populated first, capped per worker per night, so a
  run that hits the 350-minute job limit has banked the stations that matter and the next
  night continues where it stopped.
* **`refresh`** publishes in the same order, so when the forecast API throttles it is the
  least-read stations whose pages go stale.

Elevation breaks ties rather than being replaced by population, because the two rank
different things. A SNOTEL site in an empty mountain range serves nobody and scores zero —
and it is one of the most valuable stations in the fleet, since a model's idea of the
terrain is most wrong exactly there. Population decides which *populated* station comes
first; among the unpopulated ones the mountains still win.

Served population is a proxy and behaves like one. It knows nothing about whether a station
reports reliably, whether its observations arrive fresh enough to verify against, or whether
the models are already good there. Those are answered downstream, by the verification that
can say no.

### Airports are taken whole

Pacing applies to the sources that are effectively unbounded. The airport network is not
one of them, so `grow` enrolls **every** station in dynamical.org's archive that has
reported in the last year — 4,062 of them — and keeps doing so.

That is affordable because the archive answers in bulk: one DuckDB query covers every
airport in a shard, so a thousand stations cost what ten do. The observation side of an
airport is free; only its forecasts are rationed, and that rationing is what the ordering
above is for.

Which stations count as reporting used to be decided on the *previous calendar year* — a
test that judges a 2026 station on its 2025 record. It was hiding 279 live stations,
including the airports at Tokyo, Delhi, Seoul and Johannesburg, plus sites like Ship Shoal
and Pipestone that now report more often than hourly and simply had a quiet 2025. The
window is now a rolling year, so a station commissioned in March is enrolled that week
rather than the following January, and one that has been silent for a year drops out.

The bar for "reporting" is now a single observation in that year. A sparse reporter is not
a silent failure here — it trains on what it has, publishes `obs_age_days`, and reports
itself unmeasured until verification says otherwise — and excluding the weakest few dozen
was costing 279 good stations to save a forecast budget they barely move.

### Everything else is paced

`grow` declines to outrun the rest of the system. `--max-backlog` counts the registered
stations with no archive yet and adds only enough to top that number up, so enrollment
tracks what the nightly bootstrap can actually train. Registering ten thousand stations
that never get trained does not grow the site — it grows the backlog, and the site shows
exactly what it showed before.

## What limits the number of stations

Four ceilings, measured rather than assumed.

**Open-Meteo throttling** is the one that actually stops work. The API prices a request by
locations x models x variables rather than by request count, and refuses with a 429 above a
budget. Measured against the previous-runs endpoint: 10 locations x 4 models returns in
about 4 s, while 25 x 4 and 53 x 1 are both refused and clear roughly a minute later. That
is why the history endpoints batch 10 locations where the plain forecast endpoint batches
100, and why a shard's wall clock is dominated by one sequential pass per model
(measured per 10-location chunk: 1.4 s on gfs_seamless, 25 s on icon_seamless, 36 s on
ncep_hrrr_conus).

Sustained use exhausts it. After a night of refreshing, a single worker running alone was
still throttled 26 times in a row and could not complete one chunk. Refreshing 4,760
stations across four models is not something the free tier will do quickly, and no amount
of sharding changes that — the quota is the ceiling, not the parallelism.

This ceiling is the one dynamical.org's grid archives sidestep: they are read from object
storage rather than from a metered API, and the cost scales with the number of *tiles* a
fleet occupies rather than the number of stations in it. See
[Where the archived forecasts come from](#where-the-archived-forecasts-come-from-and-why-it-matters).
The models available there are a different set, so it is a change of ingredients and not
only of plumbing.

**Observation freshness** decides whether a station can be verified at all. A forecast can
only be scored where recent observations exist, so a network that publishes on a delay
yields stations that train but cannot be checked. Meteostat's bulk archive is the live
example: as of August 2026 its year-partitioned files stop at 2026-03-29, its full record
at 2025-08, and its `full/` variant at 2022. Those stations need a deep bootstrap rather
than an incremental refresh, and they publish `obs_age_days` so the staleness is legible
instead of inferred.

**Hugging Face request limits** shape how the fleet starts rather than how large it gets.
The hub allows 1000 API requests per five minutes. Twenty workers each restoring the whole
archive exceeded that, and every one of them failed on its first step — correctly refusing
to continue rather than overwrite deep history with a shallow rebuild. Each worker now
restores only the stations its shard owns (measured: 14 MB rather than 141 MB) and the
fleet staggers its starts. The quota tracks metadata calls rather than file volume: three
back-to-back scoped restores, 2,142 files in 78 seconds, drew no limit at all.

**GitHub Pages** is the loosest of the three. At roughly 83 KB per station against a ~1 GB
soft limit, the site holds on the order of 12,000 stations.

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
