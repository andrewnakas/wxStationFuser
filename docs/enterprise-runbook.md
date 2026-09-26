# Tree60 Enterprise runbook

How a customer such as a ski patrol or an avalanche centre goes from "we have
stations" to calibrated forecasts in Tree60. The data contract is
[forecast-schema-v2.md](forecast-schema-v2.md).

## One-time setup

1. **Private runner repository.** Create `tree60-enterprise-runner` as a private repo.
   - Copy [`enterprise/org-run.yml`](../enterprise/org-run.yml) into it as
     `.github/workflows/org-run.yml`, and add an `orgs.txt`.
   - Add the secrets `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.
     Use an R2 API token scoped to `tree60-data` with read and write access.
   - Set the repository variable `WXFUSER_REF` to a fuser tag.
2. **Tree60 Worker.** Deploy the `/api/v1/ent/*` routes (TreesixtyFirebase,
   `workers/api/enterprise.js`). They serve `orgs/{org}/site/…` only to members of that
   org, and write the push inbox and uploads.

## Onboarding an org

1. **Name it.** Choose an org id of lowercase letters, digits and dashes, e.g.
   `wasatch-patrol`. Add it to `orgs.txt` in the runner and create `orgs/{org}` in
   Firestore with its members.
2. **Stations.** For each station, decide where its data comes from:

   | Source | Id | Setup |
   |---|---|---|
   | A public network already enrolled (SNOTEL, ASOS) | its existing id | none |
   | Synoptic / MesoWest | `SYN:{stid}` | the runner needs `SYNOPTIC_API_TOKEN` |
   | Their own logger | `ORG:{org}:{name}` | CSV history and/or push API (below) |

3. **History.** Ask for as much logger history as they have, as CSV. A season makes
   tier 2 possible, and a year makes tier 3 possible.

   To check a file locally:

   ```sh
   wxfuser ingest-csv plot.csv --root ./org --station ORG:wasatch-patrol:plot \
     --time-col "Date Time" --tz=-07:00 --lon -111.64 \
     --col "air_temp_c=Air Temp (F):F" --col "wind_gust_ms=Gust (mph):mph" \
     --col "precip_mm=Precip Accum (in):in:cumulative" --col "snow_depth_cm=Depth (in):in"
   ```

   - **Read the diurnal check.** The temperature peak should fall in the afternoon, in
     local solar time. A peak near dawn means the time zone is wrong: SNOTEL's was, for
     years. Loggers kept on standard time all year need a fixed offset (`-07:00`), not
     `America/Denver`.
   - **Cumulative gauges.** Map them `:cumulative`. Jitter is removed, never counted
     as rain.
   - **Uploads through the product.** Upload the CSV to `orgs/{org}/uploads/` with a
     sidecar `.json` holding the same mapping. The runner ingests each file once, by
     content hash.

4. **Specs.** Write `orgs/{org}/specs.yaml`:

   ```yaml
   org: wasatch-patrol
   stations:
     - {id: "ORG:wasatch-patrol:plot", name: "Upper study plot", lat: 40.58, lon: -111.64, elev_m: 2900}
   specs:
     - station: "ORG:wasatch-patrol:plot"
       label: "Plot, HRRR + GEFS"
       models: [hrrr, gefs]
       variables: [air_temp_c, wind_gust_ms, precip_1h_mm, hn24_cm, swe_24h_mm]
       thresholds: {wind_gust_ms: [20, 25], hn24_cm: [15, 30]}
   ```

   - Models must all be dynamical.org ids (`hrrr`, `gfs16`, `gefs`, `ecmwf_ens`,
     `aifs_single`) or all Open-Meteo ids.
   - HRRR is CONUS only.
   - `hn24_cm` needs a depth sensor, and `swe_24h_mm` needs a pillow.

5. **Backfill.** Run the runner workflow manually with `org` set and `bootstrap` on.
   - HRRR history comes from dynamical.org at roughly 5 MB per run per variable per
     ~800 km tile, so 2 years of HRRR for one tile is about 40 GB. Stations in one
     tile share it.
   - Keep backfills to the years the customer needs.
6. **Pilot report.** From `verify.json`, report per variable:
   - CRPSS against the best raw model, with its 90% interval
   - interval coverage
   - skill by lead
   - MAE of the median

   Claim improvement only where `beats_raw` is true. Summer hourly precipitation can
   be worse than climatology, so say so where it is.

## Every run (hourly)

1. Pull `orgs/{org}/`.
2. Ingest new uploads and compact the push inbox.
3. Refresh every spec:
   - a trailing re-pair
   - a refit of the champion
   - a live forecast, with threshold probabilities and `obs_latest`
4. Push only the files that changed.

Weekly, the Sunday run adds `--evaluate`, which re-runs walk-forward verification and
re-chooses each champion.

## Guarantees

- Nothing an org owns is written to the public hub, GitHub Pages or this public
  repository's Actions logs. The public jobs hold no R2 credentials.
- A runner that did not pull an org's state cannot push it.
- A root pulled for one org cannot be pushed as another.
- An org's inbox cannot write another org's stations.
