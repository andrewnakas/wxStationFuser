/* Instant mode — calibrate a station the browser has never seen, in the browser.
 *
 * Enrolled stations get the full treatment: years of paired history, four tiers, weekly
 * walk-forward verification. That takes a scheduled job. But the useful part of this
 * system — "what does this model get wrong *here*" — can be approximated immediately, and
 * doing it client-side means any station is one click away instead of one pull request.
 *
 * What runs here is a deliberately reduced version of the server-side pipeline:
 *
 *   - observations from api.weather.gov (US only, roughly a week of retention)
 *   - archived forecasts from Open-Meteo for the same hours
 *   - Tier 0 bias correction, and Tier 1 EMOS fitted by minimum CRPS when there is enough
 *
 * Both APIs send `Access-Control-Allow-Origin: *`, which is what makes this possible at
 * all from a static page. Meteostat and IEM do not, so non-US stations fall back to the
 * raw multi-model view with an invitation to enroll.
 *
 * The honesty rules are the same as the server's, and matter more here because a week of
 * data is thin: the result is labelled provisional, the sample size is shown, and no
 * skill claim is made from a held-out set this small.
 */

const INSTANT = {
  MIN_PAIRS_TIER0: 24,
  MIN_PAIRS_TIER1: 72,
  QUANTILES: [0.05, 0.25, 0.5, 0.75, 0.95],
  OBS_DAYS: 7,
};

// ---------------------------------------------------------------- statistics

const SQRT2PI = Math.sqrt(2 * Math.PI);
const SQRT_PI = Math.sqrt(Math.PI);

function normPdf(z) {
  return Math.exp(-0.5 * z * z) / SQRT2PI;
}

/* Abramowitz & Stegun 7.1.26 error function; plenty accurate for a predictive CDF. */
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

function normCdf(z) {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

/* Inverse normal CDF (Acklam's rational approximation). */
function normPpf(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
             3.754408661907416];
  const pl = 0.02425;
  let q, r;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/* Closed-form CRPS of N(mu, sigma^2) — the same objective the server minimises. */
function crpsNormal(y, mu, sigma) {
  const s = Math.max(sigma, 1e-6);
  const z = (y - mu) / s;
  return s * (z * (2 * normCdf(z) - 1) + 2 * normPdf(z) - 1 / SQRT_PI);
}

/* Nelder-Mead, because there is no optimiser in the browser and EMOS has few parameters. */
function nelderMead(fn, x0, { maxIter = 400, tol = 1e-7 } = {}) {
  const n = x0.length;
  const alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5;

  let simplex = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    p[i] = p[i] !== 0 ? p[i] * 1.05 : 0.00025;
    simplex.push(p);
  }
  let values = simplex.map(fn);

  for (let iter = 0; iter < maxIter; iter++) {
    const order = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
    simplex = order.map((i) => simplex[i]);
    values = order.map((i) => values[i]);
    if (Math.abs(values[n] - values[0]) < tol) break;

    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
    }
    const worst = simplex[n];
    const reflected = centroid.map((c, j) => c + alpha * (c - worst[j]));
    const fr = fn(reflected);

    if (fr < values[0]) {
      const expanded = centroid.map((c, j) => c + gamma * (reflected[j] - c));
      const fe = fn(expanded);
      if (fe < fr) { simplex[n] = expanded; values[n] = fe; }
      else { simplex[n] = reflected; values[n] = fr; }
    } else if (fr < values[n - 1]) {
      simplex[n] = reflected; values[n] = fr;
    } else {
      const contracted = centroid.map((c, j) => c + rho * (worst[j] - c));
      const fc = fn(contracted);
      if (fc < values[n]) { simplex[n] = contracted; values[n] = fc; }
      else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + sigma * (v - simplex[0][j]));
          values[i] = fn(simplex[i]);
        }
      }
    }
  }
  const best = values.indexOf(Math.min(...values));
  return { x: simplex[best], fval: values[best] };
}

// ---------------------------------------------------------------- data

async function fetchNwsObservations(stationId, days = INSTANT.OBS_DAYS) {
  const end = new Date();
  const start = new Date(end.getTime() - days * 86400000);
  const url =
    `https://api.weather.gov/stations/${encodeURIComponent(stationId)}/observations` +
    `?start=${start.toISOString().slice(0, 19)}Z&end=${end.toISOString().slice(0, 19)}Z&limit=500`;
  const res = await fetch(url, { headers: { Accept: 'application/geo+json' } });
  if (!res.ok) throw new Error(`NWS returned ${res.status}`);
  const data = await res.json();

  const byHour = new Map();
  (data.features || []).forEach((f) => {
    const p = f.properties || {};
    if (!p.timestamp) return;
    const t = new Date(p.timestamp);
    t.setUTCMinutes(0, 0, 0);
    const key = t.toISOString();
    const val = (node) => (node && node.value != null ? node.value : null);
    const row = {
      air_temp_c: val(p.temperature),
      rh_pct: val(p.relativeHumidity),
      // The API reports wind in km/h.
      wind_speed_ms: p.windSpeed && p.windSpeed.value != null ? p.windSpeed.value / 3.6 : null,
      wind_gust_ms: p.windGust && p.windGust.value != null ? p.windGust.value / 3.6 : null,
      precip_1h_mm: val(p.precipitationLastHour),
    };
    if (!byHour.has(key)) byHour.set(key, row);
  });
  return byHour;
}

/* Observations from the Iowa Environmental Mesonet's bulk ASOS archive.
 *
 * This is what extends instant mode past the United States. IEM covers roughly 5,000
 * airport stations worldwide and — checked, not assumed — its request endpoint sends
 * `Access-Control-Allow-Origin: *`, so a static page can read it directly. Without it
 * only the ~4,300 NWS stations could be calibrated in the browser.
 *
 * The response is CSV in US customary units, which is why the conversions live here. */
async function fetchIemObservations(sid, network, days = INSTANT.OBS_DAYS) {
  const end = new Date();
  const start = new Date(end.getTime() - days * 86400000);
  const stamp = (d) => `${d.toISOString().slice(0, 13)}:00Z`;
  const url =
    'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py' +
    `?station=${encodeURIComponent(sid)}&network=${encodeURIComponent(network)}` +
    '&data=tmpf,relh,sknt,gust,p01i&tz=UTC&format=onlycomma&missing=empty' +
    `&sts=${stamp(start)}&ets=${stamp(end)}`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`IEM returned ${res.status}`);
  const text = await res.text();
  const lines = text.trim().split('\n');
  if (lines.length < 2) return new Map();

  const head = lines[0].split(',').map((s) => s.trim());
  const col = (name) => head.indexOf(name);
  const iValid = col('valid');
  const iT = col('tmpf'), iRh = col('relh'), iSk = col('sknt'), iG = col('gust'), iP = col('p01i');

  const F2C = (f) => ((f - 32) * 5) / 9;
  const KT2MS = 0.514444;
  const IN2MM = 25.4;
  const num = (parts, idx) => {
    if (idx < 0) return null;
    const raw = (parts[idx] || '').trim();
    if (!raw) return null;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  };

  const byHour = new Map();
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    const valid = (parts[iValid] || '').trim();
    if (!valid) continue;
    // "2026-08-14 06:35" in UTC; snap to the top of the hour like the server pipeline.
    const t = new Date(`${valid.replace(' ', 'T')}:00Z`);
    if (Number.isNaN(t.getTime())) continue;
    t.setUTCMinutes(0, 0, 0);
    const key = t.toISOString();
    if (byHour.has(key)) continue; // first observation in the hour wins

    const tf = num(parts, iT), sk = num(parts, iSk), g = num(parts, iG), p = num(parts, iP);
    byHour.set(key, {
      air_temp_c: tf === null ? null : F2C(tf),
      rh_pct: num(parts, iRh),
      wind_speed_ms: sk === null ? null : sk * KT2MS,
      wind_gust_ms: g === null ? null : g * KT2MS,
      precip_1h_mm: p === null ? null : p * IN2MM,
    });
  }
  return byHour;
}

async function fetchArchivedForecasts(lat, lon, models, variables, days = INSTANT.OBS_DAYS) {
  const OM_VARS = {
    air_temp_c: 'temperature_2m',
    rh_pct: 'relative_humidity_2m',
    wind_speed_ms: 'wind_speed_10m',
    wind_gust_ms: 'wind_gusts_10m',
    precip_1h_mm: 'precipitation',
  };
  const end = new Date(Date.now() - 86400000);
  const start = new Date(end.getTime() - days * 86400000);
  const iso = (d) => d.toISOString().slice(0, 10);
  const url =
    `https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(4)}` +
    `&longitude=${lon.toFixed(4)}&start_date=${iso(start)}&end_date=${iso(end)}` +
    `&hourly=${variables.map((v) => OM_VARS[v]).join(',')}` +
    `&models=${models.join(',')}&wind_speed_unit=ms&timezone=GMT`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Open-Meteo archive returned ${res.status}`);
  return res.json();
}

async function fetchLiveForecast(lat, lon, models, variables) {
  const OM_VARS = {
    air_temp_c: 'temperature_2m',
    rh_pct: 'relative_humidity_2m',
    wind_speed_ms: 'wind_speed_10m',
    wind_gust_ms: 'wind_gusts_10m',
    precip_1h_mm: 'precipitation',
  };
  const url =
    `https://api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(4)}` +
    `&longitude=${lon.toFixed(4)}&hourly=${variables.map((v) => OM_VARS[v]).join(',')}` +
    `&models=${models.join(',')}&wind_speed_unit=ms&timezone=GMT&forecast_days=7`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Open-Meteo returned ${res.status}`);
  return res.json();
}

/* Pull the per-model series for one variable out of an Open-Meteo response.
 * Keys are `var_model` with several models and bare `var` with one. */
function extractSeries(payload, omVar, models) {
  const hourly = payload.hourly || {};
  const out = {};
  models.forEach((m) => {
    const key = hourly[`${omVar}_${m}`] ? `${omVar}_${m}` : hourly[omVar] ? omVar : null;
    if (key) out[m] = hourly[key];
  });
  return { time: hourly.time || [], series: out };
}

// ---------------------------------------------------------------- fitting

/* Fit mu = a + b * fc_mean, sigma = exp(c + d * spread) by minimum CRPS.
 * The same model as the server's Tier 1, with the per-lead-bucket split dropped —
 * a week of data cannot support six parameter sets. */
function fitEmos(pairs) {
  if (pairs.length < INSTANT.MIN_PAIRS_TIER1) return null;

  const fc = pairs.map((p) => p.fc);
  const sp = pairs.map((p) => p.spread);
  const y = pairs.map((p) => p.obs);

  const meanFc = fc.reduce((a, b) => a + b, 0) / fc.length;
  const meanY = y.reduce((a, b) => a + b, 0) / y.length;
  let num = 0, den = 0;
  for (let i = 0; i < fc.length; i++) {
    num += (fc[i] - meanFc) * (y[i] - meanY);
    den += (fc[i] - meanFc) ** 2;
  }
  const b0 = den > 1e-9 ? num / den : 1.0;
  const a0 = meanY - b0 * meanFc;
  let ss = 0;
  for (let i = 0; i < fc.length; i++) ss += (y[i] - (a0 + b0 * fc[i])) ** 2;
  const sd0 = Math.max(Math.sqrt(ss / fc.length), 0.1);

  const objective = (th) => {
    const [a, b, c, d] = th;
    let total = 0;
    for (let i = 0; i < fc.length; i++) {
      const mu = a + b * fc[i];
      const sigma = Math.exp(Math.min(Math.max(c + Math.max(d, 0) * sp[i], -8), 8));
      total += crpsNormal(y[i], mu, sigma);
    }
    return total / fc.length;
  };

  const { x, fval } = nelderMead(objective, [a0, b0, Math.log(sd0), 0.0]);
  if (!x.every(Number.isFinite)) return null;
  return { a: x[0], b: x[1], c: x[2], d: Math.max(x[3], 0), crps: fval, n: pairs.length };
}

/* Mean error, the fallback when there is too little data for a distribution. */
function fitBias(pairs) {
  if (pairs.length < INSTANT.MIN_PAIRS_TIER0) return null;
  const errs = pairs.map((p) => p.fc - p.obs);
  const bias = errs.reduce((a, b) => a + b, 0) / errs.length;
  const resid = pairs.map((p) => p.obs - (p.fc - bias)).sort((a, b) => a - b);
  const q = {};
  INSTANT.QUANTILES.forEach((level) => {
    const idx = Math.min(resid.length - 1, Math.max(0, Math.round(level * (resid.length - 1))));
    q[level] = resid[idx];
  });
  return { bias, residQuantiles: q, n: pairs.length };
}

const BOUNDS = {
  rh_pct: [0, 100],
  wind_speed_ms: [0, null],
  wind_gust_ms: [0, null],
  precip_1h_mm: [0, null],
};

function applyBounds(values, variable) {
  const b = BOUNDS[variable];
  if (!b) return values;
  return values.map((v) => {
    let out = v;
    if (b[0] !== null && out < b[0]) out = b[0];
    if (b[1] !== null && out > b[1]) out = b[1];
    return out;
  });
}

// ---------------------------------------------------------------- entry point

/* Build a provisional calibrated forecast for a station the site has never trained on. */
async function instantForecast(station, models, variables) {
  // Two observation sources are reachable from a static page: the NWS API (US only) and
  // the IEM ASOS archive (global airports). Everything else — Meteostat, Synoptic, GHCN-h
  // — either blocks cross-origin requests or needs a key, so those stations must be
  // enrolled and fetched server-side.
  const nwsId = station.nws_id || (String(station.id).startsWith('NWS:')
    ? String(station.id).slice(4) : null);
  const iemId = String(station.id).startsWith('IEM:') ? String(station.id).slice(4) : null;
  const iemNetwork = station.iem_network || null;

  let fetchObs = null;
  if (nwsId) fetchObs = () => fetchNwsObservations(nwsId);
  else if (iemId && iemNetwork) fetchObs = () => fetchIemObservations(iemId, iemNetwork);

  if (!fetchObs) {
    return {
      ok: false,
      reason:
        'Instant calibration needs observations a browser can fetch directly, which means ' +
        'the US National Weather Service API or the Iowa Environmental Mesonet archive. ' +
        'This station belongs to neither, so its observations have to be collected ' +
        'server-side — enroll it and the scheduled job will do that, with years of history ' +
        'rather than one week.',
    };
  }

  const [obsByHour, archive, live] = await Promise.all([
    fetchObs(),
    fetchArchivedForecasts(station.lat, station.lon, models, variables),
    fetchLiveForecast(station.lat, station.lon, models, variables),
  ]);

  if (!obsByHour.size) {
    return { ok: false, reason: 'No recent observations were returned for this station.' };
  }

  const OM_VARS = {
    air_temp_c: 'temperature_2m',
    rh_pct: 'relative_humidity_2m',
    wind_speed_ms: 'wind_speed_10m',
    wind_gust_ms: 'wind_gusts_10m',
    precip_1h_mm: 'precipitation',
  };

  const result = { ok: true, station, models, hourly: {}, method: {}, raw: {}, provisional: true };

  variables.forEach((variable) => {
    const omVar = OM_VARS[variable];
    const hist = extractSeries(archive, omVar, models);
    const fut = extractSeries(live, omVar, models);
    if (!hist.time.length || !fut.time.length) return;

    // Pair archived forecasts against what the station actually measured.
    const pairs = [];
    hist.time.forEach((t, i) => {
      const key = new Date(t + 'Z').toISOString();
      const obsRow = obsByHour.get(key);
      if (!obsRow) return;
      const obs = obsRow[variable];
      if (obs === null || obs === undefined || !Number.isFinite(obs)) return;
      const vals = models.map((m) => (hist.series[m] ? hist.series[m][i] : null))
                         .filter((v) => v !== null && Number.isFinite(v));
      if (!vals.length) return;
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const spread = vals.length > 1
        ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length)
        : 0;
      pairs.push({ obs, fc: mean, spread });
    });

    const emos = fitEmos(pairs);
    const bias = emos ? null : fitBias(pairs);
    if (!emos && !bias) return;

    // Apply to the live forecast.
    const n = fut.time.length;
    const bands = {};
    INSTANT.QUANTILES.forEach((q) => { bands[`q${String(Math.round(q * 100)).padStart(2, '0')}`] = []; });
    const rawMeans = [];

    for (let i = 0; i < n; i++) {
      const vals = models.map((m) => (fut.series[m] ? fut.series[m][i] : null))
                         .filter((v) => v !== null && Number.isFinite(v));
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN;
      const spread = vals.length > 1
        ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length)
        : 0;
      rawMeans.push(mean);

      if (emos) {
        const mu = emos.a + emos.b * mean;
        const sigma = Math.exp(Math.min(Math.max(emos.c + emos.d * spread, -8), 8));
        INSTANT.QUANTILES.forEach((q) => {
          bands[`q${String(Math.round(q * 100)).padStart(2, '0')}`].push(mu + sigma * normPpf(q));
        });
      } else {
        const median = mean - bias.bias;
        INSTANT.QUANTILES.forEach((q) => {
          bands[`q${String(Math.round(q * 100)).padStart(2, '0')}`].push(median + bias.residQuantiles[q]);
        });
      }
    }

    // Bound, then re-sort so clamping cannot leave the bands crossed.
    const keys = Object.keys(bands);
    keys.forEach((k) => { bands[k] = applyBounds(bands[k], variable); });
    for (let i = 0; i < n; i++) {
      const col = keys.map((k) => bands[k][i]).sort((a, b) => a - b);
      keys.forEach((k, j) => { bands[k][i] = col[j]; });
    }

    bands.calibrated = true;
    result.hourly.time = fut.time.map((t) => `${t}Z`);
    result.hourly[variable] = bands;
    result.method[variable] = {
      tier: emos ? 'instant-emos' : 'instant-bias',
      label: emos
        ? `In-browser EMOS · ${emos.n} recent hours`
        : `In-browser bias correction · ${bias.n} recent hours`,
      n_pairs: emos ? emos.n : bias.n,
      calibrated: true,
    };
    models.forEach((m) => {
      if (!fut.series[m]) return;
      result.raw[m] = result.raw[m] || {};
      result.raw[m][variable] = fut.series[m];
    });
  });

  if (!Object.keys(result.method).length) {
    return {
      ok: false,
      reason:
        'Not enough overlapping observations and archived forecasts in the last week to ' +
        'calibrate anything for this station.',
    };
  }
  return result;
}

window.wxInstant = { instantForecast, fitEmos, fitBias, nelderMead, crpsNormal, normPpf, normCdf };
