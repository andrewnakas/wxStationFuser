/* wxStationFuser front end.
 *
 * Static files only: an index of enrolled stations, then one forecast.json and
 * verify.json per station. No build step, no framework — the data is small and the
 * charts are simple enough that canvas beats a charting dependency.
 *
 * The interface is built around one claim: this forecast is better than the raw model.
 * So the skill numbers are not buried in a tab — they are the first thing shown, with
 * their confidence interval, and they are allowed to say "no measurable gain" when that
 * is what the verification found.
 */

// Above this, a station's calibration is history rather than current conditions and the
// page says so. Set well past a normal refresh gap so ordinary lateness stays quiet.
const STALE_OBS_DAYS = 7;

const VARIABLE_LABELS = {
  air_temp_c: { name: 'Temperature', unit: '°C', short: 'Temp' },
  rh_pct: { name: 'Relative humidity', unit: '%', short: 'RH' },
  wind_speed_ms: { name: 'Wind speed', unit: 'm/s', short: 'Wind' },
  wind_gust_ms: { name: 'Wind gust', unit: 'm/s', short: 'Gust' },
  precip_1h_mm: { name: 'Precipitation', unit: 'mm/h', short: 'Precip' },
};

const state = {
  index: null,
  stations: [],
  catalogue: null,       // lazily loaded; tens of thousands of un-enrolled stations
  catalogueLoading: false,
  selected: null,
  variable: 'air_temp_c',
  view: 'forecast',
  forecast: null,
  verify: null,
  showRaw: true,
  instant: false,        // true when the forecast was calibrated in the browser
};

const DEFAULT_MODELS = ['gfs_seamless', 'ecmwf_ifs025', 'icon_seamless'];
const DEFAULT_MODELS_CONUS = ['gfs_seamless', 'ecmwf_ifs025', 'icon_seamless', 'ncep_hrrr_conus'];
const ALL_VARIABLES = ['air_temp_c', 'rh_pct', 'wind_speed_ms', 'wind_gust_ms', 'precip_1h_mm'];

function modelsFor(st) {
  const inConus = st.lat >= 24 && st.lat <= 50 && st.lon >= -125 && st.lon <= -66;
  return inConus ? DEFAULT_MODELS_CONUS : DEFAULT_MODELS;
}

let map, markerLayer, catalogueLayer;

// ---------------------------------------------------------------- utilities

const $ = (id) => document.getElementById(id);

function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(digits);
}

function pct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(digits)}%`;
}

function skillClass(entry) {
  if (!entry || entry.status !== 'verified') return 'mute';
  if (entry.beats_raw) return 'good';
  if (entry.crpss_vs_raw > 0) return 'warn';
  return 'bad';
}

async function getJSON(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json();
}

// ---------------------------------------------------------------- map + list

function initMap() {
  map = L.map('map', { zoomControl: false, attributionControl: false }).setView([39.8, -98.5], 3);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 18,
  }).addTo(map);
  markerLayer = L.layerGroup().addTo(map);
  catalogueLayer = L.layerGroup().addTo(map);
  // Panning or zooming changes which stations are in view, so redraw then — debounced,
  // because Leaflet fires these continuously during a drag.
  let pending = null;
  const redraw = () => {
    clearTimeout(pending);
    pending = setTimeout(() => {
      // Enrolled markers are viewport-filtered too when there are many, so they need
      // redrawing on pan; `false` keeps the map from re-fitting and fighting the user.
      if (state.stations.length > 200) renderMap(false);
      else renderCatalogueMarkers();
    }, 150);
  };
  map.on('moveend zoomend', redraw);
}

/* A station's state, kept in one place because "no measurement yet" and "measured and
 * worse" are different claims and must never be drawn the same way. An earlier version
 * tested `crpss_vs_raw > 0`, which is false for null, so every station still accumulating
 * history was coloured as though it had been measured and lost to the raw model. */
function stationState(st) {
  if (st.status !== 'ok') return 'unavailable';
  if (st.crpss_vs_raw === null || st.crpss_vs_raw === undefined) return 'warming';
  if (st.beats_raw) return 'beats';
  if (st.crpss_vs_raw > 0) return 'inconclusive';
  return 'worse';
}

const STATE_COLOR = {
  beats: '#3fb950',        // measured, conclusively better
  inconclusive: '#d29922', // measured, better but the interval includes zero
  worse: '#f85149',        // measured, not better
  warming: '#39c5cf',      // enrolled, still accumulating history (distinct from
                           // the blue used for clickable un-enrolled stations)
  unavailable: '#8b949e',  // no observations or no forecast
};

const STATE_PILL = {
  beats: 'good', inconclusive: 'warn', worse: 'bad',
  warming: 'mute', unavailable: 'mute',
};

function markerColor(st) {
  return STATE_COLOR[stationState(st)];
}

// Above this zoom the map also draws un-enrolled catalogue stations. Below it there would
// be thousands in view, which is both unreadable and slow to render.
const CATALOGUE_MIN_ZOOM = 7;
const CATALOGUE_MAX_MARKERS = 600;
// Rows rendered in the sidebar before it summarises the remainder.
const ENROLLED_LIST_LIMIT = 60;

function canInstant(st) {
  return Boolean(st.nws_id) || String(st.id).startsWith('NWS:') ||
    (String(st.id).startsWith('IEM:') && st.iem_network);
}

function renderMap(fit = true) {
  markerLayer.clearLayers();

  // Enrolled stations: larger, coloured by measured skill. Once the registry runs to
  // hundreds these are viewport-filtered like the catalogue, or every pan redraws the
  // whole set.
  const bounds = map.getBounds();
  const filterToView = state.stations.length > 200;
  const pts = [];
  state.stations.forEach((st) => {
    if (st.lat == null || st.lon == null) return;
    if (filterToView && !bounds.contains([st.lat, st.lon])) return;
    pts.push([st.lat, st.lon]);
    const m = L.circleMarker([st.lat, st.lon], {
      radius: 7,
      color: markerColor(st),
      weight: 2,
      fillColor: markerColor(st),
      fillOpacity: 0.6,
    });
    m.bindTooltip(`${st.name} — enrolled`, { direction: 'top' });
    m.on('click', () => selectStation(st.id));
    m.addTo(markerLayer);
  });

  renderCatalogueMarkers();
  if (fit && pts.length) map.fitBounds(pts, { padding: [30, 30], maxZoom: 8 });
}

/* Draw catalogue stations inside the current view so any of them can simply be clicked.
 * Rendering is bounded two ways — a zoom floor and a hard marker cap — because the
 * catalogue holds 27k stations and Leaflet will happily try to draw all of them. */
function renderCatalogueMarkers() {
  if (!catalogueLayer) return;
  catalogueLayer.clearLayers();
  if (map.getZoom() < CATALOGUE_MIN_ZOOM) {
    setMapHint(`Zoom in to see all ${state.catalogue ? state.catalogue.length.toLocaleString() : ''} stations`);
    return;
  }
  if (!state.catalogue) {
    ensureCatalogue();
    setMapHint('Loading stations…');
    return;
  }

  const bounds = map.getBounds();
  const enrolled = new Set(state.stations.map((s) => s.id));
  let shown = 0;
  let inView = 0;

  for (const st of state.catalogue) {
    if (enrolled.has(st.id)) continue;
    if (!bounds.contains([st.lat, st.lon])) continue;
    inView++;
    if (shown >= CATALOGUE_MAX_MARKERS) continue;
    shown++;
    const instant = canInstant(st);
    const m = L.circleMarker([st.lat, st.lon], {
      radius: 4,
      color: instant ? '#58a6ff' : '#6e7681',
      weight: 1,
      fillColor: instant ? '#58a6ff' : '#6e7681',
      fillOpacity: instant ? 0.5 : 0.25,
    });
    m.bindTooltip(`${st.name}${instant ? '' : ' — needs enrolling'}`, { direction: 'top' });
    m.on('click', () => selectCatalogueStation(st));
    m.addTo(catalogueLayer);
  }

  setMapHint(
    inView > shown
      ? `${shown.toLocaleString()} of ${inView.toLocaleString()} stations shown — zoom in for the rest`
      : `${shown.toLocaleString()} station${shown === 1 ? '' : 's'} in view · click any to calibrate`
  );
}

function setMapHint(text) {
  const el = $('mapHint');
  if (el) el.textContent = text || '';
}

/* The map now encodes five states; without a key the colours are just decoration. */
function renderMapLegend() {
  const el = $('mapLegend');
  if (!el) return;
  const swatch = (c, label) =>
    `<span class="lg"><i style="background:${c}"></i>${label}</span>`;
  el.innerHTML =
    swatch(STATE_COLOR.beats, 'beats raw') +
    swatch(STATE_COLOR.inconclusive, 'better, not conclusive') +
    swatch(STATE_COLOR.worse, 'no gain') +
    swatch(STATE_COLOR.warming, 'building history') +
    swatch('#58a6ff', 'click to calibrate') +
    swatch('#6e7681', 'needs enrolling');
}

/* The full catalogue is only needed once someone searches beyond the enrolled list,
 * so it is fetched on demand rather than on page load. */
async function ensureCatalogue() {
  if (state.catalogue || state.catalogueLoading) return;
  state.catalogueLoading = true;
  try {
    const raw = await getJSON('stations.min.json');
    const cols = raw.columns;
    state.catalogue = raw.stations.map((row) => {
      const o = {};
      cols.forEach((c, i) => { o[c] = row[i]; });
      o.enrolled = false;
      return o;
    });
  } catch {
    state.catalogue = [];
  } finally {
    state.catalogueLoading = false;
    renderList();
    renderCatalogueMarkers();
  }
}

/* Match every whitespace-separated token, so "denver intl" finds a station named
 * "DENVER INTERNATIONAL AIRPORT" — a single substring match would not. */
function matchesQuery(haystack, tokens) {
  return tokens.every((t) => haystack.includes(t));
}

function searchCatalogue(q, limit = 40) {
  if (!state.catalogue || !q) return [];
  const tokens = q.split(/\s+/).filter(Boolean);
  if (!tokens.length) return [];
  const enrolled = new Set(state.stations.map((s) => s.id));
  const out = [];
  for (const st of state.catalogue) {
    if (enrolled.has(st.id)) continue;
    if (matchesQuery(`${st.name} ${st.id}`.toLowerCase(), tokens)) {
      out.push(st);
      if (out.length >= limit) break;
    }
  }
  return out;
}

function renderList() {
  const q = ($('search').value || '').toLowerCase().trim();
  const ul = $('stations');
  ul.innerHTML = '';

  const tokens = q.split(/\s+/).filter(Boolean);
  const enrolledMatches = state.stations.filter(
    (s) => !q || matchesQuery(`${s.name} ${s.id}`.toLowerCase(), tokens)
  );

  if (enrolledMatches.length) {
    // Verified stations first: with hundreds enrolled, the ones with a measured result
    // are what a reader wants, and rendering every row would stall the page.
    const ranked = [...enrolledMatches].sort((a, b) => {
      const av = a.crpss_vs_raw ?? -Infinity;
      const bv = b.crpss_vs_raw ?? -Infinity;
      return bv - av;
    });
    const shown = ranked.slice(0, ENROLLED_LIST_LIMIT);
    const verified = enrolledMatches.filter((s) => s.crpss_vs_raw != null).length;
    ul.insertAdjacentHTML(
      'beforeend',
      `<li class="meta">Enrolled: ${enrolledMatches.length.toLocaleString()} · ` +
        `${verified.toLocaleString()} with a measured result</li>`
    );
    shown.forEach((st) => ul.appendChild(enrolledRow(st)));
    if (ranked.length > shown.length) {
      ul.insertAdjacentHTML(
        'beforeend',
        `<li class="meta">…and ${(ranked.length - shown.length).toLocaleString()} more — ` +
          'search by name, or click one on the map</li>'
      );
    }
  }

  if (q && q.length >= 2) {
    ensureCatalogue();
    const others = searchCatalogue(q);
    if (others.length) {
      ul.insertAdjacentHTML(
        'beforeend',
        '<li class="meta" style="margin-top:8px">Not enrolled &mdash; calibrated live in your browser</li>'
      );
      others.forEach((st) => ul.appendChild(catalogueRow(st)));
    } else if (state.catalogueLoading) {
      ul.insertAdjacentHTML('beforeend', '<li class="meta">Searching all stations…</li>');
    } else if (state.catalogue && !state.catalogue.length) {
      // Distinguish "the catalogue has not been built yet" from "nothing matched" —
      // they look identical to a user but mean very different things.
      ul.insertAdjacentHTML(
        'beforeend',
        '<li class="meta">The full station list has not been built yet, so only enrolled ' +
          'stations are searchable. Run the catalogue workflow to populate it.</li>'
      );
    }
  }

  if (!ul.children.length) {
    ul.innerHTML = '<li class="meta">No stations match.</li>';
  }
}

const STATE_LABEL = {
  warming: 'building history',
  unavailable: 'no data',
  worse: 'no gain',
};

function enrolledRow(st) {
  const li = document.createElement('li');
  if (state.selected === st.id) li.className = 'active';
  const s = stationState(st);
  const pill =
    s === 'beats' || s === 'inconclusive' || s === 'worse'
      ? `<span class="pill ${STATE_PILL[s]}">${(st.crpss_vs_raw * 100).toFixed(0)}%</span>`
      : `<span class="pill ${STATE_PILL[s]}">${STATE_LABEL[s]}</span>`;
  const elev = st.elev_m ? ` · ${Math.round(st.elev_m)} m` : '';
  li.innerHTML = `<div class="name">${st.name} ${pill}</div>
                  <div class="meta">${st.id}${elev}</div>`;
  li.onclick = () => selectStation(st.id);
  return li;
}

function catalogueRow(st) {
  const li = document.createElement('li');
  if (state.selected === st.id) li.className = 'active';
  const badge = canInstant(st)
    ? '<span class="pill mute">instant</span>'
    : '<span class="pill mute">enroll</span>';
  li.innerHTML = `<div class="name">${st.name} ${badge}</div>
                  <div class="meta">${st.id}${st.country ? ' · ' + st.country : ''}</div>`;
  li.onclick = () => selectCatalogueStation(st);
  return li;
}

// ---------------------------------------------------------------- charts

/* Size a canvas to its container at device resolution.
 * The CSS width must be applied before measuring, otherwise the canvas reports its
 * intrinsic 300px default and everything draws into a narrow strip. */
function sizeCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = '100%';
  canvas.style.height = `${height}px`;
  const W = Math.max(Math.round(canvas.getBoundingClientRect().width) || 0, 280);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, height);
  return { ctx, W, H: height };
}

function drawForecastChart(canvas, times, band, rawSeries, unit) {
  const { ctx, W, H } = sizeCanvas(canvas, 320);

  // Room at the top for the unit label so it cannot collide with the highest axis tick.
  const padL = 52, padR = 14, padT = 26, padB = 30;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const all = [];
  ['q05', 'q25', 'q50', 'q75', 'q95'].forEach((k) => band[k] && all.push(...band[k].filter(Number.isFinite)));
  if (state.showRaw) Object.values(rawSeries).forEach((s) => all.push(...s.filter(Number.isFinite)));
  if (!all.length) return;

  let ymin = Math.min(...all), ymax = Math.max(...all);
  const padY = (ymax - ymin) * 0.1 || 1;
  ymin -= padY; ymax += padY;

  const n = times.length;
  const X = (i) => padL + (plotW * i) / Math.max(n - 1, 1);
  const Y = (v) => padT + plotH - (plotH * (v - ymin)) / (ymax - ymin);

  // grid + y axis
  ctx.strokeStyle = '#2a323d';
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const v = ymin + ((ymax - ymin) * g) / 4;
    const y = Y(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(1), 6, y + 3);
  }

  // day boundaries + labels
  ctx.fillStyle = '#8b949e';
  for (let i = 0; i < n; i++) {
    const d = new Date(times[i]);
    if (d.getUTCHours() === 0) {
      ctx.strokeStyle = '#39414d';
      ctx.beginPath();
      ctx.moveTo(X(i), padT);
      ctx.lineTo(X(i), padT + plotH);
      ctx.stroke();
      ctx.fillText(`${d.getUTCMonth() + 1}/${d.getUTCDate()}`, X(i) + 3, H - 10);
    }
  }

  const bandFill = (lo, hi, color) => {
    if (!lo || !hi) return;
    ctx.fillStyle = color;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(hi[i])) continue;
      if (!started) { ctx.moveTo(X(i), Y(hi[i])); started = true; }
      else ctx.lineTo(X(i), Y(hi[i]));
    }
    for (let i = n - 1; i >= 0; i--) {
      if (!Number.isFinite(lo[i])) continue;
      ctx.lineTo(X(i), Y(lo[i]));
    }
    ctx.closePath();
    ctx.fill();
  };

  bandFill(band.q05, band.q95, 'rgba(88,166,255,0.16)');
  bandFill(band.q25, band.q75, 'rgba(88,166,255,0.32)');

  if (state.showRaw) {
    const colors = ['#d29922', '#a371f7', '#f85149', '#3fb950', '#79c0ff'];
    Object.entries(rawSeries).forEach(([, s], idx) => {
      ctx.strokeStyle = colors[idx % colors.length];
      ctx.globalAlpha = 0.55;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        if (!Number.isFinite(s[i])) continue;
        if (!started) { ctx.moveTo(X(i), Y(s[i])); started = true; }
        else ctx.lineTo(X(i), Y(s[i]));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    });
  }

  // median
  if (band.q50) {
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(band.q50[i])) continue;
      if (!started) { ctx.moveTo(X(i), Y(band.q50[i])); started = true; }
      else ctx.lineTo(X(i), Y(band.q50[i]));
    }
    ctx.stroke();
  }

  ctx.fillStyle = '#8b949e';
  ctx.font = '11px sans-serif';
  ctx.fillText(unit, 6, 14);
}

function drawPIT(canvas, hist) {
  const { ctx, W, H } = sizeCanvas(canvas, 170);
  if (!hist || !hist.length) return;

  const padL = 34, padB = 22, padT = 10, padR = 8;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const total = hist.reduce((a, b) => a + b, 0) || 1;
  const ideal = 1 / hist.length;
  const maxFrac = Math.max(...hist.map((c) => c / total), ideal * 1.6);

  const bw = plotW / hist.length;
  hist.forEach((c, i) => {
    const frac = c / total;
    const h = (frac / maxFrac) * plotH;
    ctx.fillStyle = 'rgba(88,166,255,0.55)';
    ctx.fillRect(padL + i * bw + 1, padT + plotH - h, bw - 2, h);
  });

  // uniform reference — a calibrated forecast sits on this line
  const yIdeal = padT + plotH - (ideal / maxFrac) * plotH;
  ctx.strokeStyle = '#f85149';
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, yIdeal);
  ctx.lineTo(W - padR, yIdeal);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#8b949e';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('0', padL, H - 8);
  ctx.fillText('1', W - padR - 6, H - 8);
  ctx.fillText('PIT', padL + plotW / 2 - 10, H - 8);
}

function drawLeadBars(canvas, byLead) {
  const { ctx, W, H } = sizeCanvas(canvas, 170);
  const entries = Object.entries(byLead || {}).filter(([, v]) => v.crpss_vs_raw != null);
  if (!entries.length) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px sans-serif';
    ctx.fillText('No per-lead breakdown available.', 10, 30);
    return;
  }

  const padL = 40, padB = 26, padT = 10, padR = 8;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const vals = entries.map(([, v]) => v.crpss_vs_raw);
  // Headroom so bars near the maximum are not flush against the frame, and so the
  // per-bar value labels have somewhere to sit.
  const maxAbs = Math.max(0.05, ...vals.map(Math.abs)) * 1.25;
  const zeroY = padT + plotH / 2;
  const bw = plotW / entries.length;

  ctx.strokeStyle = '#2a323d';
  ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(W - padR, zeroY); ctx.stroke();

  entries.forEach(([label, v], i) => {
    const h = ((v.crpss_vs_raw / maxAbs) * plotH) / 2;
    const cx = padL + i * bw + bw / 2;
    ctx.fillStyle = v.crpss_vs_raw >= 0 ? 'rgba(63,185,80,0.65)' : 'rgba(248,81,73,0.65)';
    ctx.fillRect(padL + i * bw + 2, h >= 0 ? zeroY - h : zeroY, bw - 4, Math.abs(h));

    ctx.textAlign = 'center';
    ctx.fillStyle = '#e6edf3';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(
      `${(v.crpss_vs_raw * 100).toFixed(0)}%`,
      cx,
      h >= 0 ? zeroY - h - 4 : zeroY + Math.abs(h) + 11
    );

    ctx.fillStyle = '#8b949e';
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(label + 'h', cx, H - 6);
    ctx.textAlign = 'start';
  });

  ctx.fillStyle = '#8b949e';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(`+${(maxAbs * 100).toFixed(0)}%`, 4, padT + 8);
  ctx.fillText(`-${(maxAbs * 100).toFixed(0)}%`, 4, padT + plotH);
}

// ---------------------------------------------------------------- rendering

function headlineHTML(sk, method, instant) {
  if (instant) {
    return `<div class="headline">
      <div class="big mute">Provisional calibration</div>
      <div class="detail">
        Fitted in your browser just now from about a week of this station's observations
        against the archived model forecasts for the same hours
        (${method ? method.label : '—'}). That is enough to correct a systematic bias and
        give an honest sense of spread, but far too little to verify a skill gain, so no
        accuracy claim is made here.
      </div>
    </div>`;
  }
  if (!sk || sk.status !== 'verified') {
    return `<div class="headline">
      <div class="big mute">Warming up</div>
      <div class="detail">Not enough verified forecast/observation pairs yet to measure a skill gain.
      The forecast below is still calibrated to this station, but its advantage over the raw model is unproven.</div>
    </div>`;
  }
  const gain = sk.crpss_vs_raw;
  const ci = sk.crpss_vs_raw_ci90 || [null, null];
  const cls = sk.beats_raw ? 'good' : 'mute';
  const verdict = sk.beats_raw
    ? `${pct(gain, 0)} more accurate than ${sk.raw_best_model}`
    : gain > 0
      ? `${pct(gain, 0)} better, but not conclusively`
      : 'No measurable gain over the raw model';

  const maeLine =
    sk.mae_gain_vs_raw == null
      ? ''
      : `<br>Median error: ${fmt(sk.mae_median)} vs ${fmt(sk.mae_raw_best)} raw
         (<strong>${sk.mae_gain_vs_raw >= 0 ? '' : ''}${pct(sk.mae_gain_vs_raw, 0)}</strong> on absolute error).`;

  return `<div class="headline">
    <div class="big ${cls}">${verdict}</div>
    <div class="detail">
      CRPS ${fmt(sk.crps, 3)} vs ${fmt(sk.crps_raw_best, 3)} for the best raw model
      (${sk.raw_best_model}), 90% CI on the gain ${pct(ci[0], 0)} to ${pct(ci[1], 0)},
      ${sk.n_eval} out-of-sample hours.
      ${maeLine}
      <br>Method: ${method ? method.label : '—'}.
      ${sk.evaluation_scheme ? `Evaluated ${sk.evaluation_scheme}.` : ''}
    </div>
  </div>`;
}

function renderForecast() {
  const f = state.forecast;
  const v = state.variable;
  const meta = VARIABLE_LABELS[v] || { name: v, unit: '' };
  const block = f.hourly[v];
  const sk = (f.skill || {})[v];
  const method = (f.method || {})[v];

  if (!block) {
    return `<div class="notice">No calibrated forecast for ${meta.name} at this station yet.</div>`;
  }

  const rawNote = !block.calibrated
    ? `<div class="notice">This variable is bias-corrected but has too few residuals for
       reliable uncertainty bands, so only the central estimate is shown.</div>`
    : '';

  // Some networks publish in arrears. Such a station still gets a calibrated forecast —
  // its archive is real — but the correction was fitted to observations that stop weeks
  // or months ago, which is a weaker claim than one fitted to last night's. Saying so is
  // the price of publishing it at all.
  const age = f.obs_age_days;
  const staleNote = (age != null && age > STALE_OBS_DAYS)
    ? `<div class="notice">The newest observation from this station is
       ${Math.round(age)} days old, so the correction is fitted to history rather than to
       current conditions. Its network publishes on a delay.</div>`
    : '';

  return `
    ${headlineHTML(sk, method, state.instant)}
    ${state.instant ? enrollLinkHTML(f.station) : ''}
    ${staleNote}
    ${rawNote}
    <div class="chart-wrap">
      <canvas id="fcChart"></canvas>
      <div class="legend">
        <span><span class="sw" style="background:#58a6ff"></span>Fused median</span>
        <span><span class="sw" style="background:rgba(88,166,255,0.32)"></span>50% interval</span>
        <span><span class="sw" style="background:rgba(88,166,255,0.16)"></span>90% interval</span>
        <span><label><input type="checkbox" id="rawToggle" ${state.showRaw ? 'checked' : ''}> show raw models</label></span>
      </div>
    </div>`;
}

function renderScorecard() {
  const vr = state.verify;
  const v = state.variable;
  if (state.instant) {
    return `<div class="notice">This station was calibrated in your browser from about a
      week of observations. Verification requires holding out data the model never saw,
      which needs far more history than that — enroll the station and the scheduled job
      will publish a full walk-forward scorecard.</div>
      ${enrollLinkHTML(state.forecast.station)}`;
  }
  if (!vr || !vr.variables || !vr.variables[v] || vr.variables[v].insufficient_data) {
    return `<div class="notice">No verification available for this variable yet.
      ${vr && vr.variables && vr.variables[v] && vr.variables[v].reason ? vr.variables[v].reason : ''}</div>`;
  }
  const c = vr.variables[v];
  const tierRows = Object.entries(c.tier_crps || {})
    .sort((a, b) => a[1] - b[1])
    .map(
      ([t, s]) =>
        `<tr><td>${t}${t === c.champion ? ' <span class="pill good">champion</span>' : ''}</td>
         <td class="num">${fmt(s, 3)}</td></tr>`
    )
    .join('');

  const rawRows = Object.entries(c.crps_raw_by_model || {})
    .sort((a, b) => (a[1].crps ?? 0) - (b[1].crps ?? 0))
    .map(
      ([m, r]) =>
        `<tr><td>${m}${m === c.raw_best_model ? ' <span class="pill mute">baseline</span>' : ''}</td>
         <td class="num">${fmt(r.crps, 3)}</td>
         <td class="num">${fmt(r.crps_fused_same_rows, 3)}</td>
         <td class="num">${pct(r.crpss, 0)}</td>
         <td class="num">${pct(r.coverage, 0)}</td></tr>`
    )
    .join('');

  return `
    <div class="grid2">
      <div>
        <h3>Calibration (PIT)</h3>
        <div class="chart-wrap"><canvas id="pitChart"></canvas></div>
        <p class="sub">A calibrated forecast gives a flat histogram on the red line.
        A U shape means the intervals are too narrow; a hump means too wide.
        ${c.pit_randomized ? 'Randomised PIT is used here because precipitation has a point mass at zero.' : ''}</p>
      </div>
      <div>
        <h3>Where the gain comes from</h3>
        <div class="chart-wrap"><canvas id="leadChart"></canvas></div>
        <p class="sub">Skill against the best raw model, by forecast lead time.</p>
      </div>
    </div>

    <h3>Interval reliability</h3>
    <table>
      <tr><th>Interval</th><th class="num">Coverage</th><th class="num">Target</th><th class="num">Mean width</th></tr>
      <tr><td>50%</td><td class="num">${pct(c.coverage50)}</td><td class="num">50.0%</td><td class="num">${fmt(c.sharpness50)}</td></tr>
      <tr><td>90%</td><td class="num">${pct(c.coverage90)}</td><td class="num">90.0%</td><td class="num">${fmt(c.sharpness90)}</td></tr>
    </table>
    <p class="sub">Coverage should match the target. Width is shown alongside because a
    forecast can always reach its coverage target by being uselessly vague.</p>

    <div class="grid2">
      <div>
        <h3>Method comparison (out-of-sample CRPS)</h3>
        <table>${tierRows || '<tr><td class="sub">—</td></tr>'}</table>
      </div>
      <div>
        <h3>Versus each raw model</h3>
        <table>
          <tr><th>Model</th><th class="num">Raw</th><th class="num">Fused</th>
              <th class="num">Gain</th><th class="num">Covers</th></tr>
          ${rawRows || '<tr><td class="sub">—</td></tr>'}
        </table>
        <p class="sub">Each comparison uses only the hours that model actually forecasts,
        with the fused score recomputed on the same hours. "Covers" is the share of
        evaluated hours the model reaches — regional models like HRRR stop at short leads,
        so comparing them across all leads would be meaningless.</p>
      </div>
    </div>

    <h3>Baselines</h3>
    <table>
      <tr><th>Comparison</th><th class="num">Skill score</th></tr>
      <tr><td>vs best raw model</td><td class="num">${pct(c.crpss_vs_raw)}</td></tr>
      <tr><td>vs climatology</td><td class="num">${pct(c.crpss_vs_climo)}</td></tr>
    </table>
    <p class="sub">Evaluated ${c.evaluation_scheme || 'out-of-sample'} over ${c.n_eval} hours
    from ${c.n_pairs} paired records.</p>`;
}

function render() {
  const el = $('main');
  if (!state.selected) {
    el.innerHTML = `<div class="empty">Select a station to see its forecast.</div>`;
    return;
  }
  const f = state.forecast;
  if (!f) {
    el.innerHTML = `<div class="empty">Loading station…</div>`;
    return;
  }

  const vars = Object.keys(f.hourly).filter((k) => k !== 'time');
  const varTabs = vars
    .map(
      (v) =>
        `<button class="tab ${v === state.variable ? 'active' : ''}" data-var="${v}">${
          (VARIABLE_LABELS[v] || { short: v }).short
        }</button>`
    )
    .join('');

  el.innerHTML = `
    <h2>${f.station.name}</h2>
    <div class="sub">${f.station.id} · ${fmt(f.station.lat, 3)}, ${fmt(f.station.lon, 3)}
      ${f.station.elev_m ? `· ${fmt(f.station.elev_m, 0)} m` : ''}
      · fusing ${f.models_used.join(', ')}</div>

    <div class="tabs">${varTabs}</div>
    <div class="tabs">
      <button class="tab ${state.view === 'forecast' ? 'active' : ''}" data-view="forecast">Forecast</button>
      <button class="tab ${state.view === 'scorecard' ? 'active' : ''}" data-view="scorecard">Verification</button>
    </div>
    <div id="panel">${state.view === 'forecast' ? renderForecast() : renderScorecard()}</div>
  `;

  el.querySelectorAll('[data-var]').forEach((b) => {
    b.onclick = () => { state.variable = b.dataset.var; render(); };
  });
  el.querySelectorAll('[data-view]').forEach((b) => {
    b.onclick = () => { state.view = b.dataset.view; render(); };
  });

  if (state.view === 'forecast') {
    const block = f.hourly[state.variable];
    if (block) {
      const raw = {};
      Object.entries(f.raw || {}).forEach(([model, vars2]) => {
        if (vars2[state.variable]) raw[model] = vars2[state.variable];
      });
      const meta = VARIABLE_LABELS[state.variable] || { unit: '' };
      drawForecastChart($('fcChart'), f.hourly.time, block, raw, meta.unit);
      const t = $('rawToggle');
      if (t) t.onchange = () => { state.showRaw = t.checked; render(); };
    }
  } else {
    const c = (state.verify && state.verify.variables && state.verify.variables[state.variable]) || {};
    if ($('pitChart')) drawPIT($('pitChart'), c.pit_hist);
    if ($('leadChart')) drawLeadBars($('leadChart'), c.by_lead);
  }
}

// ---------------------------------------------------------------- flow

/* An un-enrolled station: calibrate it in the browser, right now.
 *
 * This is a genuinely weaker product than an enrolled station — a week of observations
 * instead of years, one global fit instead of per-lead-time ones, and no out-of-sample
 * verification at all — so the result is labelled provisional throughout rather than
 * being presented as equivalent. */
async function selectCatalogueStation(st) {
  state.selected = st.id;
  state.forecast = null;
  state.verify = null;
  state.instant = true;
  renderList();
  $('main').innerHTML = `<div class="empty">Calibrating ${st.name} in your browser…</div>`;
  history.replaceState(null, '', `?station=${encodeURIComponent(st.id)}`);

  try {
    const result = await window.wxInstant.instantForecast(st, modelsFor(st), ALL_VARIABLES);
    if (!result.ok) {
      $('main').innerHTML = `
        <h2>${st.name}</h2>
        <div class="sub">${st.id} · ${fmt(st.lat, 3)}, ${fmt(st.lon, 3)}</div>
        <div class="notice">${result.reason}</div>
        ${enrollLinkHTML(st)}`;
      return;
    }
    result.station = { id: st.id, name: st.name, lat: st.lat, lon: st.lon, elev_m: st.elev_m };
    result.models_used = modelsFor(st);
    result.skill = {};
    state.forecast = result;
    const vars = Object.keys(result.hourly).filter((k) => k !== 'time');
    if (!vars.includes(state.variable)) state.variable = vars[0];
    render();
  } catch (err) {
    $('main').innerHTML = `
      <h2>${st.name}</h2>
      <div class="notice">In-browser calibration failed: ${err.message}</div>
      ${enrollLinkHTML(st)}`;
  }
}

function repoSlug() {
  // owner.github.io/repo/ -> owner/repo
  const host = location.hostname.replace('.github.io', '');
  const parts = location.pathname.split('/').filter(Boolean);
  return parts.length ? `${host}/${parts[0]}` : `${host}/${host}`;
}

function enrollLinkHTML(st) {
  const params = new URLSearchParams({
    template: 'enroll-station.yml',
    title: `Enroll: ${st.name}`,
    station_id: st.id,
    name: st.name,
    lat: String(st.lat),
    lon: String(st.lon),
  });
  if (st.elev_m != null) params.set('elev', String(st.elev_m));
  if (st.iem_network) params.set('iem_network', st.iem_network);
  if (st.nws_id) params.set('nws_id', st.nws_id);
  const url = `https://github.com/${repoSlug()}/issues/new?${params.toString()}`;
  return `<div class="notice info">
    <strong>Want the full treatment?</strong>
    <a href="${url}" target="_blank" rel="noopener">Enroll this station</a> and a scheduled
    job will backfill years of history, fit every method, verify them out-of-sample, and
    publish a measured skill number instead of a provisional one.</div>`;
}

async function selectStation(id) {
  state.instant = false;
  state.selected = id;
  state.forecast = null;
  state.verify = null;
  renderList();
  render();
  const st = state.stations.find((s) => s.id === id);
  if (!st || !st.slug) {
    $('main').innerHTML = `<div class="notice">This station has no published forecast yet
      (status: ${st ? st.status : 'unknown'}).</div>`;
    return;
  }
  history.replaceState(null, '', `?station=${encodeURIComponent(id)}`);
  try {
    state.forecast = await getJSON(`stations/${st.slug}/forecast.json`);
    const vars = Object.keys(state.forecast.hourly).filter((k) => k !== 'time');
    if (!vars.includes(state.variable)) state.variable = vars[0];
    render();
    state.verify = await getJSON(`stations/${st.slug}/verify.json`);
    if (state.view === 'scorecard') render();
  } catch (err) {
    $('main').innerHTML = `<div class="notice">Could not load this station: ${err.message}</div>`;
  }
}

async function boot() {
  initMap();
  $('search').oninput = renderList;
  try {
    state.index = await getJSON('index.json');
  } catch {
    $('main').innerHTML = `<div class="notice">No station index published yet.
      Run the refresh workflow to populate the site.</div>`;
    return;
  }
  state.stations = state.index.stations || [];
  $('generated').textContent = state.index.generated_at
    ? `updated ${new Date(state.index.generated_at).toUTCString().replace('GMT', 'UTC')}`
    : '';
  renderMap();
  renderList();
  renderMapLegend();

  const wanted = new URLSearchParams(location.search).get('station');
  // Open on a station that actually has a measured result. Landing on one that is still
  // building history shows a provisional forecast and no skill number, which is the least
  // informative first impression the site can give.
  const first =
    [...state.stations]
      .filter((s) => s.status === 'ok' && s.crpss_vs_raw != null)
      .sort((a, b) => (b.crpss_vs_raw ?? 0) - (a.crpss_vs_raw ?? 0))[0] ||
    state.stations.find((s) => s.status === 'ok');
  if (wanted && state.stations.some((s) => s.id === wanted)) selectStation(wanted);
  else if (first) selectStation(first.id);
  else render();
}

window.addEventListener('resize', () => { if (state.forecast) render(); });
boot();
