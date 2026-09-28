"""The forecast replay page: any past date, and the biggest snow days, calibrated vs raw.

Built from ``wxfuser history`` output. The page is self-contained (data inline, no
libraries) and follows the report's design (report.STYLE and FONTS). Every calibrated
forecast in it came out of the walk-forward replay, so each is what the system could
have issued on that date, never a refit with hindsight.
"""
from __future__ import annotations

import html
import json

from wxfuser.report import FONTS, MODEL_NAMES, STYLE

EXTRA_STYLE = """
.lead{max-width:64ch;color:var(--soft);margin:12px 0 0}
.switch{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.pill{font:500 14px/1 var(--body);padding:9px 14px;border:1px solid var(--rule);border-radius:999px;
color:var(--ink);background:var(--panel);cursor:pointer}
.pill[aria-pressed="true"]{background:var(--ink);color:var(--ground);border-color:var(--ink)}
.pill:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.scores{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin-top:18px}
.score{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;display:grid;gap:6px}
.score h3{font:600 20px/1 var(--display);margin:0}
.bar{display:grid;grid-template-columns:92px 1fr 64px;align-items:center;gap:8px;font-size:12.5px}
.bar .track{height:8px;background:var(--rule);border-radius:4px;overflow:hidden}
.bar .fill{display:block;height:100%;background:var(--soft)}
.bar.cal .fill{background:var(--accent)}
.bar .v{font:500 12.5px var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.events{display:grid;gap:8px;margin-top:10px}
.event{display:grid;grid-template-columns:minmax(120px,1.2fr) repeat(auto-fit,minmax(78px,1fr));gap:4px 12px;
align-items:center;text-align:left;background:var(--panel);border:1px solid var(--rule);border-radius:6px;
padding:10px 14px;cursor:pointer;font:inherit;color:inherit}
.event:hover,.event:focus-visible{border-color:var(--accent);outline:none}
.event .d{font:600 18px/1.1 var(--display)}
.event .k{font-size:11px;color:var(--soft);text-transform:uppercase;letter-spacing:.06em}
.event .n{font:500 15px var(--mono);font-variant-numeric:tabular-nums}
.event .n.best{color:var(--accent);font-weight:600}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:12px}
.controls input[type=date]{font:500 14px var(--mono);padding:7px 10px;border:1px solid var(--rule);border-radius:6px;
background:var(--panel);color:var(--ink)}
.step{font:600 16px/1 var(--body);width:38px;height:36px;border:1px solid var(--rule);border-radius:6px;
background:var(--panel);color:var(--ink);cursor:pointer}
table.miss{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
table.miss th,table.miss td{padding:7px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
table.miss th:first-child,table.miss td:first-child{text-align:left}
table.miss th{font:500 11px var(--body);letter-spacing:.06em;text-transform:uppercase;color:var(--soft)}
table.miss td.best{color:var(--accent);font-weight:600}
.tablewrap{overflow-x:auto}
.obsdot{fill:var(--ink)}
.empty{color:var(--soft);padding:24px 0}
"""

SCRIPT = r"""
const D = JSON.parse(document.getElementById('hist').textContent);
const NAMES = %NAMES%;
const VARS = {
  hn24_cm: {label: 'New snow, 24 h', unit: 'in', conv: v => v / 2.54, dec: 1, snow: true},
  swe_24h_mm: {label: 'SWE, 24 h', unit: 'in', conv: v => v / 25.4, dec: 2, snow: true},
  air_temp_c: {label: 'Temperature', unit: '°F', conv: v => v * 9 / 5 + 32, dec: 0},
};
const st = {station: 0, variable: 'hn24_cm', issueHourLocal: 8, day: null};
const $ = id => document.getElementById(id);
const H = 3600e3;

function zone(s) { const h = Math.round((s.lon || -105) / 15); return {h, name: ({'-7':'MST','-8':'PST','-6':'CST','-5':'EST'})[h] || ('UTC' + h)}; }
function fmt(v, dec) { return v == null || Number.isNaN(v) ? '—' : v.toFixed(dec); }
function local(ms, z) { return new Date(ms + z.h * H); }
function dayKey(ms, z) { return local(ms, z).toISOString().slice(0, 10); }
function label(ms, z, withHour) {
  const d = local(ms, z), wd = d.toLocaleDateString('en-US', {weekday: 'short', timeZone: 'UTC'});
  const md = d.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'});
  return withHour ? `${wd} ${md} ${String(d.getUTCHours()).padStart(2, '0')}:00` : `${wd} ${md}`;
}

function rowsFor(stIdx, variable) {
  const b = D.stations[stIdx].variables[variable];
  if (!b) return null;
  if (b._rows) return b;
  const r = b.rows, n = r.issue.length, out = [];
  for (let i = 0; i < n; i++) {
    const o = {issue: r.issue[i] * H, lead: r.lead[i], obs: r.obs[i], q05: r.q05[i], q25: r.q25[i], q50: r.q50[i], q75: r.q75[i], q95: r.q95[i], raw: {}};
    for (const m of D.models) o.raw[m] = r['raw_' + m][i];
    o.valid = o.issue + o.lead * H;
    out.push(o);
  }
  b._rows = out;
  b._issues = [...new Set(out.map(o => o.issue))].sort((a, b) => a - b);
  return b;
}

function renderStations() {
  $('stations').innerHTML = D.stations.map((s, i) =>
    `<button class="pill" aria-pressed="${i === st.station}" data-i="${i}">${s.name}</button>`).join('');
  $('stations').querySelectorAll('button').forEach(b => b.onclick = () => { st.station = +b.dataset.i; st.day = null; renderAll(); });
}

function renderScores() {
  const s = D.stations[st.station];
  const cards = Object.entries(VARS).map(([v, meta]) => {
    const b = s.variables[v]; if (!b || !b.summary.rows) return '';
    const sm = b.summary, scale = meta.conv(1) - meta.conv(0);
    const items = [['Calibrated', sm.mae_calibrated, true], ...D.models.map(m => [NAMES[m] || m, sm['mae_' + m], false])];
    const max = Math.max(...items.map(x => x[1] || 0));
    const bars = items.map(([n, v2, cal]) => `<div class="bar${cal ? ' cal' : ''}"><span>${n}</span>
      <span class="track"><span class="fill" style="width:${max ? (100 * v2 / max).toFixed(1) : 0}%"></span></span>
      <span class="v">${fmt(v2 == null ? null : v2 * scale, meta.snow ? meta.dec : 1)} ${meta.unit}</span></div>`).join('');
    const gain = sm.mae_calibrated && Math.min(...D.models.map(m => sm['mae_' + m])) ;
    const pct = gain ? Math.round(100 * (1 - sm.mae_calibrated / gain)) : null;
    return `<div class="score"><h3>${meta.label}</h3><div class="note">Average miss, ${sm.rows.toLocaleString()} forecasts
      ${pct != null ? `· <b>${pct}% smaller</b> than the best raw model` : ''}</div>${bars}</div>`;
  }).join('');
  $('scores').innerHTML = cards;
}

function renderEvents() {
  const s = D.stations[st.station], z = zone(s);
  const b = s.variables.hn24_cm;
  if (!b || !b.events || !b.events.length) { $('events').innerHTML = '<p class="empty">No snow events in this history.</p>'; return; }
  const conv = VARS.hn24_cm.conv;
  $('events').innerHTML = b.events.map((e, i) => {
    const vals = [['Observed', e.obs, 'obs'], ['Calibrated', e.q50, 'cal'], ...D.models.map(m => [NAMES[m] || m, e.raw[m], m])];
    const errs = vals.slice(1).map(v => v[1] == null ? Infinity : Math.abs(v[1] - e.obs));
    const best = errs.indexOf(Math.min(...errs)) + 1;
    return `<button class="event" data-i="${i}" aria-label="Replay ${label(e.valid * H, z)}">
      <div><div class="k">Ending ${label(e.valid * H, z, true)} ${z.name}</div><div class="d">${fmt(conv(e.obs), 1)} in observed</div></div>
      ${vals.slice(1).map((v, j) => `<div><div class="k">${v[0]}</div><div class="n${j + 1 === best ? ' best' : ''}">${fmt(v[1] == null ? null : conv(v[1]), 1)} in</div></div>`).join('')}
    </button>`;
  }).join('');
  $('events').querySelectorAll('button').forEach(btn => btn.onclick = () => {
    const e = b.events[+btn.dataset.i];
    st.variable = 'hn24_cm';
    const issueLocal = local(e.issue * H, z);
    st.issueHourLocal = issueLocal.getUTCHours() < 12 ? 8 : 20;
    st.day = issueLocal.toISOString().slice(0, 10);
    renderAll();
    $('replay').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
  });
}

function pickIssue(b, z) {
  const want = st.issueHourLocal;
  const onDay = b._issues.filter(t => dayKey(t, z) === st.day);
  const exact = onDay.find(t => local(t, z).getUTCHours() === want);
  return exact ?? onDay[0] ?? null;
}

function renderReplay() {
  const s = D.stations[st.station], z = zone(s);
  $('vars').innerHTML = Object.entries(VARS).filter(([v]) => s.variables[v]).map(([v, m]) =>
    `<button class="pill" aria-pressed="${v === st.variable}" data-v="${v}">${m.label}</button>`).join('');
  $('vars').querySelectorAll('button').forEach(b => b.onclick = () => { st.variable = b.dataset.v; renderReplay(); });
  $('when').querySelectorAll('button').forEach(b => { b.setAttribute('aria-pressed', +b.dataset.h === st.issueHourLocal); b.onclick = () => { st.issueHourLocal = +b.dataset.h; renderReplay(); }; });
  const b = rowsFor(st.station, st.variable);
  if (!b) { $('chart').innerHTML = ''; $('miss').innerHTML = ''; return; }
  const first = dayKey(b._issues[0], z), last = dayKey(b._issues[b._issues.length - 1], z);
  if (!st.day) st.day = last;
  st.day = st.day < first ? first : st.day > last ? last : st.day;
  const inp = $('day'); inp.min = first; inp.max = last; inp.value = st.day;
  const issue = pickIssue(b, z);
  const meta = VARS[st.variable];
  if (issue == null) { $('headline').textContent = `No forecast was issued on ${st.day}.`; $('chart').innerHTML = ''; $('miss').innerHTML = ''; return; }
  const rows = b._rows.filter(r => r.issue === issue).sort((a, c) => a.lead - c.lead);
  $('headline').textContent = `Forecast issued ${label(issue, z, true)} ${z.name}`;
  drawChart(rows, meta, z);
  drawTable(rows, meta, z);
}

function drawChart(rows, meta, z) {
  const W = 760, Hh = 260, pl = 48, pr = 14, pt = 22, pb = 30;
  const c = v => v == null ? null : meta.conv(v);
  const all = rows.flatMap(r => [c(r.q05), c(r.q95), c(r.obs), ...D.models.map(m => c(r.raw[m]))]).filter(v => v != null);
  if (!all.length) { $('chart').innerHTML = ''; return; }
  let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1e-6) hi = lo + 1;
  const pad = (hi - lo) * .08; lo = meta.snow ? Math.max(0, lo - pad) : lo - pad; hi += pad;
  const t0 = rows[0].valid, t1 = rows[rows.length - 1].valid === t0 ? t0 + H : rows[rows.length - 1].valid;
  const x = t => pl + (t - t0) / (t1 - t0) * (W - pl - pr), y = v => pt + (hi - v) / (hi - lo) * (Hh - pt - pb);
  const pts = (k) => rows.filter(r => c(r[k]) != null).map(r => `${x(r.valid).toFixed(1)},${y(c(r[k])).toFixed(1)}`);
  const band = (a, b2, cls) => { const f = pts(a), g = pts(b2).reverse(); return f.length ? `<polygon class="${cls}" points="${f.concat(g).join(' ')}"/>` : ''; };
  const line = (arr, cls) => arr.length > 1 ? `<polyline class="${cls}" points="${arr.join(' ')}"/>` : '';
  let g = '';
  const step = (hi - lo) / 4, dec = step >= 1 ? 0 : Math.min(2, Math.ceil(-Math.log10(step)));
  for (let i = 0; i < 5; i++) { const v = lo + step * i; g += `<line class="grid" x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}"/><text class="ax num" x="${pl - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(dec)}</text>`; }
  rows.forEach((r, i) => { if (i % Math.max(1, Math.ceil(rows.length / 8)) === 0) g += `<text class="ax" x="${x(r.valid)}" y="${Hh - 10}" text-anchor="middle">${label(r.valid, z, !meta.snow).replace(/, \d{4}/, '')}</text>`; });
  g += band('q05', 'q95', 'b90') + band('q25', 'q75', 'b50');
  for (const m of D.models) g += line(rows.filter(r => r.raw[m] != null).map(r => `${x(r.valid).toFixed(1)},${y(c(r.raw[m])).toFixed(1)}`), 'raw');
  g += line(pts('q50'), 'med');
  rows.filter(r => r.obs != null).forEach(r => { g += `<circle class="obsdot" cx="${x(r.valid)}" cy="${y(c(r.obs))}" r="${meta.snow ? 5 : 3}"/>`; });
  $('chart').innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Forecast against observations">${g}<text class="ax unit" x="${pl - 8}" y="12" text-anchor="end">${meta.unit}</text></svg>`;
}

function drawTable(rows, meta, z) {
  const c = v => v == null ? null : meta.conv(v);
  const shown = meta.snow ? rows : rows.filter((r, i) => i % 2 === 0);
  const head = `<tr><th>Valid (${z.name})</th><th>Observed</th><th>Calibrated</th>${D.models.map(m => `<th>${NAMES[m] || m}</th>`).join('')}</tr>`;
  let tot = {cal: 0}, n = 0; D.models.forEach(m => tot[m] = 0);
  const body = shown.map(r => {
    const obs = c(r.obs), vals = [['cal', c(r.q50)], ...D.models.map(m => [m, c(r.raw[m])])];
    const errs = vals.map(([, v]) => obs == null || v == null ? Infinity : Math.abs(v - obs));
    const best = errs.indexOf(Math.min(...errs));
    if (obs != null && vals.every(([, v]) => v != null)) { n++; vals.forEach(([k, v]) => tot[k] += Math.abs(v - obs)); }
    return `<tr><td>${label(r.valid, z, true)}</td><td>${fmt(obs, meta.dec)}</td>${vals.map(([k, v], i) =>
      `<td class="${i === best && obs != null ? 'best' : ''}">${fmt(v, meta.dec)}${obs != null && v != null ? ` <span class="note">(${v - obs >= 0 ? '+' : ''}${(v - obs).toFixed(meta.dec)})</span>` : ''}</td>`).join('')}</tr>`;
  }).join('');
  const foot = n ? `<tr><td><b>Average miss</b></td><td></td><td class="best"><b>${fmt(tot.cal / n, meta.dec)}</b></td>${D.models.map(m => `<td>${fmt(tot[m] / n, meta.dec)}</td>`).join('')}</tr>` : '';
  $('miss').innerHTML = `<div class="tablewrap"><table class="miss"><thead>${head}</thead><tbody>${body}${foot}</tbody></table></div>`;
}

function shiftDay(d) {
  const t = new Date(st.day + 'T12:00:00Z'); t.setUTCDate(t.getUTCDate() + d);
  st.day = t.toISOString().slice(0, 10); renderReplay();
}
$('prev').onclick = () => shiftDay(-1);
$('next').onclick = () => shiftDay(1);
$('day').onchange = e => { if (e.target.value) { st.day = e.target.value; renderReplay(); } };
function renderAll() { renderStations(); renderScores(); renderEvents(); renderReplay(); }
// Open on the biggest snow day, the reason the page exists, rather than on the last
// date, which is summer for most of the year.
(function openOnBiggestStorm() {
  const s = D.stations[st.station], e = s.variables.hn24_cm && s.variables.hn24_cm.events && s.variables.hn24_cm.events[0];
  if (!e) return;
  const z = zone(s), issueLocal = local(e.issue * H, z);
  st.issueHourLocal = issueLocal.getUTCHours() < 12 ? 8 : 20;
  st.day = issueLocal.toISOString().slice(0, 10);
})();
renderAll();
"""


def render(history: dict, *, title: str, lead: str) -> str:
    """The replay page as publishable content (title, fonts, style, body, inline data)."""
    data = json.dumps(history, separators=(",", ":")).replace("</", "<\\/")
    script = SCRIPT.replace("%NAMES%", json.dumps(MODEL_NAMES))
    body = f"""
<main class="page">
  <p class="kicker">Tree60 Enterprise · forecast replay</p>
  <h1>{html.escape(title)}</h1>
  <p class="lead">{html.escape(lead)}</p>
  <nav class="switch" id="stations" aria-label="Station"></nav>
  <section><div class="head"><h2>Over the whole replay</h2></div><div class="scores" id="scores"></div></section>
  <section><div class="head"><h2>Biggest snow days</h2>
    <p class="note">The largest observed 24 h snowfalls, each with the forecast issued the morning before. Tap one to replay it.</p></div>
    <div class="events" id="events"></div></section>
  <section id="replay"><div class="head"><h2 id="headline">Replay</h2></div>
    <nav class="switch" id="vars" aria-label="Variable"></nav>
    <div class="controls">
      <button class="step" id="prev" aria-label="Previous day">&#8592;</button>
      <input type="date" id="day" aria-label="Forecast date">
      <button class="step" id="next" aria-label="Next day">&#8594;</button>
      <span class="switch" id="when" style="margin:0">
        <button class="pill" data-h="8">Morning 08:00</button><button class="pill" data-h="20">Evening 20:00</button>
      </span>
    </div>
    <div class="chart" id="chart"></div>
    <div class="legend"><span><i class="med"></i>calibrated median</span><span><i class="b50"></i>50% range</span>
      <span><i class="b90"></i>90% range</span><span><i class="raw"></i>raw models</span><span>● observed</span></div>
    <div id="miss"></div>
  </section>
  <footer><p>Every calibrated forecast here came from a walk-forward replay: the station's model was
  refit every two weeks on what was known at the time and used to forecast the next weeks, so each is a
  forecast that could really have been issued that day. Snow totals are the 24 hours ending 08:00 local
  standard time. Tree60 Weather.</p></footer>
</main>
<script type="application/json" id="hist">{data}</script>
<script>{script}</script>"""
    return f"<title>{html.escape(title)}</title>{FONTS}<style>{STYLE}{EXTRA_STYLE}</style>{body}"
