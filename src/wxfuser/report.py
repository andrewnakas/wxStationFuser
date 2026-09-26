"""A standalone HTML report of one calibrated forecast (schema v2), to send as a file.

Self-contained on purpose (inline SVG, inline CSS, no scripts or external requests), so
it opens from an email attachment or a shared drive with nothing else. The wording
follows the Tree60 page (web/src/enterprise/service.ts). A gain is claimed only when
its 90% interval clears zero, and probabilities at the estimator's limit print as
bounds.
"""
from __future__ import annotations

import html
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

VARS = {
    # id: (label, imperial unit, to_imperial, metric unit, decimals imperial, 24 h amount)
    "air_temp_c": ("Temperature", "°F", lambda v: v * 9 / 5 + 32, "°C", 0, False),
    "rh_pct": ("Humidity", "%", lambda v: v, "%", 0, False),
    "wind_speed_ms": ("Wind", "mph", lambda v: v * 2.236936, "m/s", 0, False),
    "wind_gust_ms": ("Gusts", "mph", lambda v: v * 2.236936, "m/s", 0, False),
    "precip_1h_mm": ("Precipitation (1 h)", "in", lambda v: v / 25.4, "mm", 2, False),
    "swe_24h_mm": ("SWE, 24 h", "in", lambda v: v / 25.4, "mm", 2, True),
    "hn24_cm": ("New snow, 24 h", "in", lambda v: v / 2.54, "cm", 0, True),
}


MODEL_NAMES = {"hrrr": "HRRR", "gfs16": "GFS", "gefs": "GEFS", "ecmwf_ens": "ECMWF ENS",
               "aifs_single": "ECMWF AIFS"}


def model_name(m: str) -> str:
    return MODEL_NAMES.get(m, m.replace("_", " ").upper())


def fmt_prob(p) -> str:
    if p is None:
        return "—"
    if p <= 0.01:
        return "≤1%"
    if p >= 0.99:
        return "≥99%"
    return f"{round(p * 100)}%"


def skill_sentence(skill: dict | None) -> str:
    if skill and skill.get("reason"):
        return f"Not yet verified: {skill['reason']}."
    if not skill or skill.get("status") != "verified" or skill.get("crpss_vs_raw") is None:
        return "Not yet verified: too little history to measure skill honestly."
    pct = round(skill["crpss_vs_raw"] * 100)
    base = model_name(skill["raw_best_model"]) if skill.get("raw_best_model") else "the best raw model"
    lo, hi = (skill.get("crpss_vs_raw_ci90") or [None, None])[:2]
    ci = f" (90% interval {round(lo * 100)}–{round(hi * 100)}%)" if lo is not None and hi is not None else ""
    if skill.get("beats_raw"):
        return f"{pct}% more accurate than {base} alone{ci}, measured on forecasts made before the data was seen."
    if pct < 0:
        return f"No better than {base} alone yet{ci}. Use the raw model alongside it."
    return f"Measured {pct}% vs {base}{ci}, but not yet distinguishable from no gain."


def _times(fc: dict) -> list[datetime]:
    return [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in fc["hourly"]["time"]]


def _peak(times, probs, hours, now):
    best = None
    for t, p in zip(times, probs or []):
        if p is None or t < now - timedelta(hours=1) or t > now + timedelta(hours=hours):
            continue
        if best is None or p > best[0]:
            best = (p, t)
    return best


def _svg(times, block, raw_series, conv, unit, decimals, off: timedelta) -> str:
    """A fan chart: 90% and 50% ranges, calibrated median, raw models dashed.

    Day ticks fall on local-standard-time midnights, the clock a forecaster briefs by.
    """
    w, h, pl, pr, pt, pb = 760, 236, 50, 14, 24, 30

    def series(key):
        return [None if v is None else conv(v) for v in (block.get(key) or [None] * len(times))]
    q = {k: series(k) for k in ("q05", "q25", "q50", "q75", "q95")}
    raws = {m: [None if v is None else conv(v) for v in vals] for m, vals in raw_series.items()}
    vals = [v for arr in [*q.values(), *raws.values()] for v in arr if v is not None]
    if not vals or len(times) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    pad = (hi - lo) * 0.08
    # Amounts and speeds cannot go negative, so neither does their axis.
    lo, hi = (lo - pad if lo < 0 else max(0.0, lo - pad)), hi + pad
    t0, t1 = times[0].timestamp(), times[-1].timestamp()
    x = lambda t: pl + (t.timestamp() - t0) / (t1 - t0) * (w - pl - pr)  # noqa: E731
    y = lambda v: pt + (hi - v) / (hi - lo) * (h - pt - pb)  # noqa: E731

    def band(a, b, cls):
        pts = [(x(t), y(v)) for t, v in zip(times, a) if v is not None]
        back = [(x(t), y(v)) for t, v in zip(times, b) if v is not None][::-1]
        if not pts or not back:
            return ""
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts + back)
        return f'<polygon class="{cls}" points="{d}"/>'

    def line(arr, cls):
        segs, cur = [], []
        for t, v in zip(times, arr):
            if v is None:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{x(t):.1f},{y(v):.1f}")
        if len(cur) > 1:
            segs.append(cur)
        return "".join(f'<polyline class="{cls}" points="{" ".join(s)}"/>' for s in segs)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Forecast range" preserveAspectRatio="none">']
    # Enough decimals that neighbouring labels differ: a 0-0.02 in axis at two decimals
    # read "0.00, 0.01, 0.01, 0.02".
    step = (hi - lo) / 4
    tick_dec = max(decimals, 0 if step >= 1 else min(3, int(-math.floor(math.log10(step)))))
    for i in range(5):
        v = lo + step * i
        parts.append(f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="ax num" x="{pl - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.{tick_dec}f}</text>')
    local0 = times[0] + off
    day = local0.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1) - off
    while day < times[-1]:
        parts.append(f'<line class="grid day" x1="{x(day):.1f}" x2="{x(day):.1f}" y1="{pt}" y2="{h - pb}"/>')
        if x(day) + 30 < w - pr:
            parts.append(f'<text class="ax" x="{x(day) + 5:.1f}" y="{h - 10}">{(day + off):%a %-d}</text>')
        day += timedelta(days=1)
    parts.append(band(q["q05"], q["q95"], "b90"))
    parts.append(band(q["q25"], q["q75"], "b50"))
    for arr in raws.values():
        parts.append(line(arr, "raw"))
    parts.append(line(q["q50"], "med"))
    parts.append(f'<text class="ax unit" x="{pl - 8}" y="12" text-anchor="end">{html.escape(unit)}</text></svg>')
    return "".join(parts)


def _zone(lon: float | None) -> tuple[timedelta, str]:
    """Local standard time from longitude, named where it is a US zone."""
    hours = int(round((lon or 0.0) / 15.0))
    names = {-5: "EST", -6: "CST", -7: "MST", -8: "PST", -9: "AKST", -10: "HST"}
    return timedelta(hours=hours), names.get(hours, f"UTC{hours:+d}")


STYLE = """
:root{--ground:#F4F7F8;--panel:#FFFFFF;--ink:#172430;--soft:#56657A;--rule:#D3DCE1;
--accent:#1D5F8C;--accent-ink:#FFFFFF;--warn:#C2701A;--warn-ground:#FBF1E4;
--display:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;
--body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--ground:#0E151B;
--panel:#15202A;--ink:#E4ECF1;--soft:#93A3B3;--rule:#26343F;--accent:#6FB0DD;--accent-ink:#0E151B;
--warn:#E3A04E;--warn-ground:#2A2015}}
:root[data-theme="dark"]{color-scheme:dark;--ground:#0E151B;--panel:#15202A;--ink:#E4ECF1;--soft:#93A3B3;
--rule:#26343F;--accent:#6FB0DD;--accent-ink:#0E151B;--warn:#E3A04E;--warn-ground:#2A2015}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.55 var(--body)}
.page{max-width:860px;margin:0 auto;padding-inline:20px;padding-block:28px 56px}
.kicker{font:600 12px/1 var(--body);letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font:600 44px/1 var(--display);letter-spacing:.01em;margin:0;text-wrap:balance}
h2{font:600 26px/1.1 var(--display);letter-spacing:.02em;margin:0;text-wrap:balance}
.strip{display:flex;flex-wrap:wrap;gap:6px 22px;margin:16px 0 0;padding:12px 0;border-block:1px solid var(--rule);
font:13px/1.4 var(--mono);color:var(--soft)}
.strip b{font-weight:500;color:var(--ink)}
.warn{margin:18px 0 0;padding:10px 14px;background:var(--warn-ground);border-left:3px solid var(--warn);color:var(--ink);font-size:14px}
section{margin-top:40px;display:grid;gap:14px}
.head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:4px 16px}
.head .note{margin:0}
.note,.legend,footer{color:var(--soft);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;display:grid;gap:2px}
.tl{font:500 13px/1.3 var(--mono);color:var(--soft)}
.tv{font:600 34px/1.05 var(--display);font-variant-numeric:tabular-nums}
.tile.hot .tv{color:var(--warn)}
.tn{font-size:12.5px;color:var(--soft)}
.chart{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:10px 8px 4px}
.chart svg{width:100%;height:236px;display:block}
.grid{stroke:var(--rule);stroke-width:1}.grid.day{stroke-dasharray:2 4}
.ax{fill:var(--soft);font:11px var(--body)}.ax.num,.ax.unit{font-family:var(--mono)}
.b90{fill:var(--accent);fill-opacity:.14}.b50{fill:var(--accent);fill-opacity:.3}
.med{fill:none;stroke:var(--accent);stroke-width:2.5;stroke-linejoin:round}
.raw{fill:none;stroke:var(--soft);stroke-width:1.2;stroke-dasharray:3 3}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:16px;height:9px;border-radius:2px}
.legend i.med{height:3px;background:var(--accent)}.legend i.b50{background:var(--accent);opacity:.45}
.legend i.b90{background:var(--accent);opacity:.18}.legend i.raw{height:0;border-top:1.5px dashed var(--soft)}
.skill{margin:0;font-size:15px;max-width:68ch}
.skill.claim{font-weight:500}
.method{margin:0;max-width:68ch}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--rule);max-width:72ch}
@media (max-width:520px){h1{font-size:34px}.tv{font-size:30px}.chart svg{height:190px}}
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600'
         '&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">')


def render(fc: dict, *, units: str = "imperial", now: datetime | None = None,
           fragment: bool = False) -> str:
    """The report as a full document, or with ``fragment`` as page content only.

    The full document is what gets emailed, so it makes no external requests: system
    fonts stand in for the web fonts, which only the fragment (published as a page)
    loads.
    """
    now = now or datetime.now(UTC)
    times = _times(fc)
    st = fc["station"]
    off, zone = _zone(st.get("lon"))
    local = lambda t: t + off  # noqa: E731
    title = fc.get("spec", {}).get("label") or st["name"]
    elev = st.get("elev_m")
    sections = []
    for var, (label, iu, to_i, mu, dec, window) in VARS.items():
        block = fc["hourly"].get(var)
        if not block:
            continue
        conv = to_i if units == "imperial" else (lambda v: v)
        unit = iu if units == "imperial" else mu
        d = dec if units == "imperial" else (1 if var in ("precip_1h_mm", "swe_24h_mm") else 0)
        tiles = []
        for thr, probs in (block.get("p_exceed") or {}).items():
            p24, p48 = _peak(times, probs, 24, now), _peak(times, probs, 48, now)
            when = f" · {local(p24[1]):%a %H:%M} {zone}" if p24 else ""
            # Amber marks a hazard likely to be crossed: gusts, snow, heavy precipitation.
            # Above-freezing temperature is not one, so it stays in ink.
            hot = " hot" if p24 and p24[0] >= 0.3 and var != "air_temp_c" else ""
            tiles.append(
                f'<div class="tile{hot}"><div class="tl">&gt; {conv(float(thr)):.{d}f} {html.escape(unit)}</div>'
                f'<div class="tv">{fmt_prob(p24[0] if p24 else None)}</div>'
                f'<div class="tn">peak chance, next 24 h{when}</div>'
                f'<div class="tn">next 48 h: {fmt_prob(p48[0] if p48 else None)}</div></div>')
        raws = {m: v[var] for m, v in fc.get("raw", {}).items() if var in v}
        skill = fc.get("skill", {}).get(var) or {}
        method = fc.get("method", {}).get(var) or {}
        cov = ""
        if skill.get("status") == "verified" and skill.get("coverage90") is not None:
            cov = (f'<p class="note method">The 90% range held {round(skill["coverage90"] * 100)}% of the time'
                   + (f', the 50% range {round(skill["coverage50"] * 100)}%' if skill.get("coverage50") is not None else "")
                   + (f'. {html.escape(method.get("label", ""))}, trained on {round(method.get("train_days", 0))} days.' if method else ".")
                   + "</p>")
        claim = " claim" if skill.get("beats_raw") else ""
        sections.append(
            f'<section><div class="head"><h2>{html.escape(label)}</h2>'
            + ('<p class="note">Each hour: the total for the 24 hours ending then</p>' if window else "")
            + "</div>"
            + (f'<div class="tiles">{"".join(tiles)}</div>' if tiles else "")
            + f'<div class="chart">{_svg(times, block, raws, conv, unit, d, off)}</div>'
            + '<div class="legend"><span><i class="med"></i>calibrated median</span>'
              '<span><i class="b50"></i>50% range</span><span><i class="b90"></i>90% range</span>'
              '<span><i class="raw"></i>raw models</span></div>'
            + f'<p class="skill{claim}">{html.escape(skill_sentence(skill))}</p>{cov}</section>')

    issued = datetime.fromisoformat(fc["generated_at"].replace("Z", "+00:00"))
    obs = fc.get("obs_latest")
    stale, obs_txt = "", "none"
    if obs:
        obs_t = datetime.fromisoformat(obs.replace("Z", "+00:00"))
        age = (issued - obs_t).total_seconds() / 3600
        obs_txt = f"{local(obs_t):%a %H:%M} {zone} ({max(0, round(age))} h before issue)"
        if age > 3:
            stale = (f'<p class="warn">The newest station observation is {round(age)} h old, so the '
                     f'first hours lean on the models alone until fresh data arrives.</p>')
    elev_txt = (f"{round(elev * 3.28084):,} ft" if units == "imperial" else f"{round(elev):,} m") if elev else "—"
    strip = (f'<div class="strip"><span>station <b>{html.escape(str(st.get("id", "")))}</b></span>'
             f'<span>elevation <b>{elev_txt}</b></span>'
             + (f'<span><b>{st["lat"]:.4f}, {st["lon"]:.4f}</b></span>' if st.get("lat") is not None else "")
             + f'<span>models <b>{html.escape(" + ".join(model_name(m) for m in fc.get("models_used", [])))}</b></span>'
             f'<span>issued <b>{local(issued):%a %d %b %H:%M} {zone}</b></span>'
             f'<span>latest obs <b>{obs_txt}</b></span></div>')
    body = (f'<main class="page"><p class="kicker">Calibrated station forecast</p>'
            f'<h1>{html.escape(title)}</h1>{strip}{stale}{"".join(sections)}'
            f'<footer><p>Times are {zone}, local standard time, all year. Each forecast is the '
            f'models above corrected against this station\'s own history. The shaded ranges should '
            f'hold the observation half the time (dark) and nine times in ten (light), and each '
            f'section reports how often they did on forecasts made before the data was seen. '
            f'Tree60 Weather.</p></footer></main>')
    page_title = f"{html.escape(title)} forecast"
    if fragment:
        return f"<title>{page_title}</title>{FONTS}<style>{STYLE}</style>{body}"
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{page_title}</title><style>{STYLE}</style></head><body>{body}</body></html>")


def write_report(fc: dict, path: str | Path, *, units: str = "imperial", fragment: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(fc, units=units, fragment=fragment), encoding="utf-8")
    return p
