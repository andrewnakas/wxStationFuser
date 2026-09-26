"""A standalone HTML report of one calibrated forecast (schema v2), to send as a file.

Self-contained on purpose (inline SVG, inline CSS, no scripts or external requests), so
it opens from an email attachment or a shared drive with nothing else. The wording
follows the Tree60 page (web/src/enterprise/service.ts). A gain is claimed only when
its 90% interval clears zero, and probabilities at the estimator's limit print as
bounds.
"""
from __future__ import annotations

import html
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


def _svg(times, block, raw_series, conv, unit, decimals) -> str:
    """A fan chart: 90% and 50% ranges, calibrated median, raw models dashed."""
    w, h, pl, pr, pt, pb = 760, 220, 48, 12, 10, 28
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

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" preserveAspectRatio="none">']
    for i in range(5):  # grid and y labels
        v = lo + (hi - lo) * i / 4
        parts.append(f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="ax" x="{pl - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.{decimals}f}</text>')
    day = times[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while day < times[-1]:  # midnight UTC ticks, labelled by date
        parts.append(f'<line class="grid" x1="{x(day):.1f}" x2="{x(day):.1f}" y1="{pt}" y2="{h - pb}"/>')
        parts.append(f'<text class="ax" x="{x(day):.1f}" y="{h - 8}" text-anchor="middle">{day:%a %d}</text>')
        day += timedelta(days=1)
    parts.append(band(q["q05"], q["q95"], "b90"))
    parts.append(band(q["q25"], q["q75"], "b50"))
    for arr in raws.values():
        parts.append(line(arr, "raw"))
    parts.append(line(q["q50"], "med"))
    parts.append(f'<text class="ax" x="4" y="{pt + 10}">{html.escape(unit)}</text></svg>')
    return "".join(parts)


def render(fc: dict, *, units: str = "imperial", now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    times = _times(fc)
    st = fc["station"]
    title = fc.get("spec", {}).get("label") or st["name"]
    elev = st.get("elev_m")
    elev_txt = "" if elev is None else (f" · {round(elev * 3.28084):,} ft" if units == "imperial" else f" · {round(elev):,} m")
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
            when = f" · {p24[1]:%a %H:%M} UTC" if p24 else ""
            tiles.append(
                f'<div class="tile"><div class="tl">&gt; {conv(float(thr)):.{d}f} {html.escape(unit)}</div>'
                f'<div class="tv">{fmt_prob(p24[0] if p24 else None)}</div>'
                f'<div class="tn">peak chance, next 24 h{when}</div>'
                f'<div class="tn">48 h: {fmt_prob(p48[0] if p48 else None)}</div></div>')
        raws = {m: v[var] for m, v in fc.get("raw", {}).items() if var in v}
        skill = fc.get("skill", {}).get(var) or {}
        method = fc.get("method", {}).get(var) or {}
        cov = ""
        if skill.get("status") == "verified" and skill.get("coverage90") is not None:
            cov = (f'<p class="note">The 90% range held {round(skill["coverage90"] * 100)}% of the time'
                   + (f', the 50% range {round(skill["coverage50"] * 100)}%' if skill.get("coverage50") is not None else "")
                   + (f'. Method: {html.escape(method.get("label", ""))}, {round(method.get("train_days", 0))} days of history.' if method else ".")
                   + "</p>")
        sections.append(
            f'<section><h2>{html.escape(label)}</h2>'
            + ('<p class="note">Each hour shows the total for the 24 hours ending then.</p>' if window else "")
            + (f'<div class="tiles">{"".join(tiles)}</div>' if tiles else "")
            + f'<div class="chart">{_svg(times, block, raws, conv, unit, d)}</div>'
            + '<div class="legend"><span class="k med"></span>calibrated median <span class="k b50"></span>50% range '
              '<span class="k b90"></span>90% range <span class="k raw"></span>raw models</div>'
            + f'<p class="skill">{html.escape(skill_sentence(skill))}</p>{cov}</section>')

    issued = datetime.fromisoformat(fc["generated_at"].replace("Z", "+00:00"))
    obs = fc.get("obs_latest")
    stale = ""
    if obs:
        age = (issued - datetime.fromisoformat(obs.replace("Z", "+00:00"))).total_seconds() / 3600
        if age > 3:
            stale = (f'<p class="warn">The newest station observation is {round(age)} h old; the first '
                     f'hours lean on the models alone.</p>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}: calibrated forecast</title><style>
:root{{--ink:#17202a;--soft:#5b6673;--line:#d9dee4;--bg:#fbfbfa;--card:#fff;--brand:#1f6fb2;--warn:#b7791f}}
@media (prefers-color-scheme:dark){{:root{{--ink:#e8ecf0;--soft:#a3adb8;--line:#2f3943;--bg:#12171c;--card:#1a2129;--brand:#5aa7e8;--warn:#e0a84a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:820px;margin:0 auto;padding:24px 16px 48px}}
h1{{margin:0 0 4px;font-size:26px}} h2{{margin:28px 0 8px;font-size:18px}}
.lead,.note,.tn,.tl,.legend,footer{{color:var(--soft);font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin:10px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.tv{{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}}
.chart svg{{width:100%;height:220px;display:block}}
.grid{{stroke:var(--line);stroke-width:1}} .ax{{fill:var(--soft);font-size:11px}}
.b90{{fill:var(--brand);opacity:.16}} .b50{{fill:var(--brand);opacity:.30}}
.med{{fill:none;stroke:var(--brand);stroke-width:2.5}} .raw{{fill:none;stroke:var(--soft);stroke-width:1;stroke-dasharray:3 3}}
.legend .k{{display:inline-block;width:14px;height:8px;margin:0 4px 0 10px;vertical-align:middle;border-radius:2px}}
.legend .k.med{{background:var(--brand)}} .legend .k.b50{{background:var(--brand);opacity:.45}}
.legend .k.b90{{background:var(--brand);opacity:.2}} .legend .k.raw{{border-top:1px dashed var(--soft);height:0}}
.skill{{margin:10px 0 2px}} .warn{{border:1px solid var(--warn);border-radius:8px;padding:8px 12px}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="lead">{html.escape(st["name"])}{elev_txt} · {html.escape(" + ".join(model_name(m) for m in fc.get("models_used", [])))}, calibrated to this station against its own history</p>
{stale}{"".join(sections)}
<footer><p>Issued {issued:%Y-%m-%d %H:%M} UTC. Times on the charts are UTC. Shaded ranges should hold the observation half the time (dark) and nine times in ten (light); each section reports how often they actually did in walk-forward testing. Tree60 Weather.</p></footer>
</main></body></html>"""


def write_report(fc: dict, path: str | Path, *, units: str = "imperial") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(fc, units=units), encoding="utf-8")
    return p
