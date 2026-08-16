"""Write the comment posted back on a successful enrollment.

The point is to answer the question the person actually asked — "is this any good at my
station?" — rather than to announce that a job finished. So it leads with the measured
skill against the raw model, including the cases where there is no measurable gain yet,
and links to the page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIABLE_NAMES = {
    "air_temp_c": "Temperature",
    "rh_pct": "Relative humidity",
    "wind_speed_ms": "Wind speed",
    "wind_gust_ms": "Wind gusts",
    "precip_1h_mm": "Precipitation",
}


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--site-dir", default="site")
    args = ap.parse_args()

    owner, repo = args.repo.split("/", 1)
    url = f"https://{owner}.github.io/{repo}/?station="

    forecast_path = Path(args.site_dir) / "stations" / args.slug / "forecast.json"
    if not forecast_path.exists():
        print(
            f"**{args.name}** is enrolled, but no forecast was published yet — most "
            f"likely its observation history is too short or too sparse to calibrate "
            f"against. The scheduled refresh will keep trying as more observations "
            f"arrive."
        )
        return 0

    payload = json.loads(forecast_path.read_text())
    station = payload.get("station", {})
    skill = payload.get("skill", {})
    method = payload.get("method", {})

    lines = [
        f"### {args.name} is live",
        "",
        f"[View the forecast]({url}{station.get('id', '')}) · "
        f"fusing {', '.join(payload.get('models_used', []))}",
        "",
        "| Variable | Method | vs best raw model | 90% CI | Verdict |",
        "|---|---|---|---|---|",
    ]

    any_verified = False
    for var, sk in skill.items():
        label = VARIABLE_NAMES.get(var, var)
        meth = (method.get(var) or {}).get("label", "—")
        if sk.get("status") != "verified":
            lines.append(f"| {label} | {meth} | — | — | still warming up |")
            continue
        any_verified = True
        ci = sk.get("crpss_vs_raw_ci90") or [None, None]
        verdict = (
            "beats it"
            if sk.get("beats_raw")
            else "better, not conclusively"
            if (sk.get("crpss_vs_raw") or 0) > 0
            else "no measurable gain"
        )
        lines.append(
            f"| {label} | {meth} | {pct(sk.get('crpss_vs_raw'))} "
            f"| {pct(ci[0])} to {pct(ci[1])} | {verdict} |"
        )

    lines += [
        "",
        "Skill is measured as CRPS against the strongest raw model on the same forecast "
        "hours, evaluated walk-forward so no fit ever scores data it was trained on. "
        "A verdict of *beats it* means the 90% confidence interval on the gain lies "
        "entirely above zero.",
    ]
    if not any_verified:
        lines.append(
            "\nNothing is verified yet — that is expected for a new station and will "
            "resolve as paired forecasts and observations accumulate."
        )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
