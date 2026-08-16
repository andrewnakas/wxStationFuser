"""Command line entry point — the interface the GitHub Actions workflows drive."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wxfuser.data.registry import Station, load_registry, upsert
from wxfuser.pipeline import core, emit


def _station_or_exit(station_id: str) -> Station:
    from wxfuser.data.registry import find

    station = find(station_id)
    if station is None:
        sys.exit(f"station {station_id!r} is not in stations.yaml")
    return station


def cmd_bootstrap(args) -> int:
    """Pull deep history for a station and train it from scratch."""
    station = _station_or_exit(args.station)
    entry = core.run_station(station, bootstrap=True, years=args.years)
    print(f"bootstrap complete: {entry.get('status')}")
    return 0 if entry.get("status") == "ok" else 1


def _stage_catalogue() -> None:
    """Copy the station catalogue from state into the published site, if we have one.

    The catalogue is rebuilt weekly by its own workflow, but the site is republished
    every few hours from a fresh checkout. Without this the search box would 404 on
    every deploy that did not immediately follow a catalogue build, leaving the site
    able to show enrolled stations only.
    """
    import shutil

    src = core.STATE_DIR / "catalogue" / "stations.min.json"
    dst = core.SITE_DIR / "stations.min.json"
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"staged catalogue from {src} ({src.stat().st_size // 1024} KB)")
    elif not src.exists() and not dst.exists():
        print("no station catalogue available; search will cover enrolled stations only")


def cmd_refresh(args) -> int:
    """Update every enrolled station and rewrite the site JSON."""
    stations = load_registry()
    _stage_catalogue()
    if args.station:
        stations = [s for s in stations if s.id == args.station]
    if not stations:
        print("no stations enrolled; nothing to do")
        emit.write_json(emit.index_json([]), core.SITE_DIR / "index.json")
        return 0

    entries = []
    failures = 0
    for station in stations:
        try:
            entries.append(core.run_station(station, bootstrap=args.bootstrap))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{station.id}] FAILED: {exc}", flush=True)
            entries.append({"id": station.id, "name": station.name, "lat": station.lat,
                            "lon": station.lon, "status": "error"})

    emit.write_json(emit.index_json(entries), core.SITE_DIR / "index.json")
    ok = sum(1 for e in entries if e.get("status") == "ok")
    print(f"refresh complete: {ok}/{len(entries)} stations published, {failures} errors")
    # A partial refresh still deploys: a stale station is better than an empty site.
    return 0


def cmd_enroll(args) -> int:
    """Add a station to the registry and bootstrap it."""
    station = Station(
        id=args.station,
        name=args.name or args.station,
        lat=args.lat,
        lon=args.lon,
        elev_m=args.elev,
        models=args.models.split(",") if args.models else [],
        variables=args.variables.split(",") if args.variables else [],
        iem_network=args.iem_network,
        ghcnh_id=args.ghcnh_id,
        nws_id=args.nws_id,
        enrolled_at=__import__("datetime").date.today().isoformat(),
    )
    upsert(station)
    print(f"enrolled {station.id} ({station.name})")
    if args.no_bootstrap:
        return 0
    entry = core.run_station(station, bootstrap=True, years=args.years)
    return 0 if entry.get("status") == "ok" else 1


def cmd_retrain(args) -> int:
    """Re-run walk-forward verification and re-decide the champion for every station.

    Expensive (dozens of refits per variable), so it runs weekly rather than on the
    forecast cadence.
    """
    stations = load_registry()
    if args.station:
        stations = [s for s in stations if s.id == args.station]
    if not stations:
        print("no stations enrolled")
        return 0

    entries = []
    for station in stations:
        try:
            entries.append(core.run_station(station, evaluate=True))
        except Exception as exc:  # noqa: BLE001
            print(f"[{station.id}] FAILED: {exc}", flush=True)
            entries.append({"id": station.id, "name": station.name, "lat": station.lat,
                            "lon": station.lon, "status": "error"})
    emit.write_json(emit.index_json(entries), core.SITE_DIR / "index.json")
    ok = sum(1 for e in entries if e.get("status") == "ok")
    print(f"retrain complete: {ok}/{len(entries)} stations evaluated")
    return 0


def cmd_enroll_bulk(args) -> int:
    """Enroll many stations at once from the bulk sources.

    Writes registry entries and, unless told not to, bootstraps them in shard-sized
    batches so one long-running invocation can be interrupted without losing the
    stations already trained.
    """
    from wxfuser.data import bulk
    from wxfuser.data.registry import load_registry, save_registry
    from wxfuser.pipeline import bulk_run

    universe = bulk.enrollable_universe(
        include_asos=not args.no_asos,
        include_snotel=not args.no_snotel,
        min_elevation_m=args.min_elevation,
        countries=args.countries.split(",") if args.countries else None,
    )
    if universe.empty:
        print("no stations matched")
        return 1

    if args.limit:
        # Highest first, so a capped run selects the complex terrain where the correction
        # has the most to do rather than an arbitrary alphabetical slice.
        universe = universe.sort_values("elev_m", ascending=False, na_position="last")
        universe = universe.head(args.limit)

    existing = {s.id for s in load_registry()}
    today = __import__("datetime").date.today().isoformat()
    added = []
    for _, r in universe.iterrows():
        if r["id"] in existing:
            continue
        added.append(
            Station(
                id=str(r["id"]),
                name=str(r["name"])[:80],
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                elev_m=float(r["elev_m"]) if pd_notna(r.get("elev_m")) else None,
                country=str(r["country"]) if pd_notna(r.get("country")) else None,
                enrolled_at=today,
            )
        )

    print(f"\n{len(added)} new stations to enroll ({len(existing)} already registered)")
    if args.dry_run:
        for s in added[:10]:
            print(f"  {s.id:24s} {s.name[:40]:42s} {s.elev_m or 0:6.0f} m")
        if len(added) > 10:
            print(f"  … and {len(added) - 10} more")
        return 0

    registry = load_registry() + added
    registry.sort(key=lambda s: s.id)
    save_registry(registry)
    print(f"registry now holds {len(registry)} stations")

    if args.no_bootstrap:
        return 0

    batch = args.batch
    total_ok = 0
    for i in range(0, len(added), batch):
        chunk = added[i : i + batch]
        print(f"\n=== bootstrapping {i + 1}..{i + len(chunk)} of {len(added)} ===")
        entries = bulk_run.run_stations(
            chunk, bootstrap=True, evaluate=True, years=args.years
        )
        total_ok += sum(1 for e in entries if e.get("status") == "ok")
        print(f"  cumulative published: {total_ok}")
    print(f"\nbulk enroll complete: {total_ok}/{len(added)} published")
    return 0


def pd_notna(v) -> bool:
    import pandas as pd

    return v is not None and not pd.isna(v)


def cmd_catalogue(args) -> int:
    """Rebuild the global station catalogue the site's search box reads."""
    import json

    from wxfuser.data import catalogue

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    df = catalogue.build(sources=sources, reporting_only=not args.include_non_reporting)
    if df.empty:
        print("catalogue build produced no stations")
        return 1

    out_dir = core.STATE_DIR / "catalogue"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "stations.parquet", index=False, compression="zstd")

    payload = catalogue.to_min_json(df)
    site_path = core.SITE_DIR / "stations.min.json"
    site_path.parent.mkdir(parents=True, exist_ok=True)
    with open(site_path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    # Also keep it in state so later refreshes, which run from a clean checkout, can
    # republish the catalogue without rebuilding it.
    with open(out_dir / "stations.min.json", "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    size_kb = site_path.stat().st_size / 1024
    by_net = df["network"].value_counts().to_dict()
    print(f"catalogue: {len(df)} stations {by_net} -> {site_path} ({size_kb:.0f} KB)")
    return 0


def cmd_list(args) -> int:
    stations = load_registry()
    if not stations:
        print("no stations enrolled")
        return 0
    for s in stations:
        models = ",".join(s.resolved_models())
        print(f"{s.id:<24} {s.name:<32} {s.lat:8.4f} {s.lon:9.4f}  {models}")
    return 0


def cmd_verify(args) -> int:
    """Print the verification scorecard for one station without republishing."""
    import json

    station = _station_or_exit(args.station)
    path = core.SITE_DIR / "stations" / station.slug / "verify.json"
    if not path.exists():
        sys.exit(f"no verification found at {path}; run refresh first")
    print(json.dumps(json.loads(Path(path).read_text()), indent=2)[:8000])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wxfuser", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enroll", help="add a station and train it")
    p.add_argument("station", help="station id, e.g. IEM:DEN or NWS:KDEN")
    p.add_argument("--name")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--elev", type=float)
    p.add_argument("--models", help="comma-separated Open-Meteo model ids")
    p.add_argument("--variables", help="comma-separated canonical variables")
    p.add_argument("--iem-network")
    p.add_argument("--ghcnh-id")
    p.add_argument("--nws-id")
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--no-bootstrap", action="store_true")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("bootstrap", help="pull deep history and train one station")
    p.add_argument("station")
    p.add_argument("--years", type=float, default=2.0)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("refresh", help="update enrolled stations and write site JSON")
    p.add_argument("--station", help="limit to one station")
    p.add_argument("--bootstrap", action="store_true", help="deep pull instead of incremental")
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("retrain", help="re-run verification and re-select champions")
    p.add_argument("--station")
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("enroll-bulk", help="enroll many stations from the bulk sources")
    p.add_argument("--limit", type=int, help="cap the number enrolled, highest elevation first")
    p.add_argument("--min-elevation", type=float,
                   help="only stations at or above this elevation in metres")
    p.add_argument("--countries", help="comma-separated ISO country codes")
    p.add_argument("--no-asos", action="store_true", help="skip the ASOS airport archive")
    p.add_argument("--no-snotel", action="store_true", help="skip SNOTEL mountain sites")
    p.add_argument("--years", type=float, default=2.0, help="years of history to backfill")
    p.add_argument("--batch", type=int, default=200,
                   help="stations bootstrapped per batch; smaller batches checkpoint sooner")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="register them but leave training to the scheduled jobs")
    p.add_argument("--dry-run", action="store_true", help="show what would be enrolled")
    p.set_defaults(func=cmd_enroll_bulk)

    p = sub.add_parser("catalogue", help="rebuild the global station catalogue")
    p.add_argument("--sources", default="nws,iem,meteostat")
    p.add_argument(
        "--include-non-reporting",
        action="store_true",
        help="keep NWS river/precip gauges that do not serve hourly weather observations",
    )
    p.set_defaults(func=cmd_catalogue)

    p = sub.add_parser("list", help="list enrolled stations")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("verify", help="show a station's verification scorecard")
    p.add_argument("station")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
