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


def shard_of(station_id: str, of: int) -> int:
    """Which worker owns a station, derived from its id alone.

    Deliberately not positional. An index-based split reassigns almost every station the
    moment the registry grows, which matters in two ways: a run already in flight holds a
    station list from startup while its checkpoint upload recomputes the split from the
    registry on disk, so the two disagree and a worker uploads paths it never processed;
    and re-running after adding stations reshuffles the work rather than resuming it.

    Hashing the id keeps a station on the same worker for life, so the registry can grow
    at any time. md5 rather than hash() because the latter is randomised per process.
    """
    import hashlib

    digest = hashlib.md5(station_id.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") % of


def select_shard(stations: list, shard: int | None, of: int | None) -> list:
    """The slice of the registry this worker is responsible for.

    Assignment spreads networks and regions across workers because it depends on the id's
    hash rather than its position, so no worker ends up holding all the SNOTEL sites —
    which would run far longer than the rest, having no bulk observation source.
    """
    if not of or of <= 1:
        return stations
    shard = shard or 0
    picked = [s for s in stations if shard_of(s.id, of) == shard]
    print(f"shard {shard + 1}/{of}: {len(picked)} of {len(stations)} stations", flush=True)
    return picked


def filter_sources(stations: list, spec: str | None) -> list:
    """Restrict a run to particular networks.

    Forecast requests are the scarce resource — Open-Meteo prices them by locations x
    models and throttles hard — so it is worth being able to spend them only where an
    observation can actually arrive. A network publishing months in arrears returns
    nothing for an incremental window, and every forecast fetched for it is budget taken
    from a station that would have produced a verifiable pair.

    Filtering does not disturb the split: ``shard_of`` reads the id alone, so a station
    keeps its worker whether or not its network was selected.
    """
    if not spec:
        return stations
    from wxfuser.pipeline.bulk_run import _source_of

    wanted = {s.strip().upper() for s in spec.split(",") if s.strip()}
    picked = [s for s in stations if _source_of(s.id) in wanted]
    print(f"sources {sorted(wanted)}: {len(picked)} of {len(stations)} stations", flush=True)
    return picked


def order_stations(stations: list, order: str | None) -> list:
    """Put the stations a run cares most about first.

    Order matters because a run is not guaranteed to finish. Shards are capped at the
    Actions job limit and the forecast API throttles, so the tail of a long shard is the
    part most likely to be dropped — and with registry order that tail is an arbitrary
    slice of the alphabet. Sorting by served population means an interrupted run has
    spent its budget on the stations people actually look up.

    Falls back to the given order if the gazetteer cannot be fetched. A ranking is an
    optimisation; failing to fetch one is not a reason to refresh nothing.
    """
    if not order or order == "registry" or len(stations) < 2:
        return stations
    if order != "prominence":
        raise SystemExit(f"unknown --order {order!r}")
    from wxfuser.data import prominence

    try:
        ranked = prominence.sort_stations(stations)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: prominence ranking unavailable ({exc}); keeping registry order",
              flush=True)
        return stations
    head = ", ".join(s.id for s in ranked[:3])
    print(f"ordered by served population: {head} first", flush=True)
    return ranked


def only_untrained(stations: list, *, use_hub: bool = False) -> list:
    """Drop stations that already have a paired archive.

    What makes a scheduled bootstrap possible. The registry holds far more stations than
    one run can backfill, so a nightly job has to be able to ask "what is still
    outstanding" and work only on that; without it every run would start again at the
    beginning of the registry and never reach the end of it.
    """
    trained = core.trained_slugs(use_hub=use_hub)
    picked = [s for s in stations if s.slug not in trained]
    print(f"untrained: {len(picked)} of {len(stations)} stations have no archive yet",
          flush=True)
    return picked


def _untrained_count() -> int:
    """How many registered stations are still waiting for their first backfill."""
    trained = core.trained_slugs(use_hub=True)
    stations = load_registry()
    n = sum(1 for s in stations if s.slug not in trained)
    print(f"registry: {len(stations)} stations, {n} not yet bootstrapped")
    return n


def _index_path(shard: int | None, of: int | None):
    """Where this worker writes its index entries.

    Sharded runs each write a fragment; a later job merges them. Writing the real
    index.json from a shard would publish a site listing only that shard's stations.
    """
    if of and of > 1:
        return core.SITE_DIR / f"index-shard-{shard or 0}.json"
    return core.SITE_DIR / "index.json"


def cmd_refresh(args) -> int:
    """Update enrolled stations and write the site JSON.

    Uses the bulk path, which acquires observations and forecasts for the whole shard in
    a handful of requests rather than a pair per station.
    """
    from wxfuser.pipeline import bulk_run

    stations = load_registry()
    _stage_catalogue()
    if args.station:
        stations = [s for s in stations if s.id == args.station]
    stations = filter_sources(stations, getattr(args, "sources", None))
    stations = select_shard(stations, args.shard, args.of)
    if getattr(args, "only_untrained", False):
        stations = only_untrained(stations)
    stations = order_stations(stations, getattr(args, "order", None))
    if getattr(args, "limit", None):
        # Applied last, so the cap keeps the head of the ranking rather than an
        # arbitrary slice of it.
        if len(stations) > args.limit:
            print(f"capped at {args.limit} of {len(stations)} stations", flush=True)
        stations = stations[: args.limit]

    if not stations:
        print("no stations to refresh")
        emit.write_json(emit.index_json([]), _index_path(args.shard, args.of))
        return 0

    # Work in batches and checkpoint between them. A bootstrap shard runs for hours, and
    # a job that hits its timeout is killed outright — its final upload step never runs,
    # so everything it built dies with the runner. Checkpointing means an interrupted
    # shard loses one batch rather than an afternoon, and re-running resumes from there.
    checkpoint = args.checkpoint_every or len(stations)
    entries: list[dict] = []
    for i in range(0, len(stations), checkpoint):
        chunk = stations[i : i + checkpoint]
        entries.extend(
            bulk_run.run_stations(
                chunk,
                bootstrap=args.bootstrap,
                # Bootstrapping no longer forces evaluation. Walk-forward verification is
                # around 88% of the per-station cost — 278 hours across the full registry
                # versus 33 for the archive and fit alone — and the weekly retrain already
                # does exactly that job, sharded. Bootstrap's irreplaceable work is the
                # network-bound history it downloads; verification can follow on its own
                # schedule, and until it does a station honestly reports itself unmeasured.
                evaluate=args.evaluate,
                years=args.years,
            )
        )
        if checkpoint < len(stations):
            done = min(i + checkpoint, len(stations))
            ok_so_far = sum(1 for e in entries if e.get("status") == "ok")
            print(f"--- checkpoint: {done}/{len(stations)} stations, {ok_so_far} published",
                  flush=True)
            _checkpoint_state(args.shard, args.of)

    emit.write_json(emit.index_json(entries), _index_path(args.shard, args.of))
    ok = sum(1 for e in entries if e.get("status") == "ok")
    print(f"refresh complete: {ok}/{len(entries)} stations published")
    # A partial refresh still deploys: a stale station beats an empty site.
    return 0


def _checkpoint_state(shard: int | None = None, of: int | None = None) -> None:
    """Push state to the hub mid-run, if one is configured.

    Scoped to this worker's stations: uploading everything would push back the stale
    copies of other shards' stations that this runner restored at startup.

    Best-effort by design: a failed checkpoint should slow the run down, not end it.
    """
    import os
    import subprocess

    if not os.environ.get("HF_TOKEN"):
        return
    script = Path(__file__).resolve().parents[2] / "scripts" / "sync_state.py"
    if not script.exists():
        return
    try:
        cmd = [sys.executable, str(script), "upload"]
        if shard is not None and of:
            cmd += ["--shard", str(shard), "--of", str(of)]
        subprocess.run(cmd, check=False, timeout=900)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: checkpoint upload failed ({exc})", flush=True)


def cmd_merge_index(args) -> int:
    """Combine per-shard index fragments into the single index the site reads.

    Also stages the catalogue, because this runs in the publish job, which assembles the
    site from shard artifacts and never ran a refresh of its own.
    """
    import glob
    import json as _json

    _stage_catalogue()
    entries = []
    frags = sorted(glob.glob(str(core.SITE_DIR / "index-shard-*.json")))
    for f in frags:
        try:
            entries.extend(_json.loads(Path(f).read_text()).get("stations", []))
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: could not read {f} ({exc})")
    # Deduplicate defensively: a re-run shard could contribute a station twice, and the
    # map would then draw it twice.
    seen, unique = set(), []
    for e in entries:
        if e.get("id") in seen:
            continue
        seen.add(e.get("id"))
        unique.append(e)

    # The index is cumulative, not a snapshot of this run. A run may deliberately cover
    # part of the registry — a shard retry, or --sources aimed at the networks whose
    # observations can actually arrive — and rebuilding from its fragments alone would
    # erase every station it did not touch from the map. So this run's entries are laid
    # over the last published index rather than replacing it.
    baseline = _load_index_baseline()
    merged = {e.get("id"): e for e in baseline}
    merged.update({e.get("id"): e for e in unique})
    final = list(merged.values())

    # The publish job runs with `if: always()`, so it also runs when every refresh shard
    # failed. Without this it collects no artifacts, merges an empty index, and deploys it
    # over a working site — which is exactly how the map went to zero stations after the
    # state sync broke. An empty registry is a legitimate cold start; an empty index
    # against a populated registry is a failed run, and must not reach Pages.
    if not final and load_registry():
        print(f"refusing to publish an empty index over {len(load_registry())} enrolled "
              f"stations: no shard produced output, so this run has nothing to say.")
        return 1

    emit.write_json(emit.index_json(final), core.SITE_DIR / "index.json")
    # Keep the baseline in state so the next run inherits it. The site itself is rebuilt
    # from a fresh checkout every deploy and so cannot carry anything forward.
    emit.write_json(emit.index_json(final), _index_baseline_path())
    ok = sum(1 for e in final if e.get("status") == "ok")
    print(f"merged {len(frags)} shards -> {len(unique)} fresh, "
          f"{len(final)} total stations, {ok} published")
    return 0


def _index_baseline_path():
    return core.STATE_DIR / "site" / "index.json"


def _load_index_baseline() -> list[dict]:
    """The last published index, so a partial run adds to the map instead of replacing it."""
    import json as _json

    path = _index_baseline_path()
    if not path.exists():
        return []
    try:
        return _json.loads(path.read_text()).get("stations", [])
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: could not read index baseline ({exc})")
        return []


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
    from wxfuser.pipeline import bulk_run

    stations = load_registry()
    if args.station:
        stations = [s for s in stations if s.id == args.station]
    stations = select_shard(stations, args.shard, args.of)
    if not stations:
        print("no stations to retrain")
        return 0

    entries = bulk_run.run_stations(stations, bootstrap=False, evaluate=True)
    emit.write_json(emit.index_json(entries), _index_path(args.shard, args.of))
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
        include_meteostat=args.meteostat,
        meteostat_exclude_countries=(
            args.meteostat_exclude.split(",") if args.meteostat_exclude else None
        ),
        min_elevation_m=args.min_elevation,
        countries=args.countries.split(",") if args.countries else None,
    )
    if universe.empty:
        print("no stations matched")
        return 1

    if args.order == "prominence":
        # Most-read first. The alternative — enrolling in whatever order the source
        # archives happen to list stations in — spends a capped run on an arbitrary slice
        # of the alphabet, which is how a fleet ends up holding four thousand stations
        # and not the one the user searched for.
        from wxfuser.data import prominence

        try:
            universe = prominence.rank(universe)
        except Exception as exc:  # noqa: BLE001
            # A refresh degrades to registry order when the gazetteer is unreachable,
            # because publishing the same stations in a worse order still publishes them.
            # Enrollment is not the same trade: it decides which stations exist at all,
            # and adding a few hundred arbitrary ones is worse than adding none and
            # trying again next week.
            sys.exit(f"cannot rank by served population ({exc}); "
                     f"declining to enroll an unranked selection")
    elif args.order == "elevation":
        # Highest first: the complex terrain where the correction has the most to do.
        universe = universe.sort_values("elev_m", ascending=False, na_position="last")

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

    # The cap applies to what is *new*, not to what was considered. Applying it to the
    # universe instead means a run whose top hundred are already enrolled adds nothing
    # and reports success, which is how an automated grow loop silently stalls.
    limit = args.limit
    if args.max_backlog is not None:
        room = max(0, args.max_backlog - _untrained_count())
        limit = min(limit, room) if limit else room
        print(f"backlog cap: room for {room} more untrained stations "
              f"(target backlog {args.max_backlog})")
    if limit is not None:
        added = added[:limit]

    print(f"\n{len(added)} new stations to enroll ({len(existing)} already registered)")
    if not added:
        # Not a failure. A scheduled grow run that finds the backlog already full has
        # done its job by declining to add to it.
        return 0
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
    p.add_argument("--evaluate", action="store_true",
                   help="re-run verification and re-choose the champion")
    p.add_argument("--years", type=float, default=2.0, help="years to backfill when bootstrapping")
    p.add_argument("--shard", type=int, help="0-based index of this worker")
    p.add_argument("--of", type=int, help="total number of workers")
    p.add_argument("--checkpoint-every", type=int,
                   help="push state to the hub every N stations, so a killed job loses "
                        "one batch rather than the whole run")
    p.add_argument("--sources",
                   help="comma-separated networks to process (ASOS, SNOTEL, MS). Forecast "
                        "requests are the scarce resource, so an incremental run can skip "
                        "networks whose observations cannot arrive yet")
    p.add_argument("--order", choices=["registry", "prominence"], default="registry",
                   help="prominence puts the stations serving the most people first, so "
                        "an interrupted run drops the least-read ones")
    p.add_argument("--only-untrained", action="store_true",
                   help="skip stations that already have a paired archive; what lets a "
                        "scheduled bootstrap work through the backlog instead of "
                        "restarting it")
    p.add_argument("--limit", type=int,
                   help="process at most this many stations (per shard), after ordering")
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("merge-index", help="combine per-shard index fragments")
    p.set_defaults(func=cmd_merge_index)

    p = sub.add_parser("retrain", help="re-run verification and re-select champions")
    p.add_argument("--station")
    p.add_argument("--shard", type=int, help="0-based index of this worker")
    p.add_argument("--of", type=int, help="total number of workers")
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("enroll-bulk", help="enroll many stations from the bulk sources")
    p.add_argument("--limit", type=int, help="cap how many new stations are added")
    p.add_argument("--order", choices=["prominence", "elevation", "source"],
                   default="prominence",
                   help="which stations a capped run takes: prominence = the ones serving "
                        "the most people, elevation = the hardest terrain")
    p.add_argument("--max-backlog", type=int,
                   help="add only enough to bring the count of not-yet-bootstrapped "
                        "stations up to this number; the self-pacing that lets enrollment "
                        "run unattended without outgrowing what the fleet can train")
    p.add_argument("--min-elevation", type=float,
                   help="only stations at or above this elevation in metres")
    p.add_argument("--countries", help="comma-separated ISO country codes")
    p.add_argument("--no-asos", action="store_true", help="skip the ASOS airport archive")
    p.add_argument("--no-snotel", action="store_true", help="skip SNOTEL mountain sites")
    p.add_argument("--meteostat", action="store_true",
                   help="include Meteostat, which supplies most non-US coverage")
    p.add_argument("--meteostat-exclude", metavar="CC,CC",
                   help="country codes to skip for Meteostat, e.g. already-covered ones")
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
