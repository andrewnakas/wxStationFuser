"""Command line entry point — the interface the GitHub Actions workflows drive."""
from __future__ import annotations

import argparse
import json
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


def filter_bbox(stations: list, spec: str | None) -> list:
    """Restrict a run to a geographic box, given as ``south,west,north,east``.

    Refreshing the whole fleet is hours of throttled work, so when a particular region
    matters — a mountain range in its snow season, a coast before a storm — it is worth
    being able to spend the budget there rather than waiting for a full pass to come
    round to it. The western US, Rockies included, is ``31,-125,49.5,-104``.
    """
    if not spec:
        return stations
    try:
        south, west, north, east = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(
            f"--bbox wants four numbers, south,west,north,east; got {spec!r}"
        ) from None
    picked = [
        s for s in stations
        if south <= s.lat <= north and west <= s.lon <= east
    ]
    print(f"bbox {spec}: {len(picked)} of {len(stations)} stations", flush=True)
    return picked


def only_trained(stations: list) -> list:
    """Keep only the stations that already have an archive to publish from.

    The fastest way to refill an empty map. The registry holds thousands of stations that
    have been enrolled but not yet backfilled; they cannot produce a page, so under a
    rationed forecast API every request they consume is one the map does not get.
    """
    trained = core.trained_slugs()
    picked = [s for s in stations if s.slug in trained]
    print(f"trained: {len(picked)} of {len(stations)} stations have an archive to publish",
          flush=True)
    return picked


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
    stations = filter_bbox(stations, getattr(args, "bbox", None))
    if getattr(args, "only_trained", False):
        stations = only_trained(stations)
    if getattr(args, "limit", None) and len(stations) > args.limit:
        print(f"capped at {args.limit} of {len(stations)} stations", flush=True)
        stations = stations[: args.limit]

    if getattr(args, "rebuild", False) and not args.bootstrap:
        raise SystemExit("--rebuild discards history, so it needs --bootstrap to replace it")

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
        if getattr(args, "rebuild", False):
            # Per chunk, not up front: an interrupted run then leaves the stations it
            # never reached exactly as they were, rather than stripped of their history.
            for st in chunk:
                core.discard_history(st)
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


def cmd_repair_obs(args) -> int:
    """Re-pair stored forecasts against re-fetched observations; see repair_observations."""
    from wxfuser.pipeline import bulk_run

    stations = filter_sources(load_registry(), args.sources)
    stations = select_shard(stations, args.shard, args.of)
    step = args.checkpoint_every or len(stations) or 1
    repaired = untouched = 0
    for i in range(0, len(stations), step):
        for r in bulk_run.repair_observations(stations[i : i + step]):
            if r["status"] == "repaired":
                repaired += 1
            else:
                untouched += 1
        _checkpoint_state(args.shard, args.of)
    print(f"repair complete: {repaired} repaired, {untouched} left untouched")
    return 0


def cmd_org_run(args) -> int:
    """Run every spec in one org's spec file, into that org's private state root.

    The org's state and published forecasts live under ``--root`` (the org's R2 prefix,
    mirrored locally), never in the public state or site directories. Nothing an org
    owns can end up on the public hub or GitHub Pages by running the public jobs, and
    nothing here writes to theirs.
    """
    from wxfuser.data import org_obs
    from wxfuser.pipeline import org as org_run
    from wxfuser.spec import load_org

    org, specs = load_org(args.specs)
    root = org_run.use_root(args.root)
    uploaded = org_obs.ingest_uploads(root / "uploads", org)
    if uploaded:
        print(f"uploads: ingested {', '.join(uploaded)}", flush=True)
    held = org_obs.compact_inbox(root / "inbox", org)
    if held:
        print(f"inbox: {len(held)} stations updated", flush=True)
    if args.spec:
        specs = [s for s in specs if s.key() == args.spec]
    print(f"org {org}: {len(specs)} specs", flush=True)

    index = [org_run.run_spec(spec, bootstrap=args.bootstrap, years=args.years,
                              evaluate=args.evaluate or None) for spec in specs]
    org_run.write_index(org, index)
    ok = sum(1 for e in index if e["status"] == "ok")
    print(f"org {org}: {ok}/{len(index)} specs published")
    return 0


def cmd_calibrate(args) -> int:
    """From a link to a station's data to a calibrated forecast and report."""
    from datetime import date, timedelta

    from wxfuser import quick
    from wxfuser.data import obs as obs_mod
    from wxfuser.data import org_obs
    from wxfuser.pipeline import org as org_run
    from wxfuser.report import write_report
    from wxfuser.spec import FusionSpec

    res = quick.resolve(args.link, name=args.name, lat=args.lat, lon=args.lon,
                        elev_m=args.elev)
    st = res.station
    out = Path(args.out or f"calibrations/{quick.slugify(st.name)}")
    org_run.use_root(out / "root")
    print(f"{res.kind}: {st.name} ({st.id}) at {st.lat:.4f}, {st.lon:.4f}"
          + (f", {st.elev_m:.0f} m" if st.elev_m else ""), flush=True)

    years = args.years
    if res.kind == "csv":
        df = quick.load_csv(res.csv_url)
        if args.col:
            time_col, mapping_specs = args.time_col, args.col
            if not time_col:
                raise SystemExit("--col needs --time-col")
        else:
            time_col, mapping_specs, notes = quick.detect_columns(df)
            time_col = args.time_col or time_col
            print("detected columns (override with --time-col and --col):")
            for n in notes:
                print(f"  {n}")
        mapping = org_obs.parse_mapping(mapping_specs)
        tmp = out / "source.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=False)
        if args.tz:
            tz = args.tz
        else:
            tz, why = quick.choose_time_zone(str(tmp), st.id, mapping, time_col, st.lon)
            print(f"  {why} (override with --tz)")
        records = org_obs.read_csv(tmp, st.id, mapping, time_col=time_col, tz=tz)
        org_obs.append_raw(records)
        span = (records["time"].max() - records["time"].min()).days
        print(f"{len(records)} records, {records['time'].min():%Y-%m-%d} .. "
              f"{records['time'].max():%Y-%m-%d} UTC ({span} days)", flush=True)
        if years is None:
            years = max(0.1, min(2.0, span / 365.0))
        columns = {c for c in records.columns if records[c].notna().any()}
    else:
        today = date.today()
        probe = obs_mod.fetch_obs(st.id, today - timedelta(days=10), today,
                                  iem_network=st.iem_network)
        columns = {c for c in probe.columns if probe[c].notna().any()}
        if years is None:
            years = 1.0

    variables = args.vars.split(",") if args.vars else quick.variables_for(columns)
    if not variables:
        raise SystemExit("no forecastable variables found in this station's data")
    models = args.models.split(",") if args.models else quick.default_models(st.lat, st.lon)
    thresholds = {v: t for v, t in quick.DEFAULT_THRESHOLDS.items() if v in variables}
    spec = FusionSpec(org="adhoc", station=st, models=models, variables=variables,
                      thresholds=thresholds, label=args.name or st.name)
    spec.validate()
    print(f"calibrating {', '.join(variables)} with {' + '.join(models)} "
          f"over {years:.2f} years of history", flush=True)
    entry = org_run.run_spec(spec, bootstrap=True, years=years)
    org_run.write_index("adhoc", [entry])
    if entry["status"] != "ok":
        print(f"not published: {entry['status']}")
        return 1
    fc_path = org_run.forecast_path(spec)
    payload = quick.withhold_unverifiable_snow(
        json.loads(fc_path.read_text()), core.pairs_path(spec.to_station()))
    emit.write_json(payload, fc_path)
    (out / "forecast.json").write_text(fc_path.read_text())
    report = write_report(payload, out / "report.html")
    # The same report as page content (web fonts, no document shell), for publishing.
    page = write_report(payload, out / "page.html", fragment=True)
    print(f"\nforecast: {out / 'forecast.json'}\nreport:   {report}\npage:     {page}")
    return 0


def cmd_snotel_archive(args) -> int:
    """Fetch every SNOTEL station's full hourly history (resumable)."""
    from wxfuser import archive

    states = args.states.split(",") if args.states else None
    result = archive.build_snotel_archive(args.out, workers=args.workers, limit=args.limit,
                                          states=states)
    print(f"archive: {result['archived']}/{result['stations']} stations in {args.out}")
    if args.upload:
        archive.upload_to_hub(args.out)
        print(f"published to https://huggingface.co/datasets/{archive.HUB_REPO}")
    return 0


def cmd_point_archive(args) -> int:
    """Extract a model's archived runs at SNOTEL points, month by month (resumable)."""
    from datetime import date

    from wxfuser import points

    result = points.build(
        args.model,
        states=args.states.split(",") if args.states else None,
        out=args.out,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        init_hours=[int(h) for h in args.init_hours.split(",")] if args.init_hours else None,
        shard=args.shard or 0, of=args.of or 1, hub_repo=args.hub_repo,
        budget_s=args.budget_minutes * 60 if args.budget_minutes else None,
    )
    print(f"{result['model']}: wrote {len(result['written'])} months, {result['remaining']} remain")
    return 0


def cmd_network_train(args) -> int:
    """Evaluate on unseen stations and time, then fit and save the network model."""
    from wxfuser import network

    models = args.models.split(",")
    root = Path(args.archive)
    if args.fetch:
        root = network.fetch_archive(root, models)
    report = {}
    for variable in args.variables.split(","):
        print(f"== {variable}", flush=True)
        data = network.build_dataset(root, models, variable, issue_fraction=args.issue_fraction)
        if data.empty:
            print("  no paired data")
            continue
        print(f"  {len(data):,} rows, {data['station_id'].nunique()} stations, "
              f"{data['valid_time'].min():%Y-%m} .. {data['valid_time'].max():%Y-%m}", flush=True)
        ev = network.leave_stations_out(data, variable, models, folds=args.folds,
                                        rounds=args.rounds, test_from=args.test_from)
        print(f"  CRPSS vs {ev['raw_best_model']}: {ev['crpss_vs_raw_best']:.3f} "
              f"({ev['scheme']}, {ev['stations']} stations, {ev['rows']:,} rows); "
              f"{ev['stations_improved']:.0%} of stations improved", flush=True)
        model = network.fit(data, variable, models, rounds=args.rounds)
        path = network.save(model, Path(args.out) / f"{variable}.json", ev)
        print(f"  saved {path}")
        report[variable] = ev
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "evaluation.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


def cmd_network_compare(args) -> int:
    """Train model sets on one table and score them on identical held-out rows."""
    from wxfuser import network

    sets = {name: ms.split(",") for name, ms in (x.split("=", 1) for x in args.sets)}
    every = sorted({m for ms in sets.values() for m in ms})
    root = Path(args.archive)
    if args.fetch:
        root = network.fetch_archive(root, every)
    report = {}
    for variable in args.variables.split(","):
        print(f"== {variable}", flush=True)
        data = network.build_dataset(root, every, variable, issue_fraction=args.issue_fraction)
        thresholds = tuple(float(t) for t in args.thresholds.split(",")) if (
            args.thresholds and variable in ("hn24_cm", "swe_24h_mm", "precip_1h_mm")) else ()
        if variable == "precip_1h_mm":
            thresholds = (1.0,)
        report[variable] = network.compare(data, variable, sets, test_from=args.test_from,
                                           rounds=args.rounds, thresholds=thresholds)
        print(json.dumps(report[variable], indent=1, default=str), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "comparison.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


def cmd_history(args) -> int:
    """Replay stations' calibrated forecasts through history, for the explorer page."""
    import pandas as pd

    from wxfuser import history, network
    from wxfuser.data.bulk import snotel_stations

    models = args.models.split(",")
    ids = args.stations.split(",")
    root = Path(args.archive)
    if args.fetch:
        root = network.fetch_archive(root, models)
    meta = snotel_stations().set_index("id")
    out = {"models": models, "since": args.since, "stations": []}
    for sid in ids:
        entry = {"id": sid, "name": str(meta.loc[sid, "name"]) if sid in meta.index else sid,
                 "elev_m": float(meta.loc[sid, "elev_m"]) if sid in meta.index else None,
                 "lon": float(meta.loc[sid, "lon"]) if sid in meta.index else None,
                 "variables": {}}
        for variable in history.SAMPLING:
            print(f"== {sid} {variable}", flush=True)
            data = network.build_dataset(root, models, variable, station_ids=[sid])
            data = data[pd.to_datetime(data["valid_time"]) >= pd.Timestamp(args.since)]
            frame, champion = history.replay(data, variable, models)
            if frame.empty:
                continue
            frame = history.sample(frame, variable)
            block = {"champion": champion, "summary": history.summary(frame, models),
                     "rows": history.columnar(frame, models)}
            if variable in ("hn24_cm", "swe_24h_mm"):
                block["events"] = history.events(frame, models)
            print(f"  {champion}: {len(frame):,} rows, {block['summary']}", flush=True)
            entry["variables"][variable] = block
        out["stations"].append(entry)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_ingest_csv(args) -> int:
    """Append a logger CSV to an org station's observation store."""
    from wxfuser.data import org_obs

    if not args.station.startswith("ORG:"):
        raise SystemExit("ingest-csv writes org stations only (ORG:<org>:<id>)")
    mapping = org_obs.parse_mapping(args.col)
    records = org_obs.read_csv(args.file, args.station, mapping,
                               time_col=args.time_col, tz=args.tz)
    if records.empty:
        raise SystemExit("no readable rows; check --time-col and --tz")
    print(f"{len(records)} records, {records['time'].min()} .. {records['time'].max()} UTC")
    if args.lon is not None:
        peak = org_obs.diurnal_peak_solar_hour(records, args.lon)
        if peak is not None:
            print(f"temperature peaks at {peak:.0f}:00 local solar time")
            if not 11 <= peak <= 18:
                # Not fatal: a site in a deep, west-facing valley can peak late. Worth
                # a look, though, because a wrong zone is far more common.
                print("WARNING: an afternoon peak is expected; check --tz. A peak near "
                      "dawn usually means local time was read as UTC or vice versa.")
    held = org_obs.append_raw(records, Path(args.root) / "observations")
    print(f"store now holds {held.get(args.station, 0)} records for {args.station}")
    return 0


def cmd_station_key(args) -> int:
    """Issue a push-API key for one org station, storing only its hash.

    The key is printed once and never stored. ``keys.json`` in the org root (served to
    the Worker as orgs/{org}/keys.json) maps each key's SHA-256 to its station, so a
    leaked copy of the table authenticates nothing. Re-issuing for a station revokes
    its previous key.
    """
    import hashlib
    import json as _json
    import secrets

    prefix = f"ORG:{args.org}:"
    if not args.station.startswith(prefix):
        raise SystemExit(f"station must be one of this org's: {prefix}<id>")
    path = Path(args.root) / "keys.json"
    table = _json.loads(path.read_text()) if path.exists() else {}
    table = {h: st for h, st in table.items() if st != args.station}
    key = f"t60sk_{secrets.token_urlsafe(32)}"
    table[hashlib.sha256(key.encode()).hexdigest()] = args.station
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(table, indent=1, sort_keys=True))
    print(f"station key for {args.station} (shown once; previous key revoked):\n{key}")
    print("It takes effect after the next org-sync push, within 5 minutes.")
    return 0


def cmd_org_sync(args) -> int:
    """Pull an org's private prefix from R2, or push it back."""
    from wxfuser import r2

    if args.direction == "pull":
        n = r2.pull(args.org, args.root)
        print(f"pulled {n} files for {args.org}")
    else:
        n = r2.push(args.org, args.root)
        print(f"pushed {n} changed files for {args.org}")
    return 0


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
    p.add_argument("--evaluate", action="store_true",
                   help="re-run verification and re-choose the champion")
    p.add_argument("--years", type=float, default=2.0, help="years to backfill when bootstrapping")
    p.add_argument(
        "--rebuild", action="store_true",
        help="with --bootstrap: discard each station's stored pairs and observations first, "
             "for when stored history is known to be wrong rather than merely short",
    )
    p.add_argument("--shard", type=int, help="0-based index of this worker")
    p.add_argument("--of", type=int, help="total number of workers")
    p.add_argument("--checkpoint-every", type=int,
                   help="push state to the hub every N stations, so a killed job loses "
                        "one batch rather than the whole run")
    p.add_argument("--sources",
                   help="comma-separated networks to process (ASOS, SNOTEL, MS). Forecast "
                        "requests are the scarce resource, so an incremental run can skip "
                        "networks whose observations cannot arrive yet")
    p.add_argument("--bbox", metavar="S,W,N,E",
                   help="only stations inside this box, e.g. 31,-125,49.5,-104 for the "
                        "western US; lets a run spend its budget on one region")
    p.add_argument("--only-trained", action="store_true",
                   help="only stations that already have an archive, and can therefore "
                        "publish a page; the fastest way to refill an empty map")
    p.add_argument("--limit", type=int,
                   help="process at most this many stations per shard")
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("org-run", help="run one org's fusion specs into its private root")
    p.add_argument("--specs", required=True, help="the org's specs.yaml")
    p.add_argument("--root", required=True,
                   help="the org's local state root (mirrors orgs/{org}/ in R2)")
    p.add_argument("--spec", help="run only the spec with this key")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--years", type=float, default=2.0)
    p.set_defaults(func=cmd_org_run)

    p = sub.add_parser(
        "calibrate",
        help="from a link to a station's data (SNOTEL, IEM, MesoWest, or any CSV/Google "
             "Sheet) to a calibrated forecast and report",
    )
    p.add_argument("link", help="station page URL, CSV URL, Google Sheets link, or local file")
    p.add_argument("--name", help="station name for the report")
    p.add_argument("--lat", type=float, help="latitude (required for CSV data)")
    p.add_argument("--lon", type=float, help="longitude (required for CSV data)")
    p.add_argument("--elev", type=float, help="elevation in metres")
    p.add_argument("--models", help="comma-separated, e.g. hrrr,gefs (default: by region)")
    p.add_argument("--vars", help="comma-separated variables (default: whatever the data has)")
    p.add_argument("--years", type=float, help="history to train on (default: the CSV's span, "
                                                 "or 1 year for network stations)")
    p.add_argument("--time-col", help="override the detected timestamp column")
    p.add_argument("--tz", help="override the detected time zone (UTC, -07:00, America/Denver)")
    p.add_argument("--col", action="append", metavar="TARGET=COLUMN:UNIT",
                   help="override column detection; repeat per column (as ingest-csv)")
    p.add_argument("--out", help="output directory (default calibrations/<name>)")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("snotel-archive",
                       help="fetch every SNOTEL station's full hourly history (resumable)")
    p.add_argument("--out", default="archive/snotel")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="only this many stations (for a trial)")
    p.add_argument("--states", help="comma-separated state codes, e.g. MT or MT,ID,WY")
    p.add_argument("--no-upload", dest="upload", action="store_false",
                   help="keep the archive local instead of publishing it to nakas/wxfuser-archive")
    p.set_defaults(func=cmd_snotel_archive)

    p = sub.add_parser("point-archive",
                       help="extract a dynamical.org model's archived runs at SNOTEL points")
    p.add_argument("--model", required=True, help="hrrr, gefs, ecmwf_ens, gfs16, aifs_single")
    p.add_argument("--states", help="comma-separated state codes, e.g. MT")
    p.add_argument("--out", default="archive/points")
    p.add_argument("--start", help="first month (YYYY-MM-DD); default: the store's start")
    p.add_argument("--end", help="last date; default: the store's latest run")
    p.add_argument("--init-hours", help="only runs initialised at these UTC hours, e.g. 0,12")
    p.add_argument("--shard", type=int)
    p.add_argument("--of", type=int)
    p.add_argument("--hub-repo", help="publish each month to this Hugging Face dataset")
    p.add_argument("--budget-minutes", type=float,
                   help="stop starting new months after this long (CI jobs)")
    p.set_defaults(func=cmd_point_archive)

    p = sub.add_parser("network-train",
                       help="train one calibration model across the archived station network")
    p.add_argument("--archive", default="archive/hub", help="local copy of nakas/wxfuser-archive")
    p.add_argument("--fetch", action="store_true", help="download the archive first")
    p.add_argument("--models", required=True, help="e.g. hrrr,gefs,ecmwf_ens")
    p.add_argument("--variables", default="air_temp_c")
    p.add_argument("--test-from", help="score only on hours from this date (and unseen stations)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=400)
    p.add_argument("--issue-fraction", type=float, default=1.0,
                   help="train on a random share of issue times, to fit in memory")
    p.add_argument("--out", default="archive/network")
    p.set_defaults(func=cmd_network_train)

    p = sub.add_parser("network-compare",
                       help="score model sets on identical held-out rows, with winter and event views")
    p.add_argument("--sets", action="append", required=True, metavar="NAME=M1,M2",
                   help="e.g. --sets hrrr=hrrr --sets fused=hrrr,gefs,ecmwf_ens")
    p.add_argument("--archive", default="archive/hub")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--variables", default="hn24_cm")
    p.add_argument("--test-from", required=True)
    p.add_argument("--thresholds", default="15,30", help="event thresholds for snow (cm / mm)")
    p.add_argument("--issue-fraction", type=float, default=0.4)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--out", default="archive/network/compare")
    p.set_defaults(func=cmd_network_compare)

    p = sub.add_parser("history", help="replay calibrated forecasts through history (explorer data)")
    p.add_argument("--stations", required=True, help="comma-separated SNOTEL triplets")
    p.add_argument("--models", default="hrrr,gefs,ecmwf_ens")
    p.add_argument("--since", default="2024-04-01")
    p.add_argument("--archive", default="archive/hub")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--out", default="archive/history.json")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("ingest-csv", help="add a logger CSV to an org station's observations")
    p.add_argument("file")
    p.add_argument("--root", required=True, help="the org's local state root")
    p.add_argument("--station", required=True, help="ORG:<org>:<id>")
    p.add_argument("--time-col", required=True)
    p.add_argument("--tz", required=True,
                   help="IANA zone (America/Denver), UTC, or a fixed offset (-07:00) for "
                        "loggers kept on standard time all year")
    p.add_argument("--col", action="append", required=True, metavar="TARGET=COLUMN:UNIT",
                   help="e.g. air_temp_c=TempF:F, wind_gust_ms=Gust:mph, "
                        "precip_mm=Precip:in:cumulative; repeat per column")
    p.add_argument("--lon", type=float, help="station longitude, for a time-zone sanity check")
    p.set_defaults(func=cmd_ingest_csv)

    p = sub.add_parser("station-key", help="issue a push-API key for an org station")
    p.add_argument("--org", required=True)
    p.add_argument("--station", required=True, help="ORG:<org>:<id>")
    p.add_argument("--root", required=True, help="the org's local root (pulled from R2)")
    p.set_defaults(func=cmd_station_key)

    p = sub.add_parser("org-sync", help="pull or push one org's private state in R2")
    p.add_argument("direction", choices=["pull", "push"])
    p.add_argument("--org", required=True)
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_org_sync)

    p = sub.add_parser(
        "repair-obs",
        help="re-pair stored forecasts with re-fetched observations, for archives whose "
             "observation side is known to be wrong",
    )
    p.add_argument("--sources", required=True,
                   help="networks to repair, e.g. SNOTEL; required so a repair is never "
                        "run across the whole registry by accident")
    p.add_argument("--shard", type=int)
    p.add_argument("--of", type=int)
    p.add_argument("--checkpoint-every", type=int, default=40)
    p.set_defaults(func=cmd_repair_obs)

    p = sub.add_parser("merge-index", help="combine per-shard index fragments")
    p.set_defaults(func=cmd_merge_index)

    p = sub.add_parser("retrain", help="re-run verification and re-select champions")
    p.add_argument("--station")
    p.add_argument("--shard", type=int, help="0-based index of this worker")
    p.add_argument("--of", type=int, help="total number of workers")
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("enroll-bulk", help="enroll many stations from the bulk sources")
    p.add_argument("--limit", type=int, help="cap the number enrolled, highest elevation first")
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
