"""Compare public source timestamps with May issue times using only frozen forecast keys."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def as_time(s):
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.utcoffset() is None:
        raise ValueError(f"Timestamp requires an offset: {s}")
    return t


def dock_map(path, month):
    assigned = defaultdict(set)
    seen = set()
    with path.open() as source:
        for row in csv.DictReader(source):
            if row["month"] == month:
                if row["dock"] in seen:
                    raise ValueError("Dock assigned twice")
                seen.add(row["dock"])
                assigned[row["complex_id"]].add(row["dock"])
    if not seen:
        raise ValueError("Empty monthly dock map")
    return assigned


def run(metadata, detail, output):
    m = json.loads(metadata.read_bytes())
    bike_time = {k: parsedate_to_datetime(v["last_modified"]) for k, v in m["bike"].items()}
    with gzip.open(detail/"weekly_source_times.csv.gz", "rt") as source:
        weekly = {(r["transit_timestamp"][:13], r["station_complex_id"]): r for r in csv.DictReader(source)}
    geometry = defaultdict(list)
    with (detail/"geometry_source_times.csv").open() as source:
        for r in csv.DictReader(source):
            geometry[r["source_day"]].append(as_time(r["first_created"]))
    map_times = {d: max(times) for d, times in geometry.items()}
    with (STUDY/"origins.csv").open() as source:
        ledger = {r["event_id"]: r for r in csv.DictReader(source)}
    prior = dock_map(STUDY.parent/"winter_spring/docks.csv", "2026-04")
    current = dock_map(STUDY/"panel/docks.csv", "2026-05")
    prior_docks = set.union(*prior.values())
    current_docks = set.union(*current.values())
    identical = {c for c in current if current[c] == prior.get(c, set())}
    shared_assignments = sum(len(s & prior.get(c, set())) for c,s in current.items())
    maps = dict(april_docks=len(prior_docks), may_docks=len(current_docks),
                retained_ids=len(prior_docks & current_docks), new_ids=len(current_docks - prior_docks),
                dropped_ids=len(prior_docks - current_docks),
                retained_ids_changing_complex=len(prior_docks & current_docks) - shared_assignments,
                april_complexes=len(prior), may_complexes=len(current),
                may_complexes_with_identical_docks=len(identical))
    controls = STUDY/"controls"
    receipt = json.loads((controls/"receipt.json").read_text())
    path = controls/"operator_rail.csv.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["files"][path.name]["sha256"]:
        raise ValueError("Frozen forecast keys have changed")
    pop = Counter()
    by_event = defaultdict(Counter)
    origins_seen = set()
    used_weekly = set()
    with gzip.open(path, "rt") as source:
        for r in csv.DictReader(source):
            eid, cid, hour = r["event_id"], r["complex_id"], r["hour"]
            if (cid,hour) in origins_seen or ledger[eid]["hour"] != hour or cid not in current:
                raise ValueError("Station-hour mismatch")
            origins_seen.add((cid,hour))
            pop[eid] += 1
            stats = by_event[eid]
            stats["rows_with_changed_dock_set"] += cid not in identical
            origin = as_time(ledger[eid]["origin"])
            lag = as_time(hour) - timedelta(days=7)
            if as_time(m["rail"][str(lag.date())]["first_created"]) <= origin:
                lag_key = lag.strftime("%Y-%m-%dT%H"), cid
                source_row = weekly[lag_key]
                used_weekly.add(lag_key)
                stats["rail_weekly_rows_created"] += as_time(source_row["last_created"]) <= origin
                stats["rail_weekly_rows_last_updated"] += as_time(source_row["last_updated"]) <= origin
    if used_weekly != set(weekly):
        raise ValueError("Unexpected weekly source records")
    dates = defaultdict(Counter)
    events = Counter()
    rows = Counter()
    for eid, n in pop.items():
        origin = as_time(ledger[eid]["origin"])
        lag_day = (as_time(ledger[eid]["hour"]) - timedelta(days=7)).date()
        bike_lag = bike_time[lag_day.strftime("%Y%m")+".zip"] <= origin
        bike_map = bike_time["202604.zip"] <= origin
        geom = map_times["2026-04-29"] <= origin
        rail_created = by_event[eid]["rail_weekly_rows_created"]
        counts = dict(bike_map_object=n * bike_map, bike_weekly_object=n * bike_lag,
                      geometry_first_rows_created=n * geom, rail_weekly_rows_created=rail_created,
                      rail_weekly_rows_last_updated=by_event[eid]["rail_weekly_rows_last_updated"],
                      both_weekly_and_mapping=rail_created if bike_lag and bike_map and geom else 0)
        stats = dates[str(origin.date())]
        stats["paired_origins"] += 1
        stats["paired_rows"] += n
        stats["rows_with_changed_dock_set"] += by_event[eid]["rows_with_changed_dock_set"]
        for k, count in counts.items():
            events[k] += int(count > 0)
            rows[k] += count
            stats[k] += count
    if sum(pop.values()) != receipt["evaluation_rows"] or len(pop) != 1593:
        raise ValueError("Different May support")
    output.mkdir(parents=True, exist_ok=True)
    with (output/"by_date.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["issue_date", *next(iter(dates.values()))],lineterminator="\n")
        w.writeheader()
        for d in sorted(dates):
            w.writerow(dict(issue_date=d, **dates[d]))
    report = dict(paired_origins=len(pop), paired_rows=sum(pop.values()), origins_with_any_passing_row=events, station_hours=rows,
                  dock_maps=maps, complete_geometry_first_row_times={d:t.isoformat() for d,t in map_times.items()}, rows_with_changed_dock_set=sum(r["rows_with_changed_dock_set"] for r in dates.values()))
    (output/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metadata",type=Path)
    p.add_argument("detail",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    run(args.metadata, args.detail, args.output)
