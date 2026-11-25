"""Rebuild the frozen May validation population, geography and paired outcome mask."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "may_parser"))
from audit_cohort import timetable
from build_sample import api, inputs, write_csv
from build_winter_spring_panel import SPRING, coordinates, prepare_bikes, rail_day, write_panel
from extract_alert_roles import StationDictionary
from score import resolve, rule_output

MAY, JUNE = date(2026, 5, 1), date(2026, 6, 1)
CUTOVER = date(2026, 5, 17)
CAPTURE = datetime(2026, 5, 18, 0, 26, 20, tzinfo=timezone.utc)


def alert_history(records, sources):
    first = {r["event_id"]: r for r in records("may_notices.json", sources["may_notices.json"]["url"])}
    query = {"$where": "agency='NYCT Subway' AND date >= '2026-05-01T04:00:00' AND date < '2026-06-01T04:00:00'",
             "$order": "date,event_id,update_number,alert_id", "$limit": 50000}
    rows = records("versions-2026-05.json", api("7kct-peq7", **query))
    if len(rows) >= query["$limit"]:
        raise ValueError("Truncated version history")
    history = defaultdict(dict)
    for row in rows:
        eid, n = row["event_id"], int(row["update_number"])
        if n in history[eid]:
            raise ValueError("Duplicate version")
        history[eid][n] = row
    if first != {eid: v[0] for eid, v in history.items() if 0 in v}:
        raise ValueError("First notices differ between snapshots")
    return first, history


def summarize(rows, origins):
    def observed(row, fields):
        return all(row[f] is not None for f in fields)
    both = [r for r in rows if observed(r, ("rail", "bike"))]
    paired = [r for r in both if observed(r, ("weekly_rail", "weekly_bike"))]
    def tally(group):
        return dict(station_hours=len(group), origins=len({r["event_id"] for r in group}),
                    hours=len({r["hour"] for r in group}), dates=len({r["hour"][:10] for r in group}),
                    complexes=len({r["complex_id"] for r in group}))
    clean = {e for e, r in origins.items() if not any(int(r[k]) for k in ("audit_case","audit_text","audit_episode"))}
    fault = {e for e, r in origins.items() if r["class_cue"] == "fault or response wording"}
    return dict(before_mask=tally(rows), both_targets=tally(both), both_targets_and_lags=tally(paired),
                paired_after_audit_exclusion=tally([r for r in paired if r["event_id"] in clean]),
                paired_fault_response=tally([r for r in paired if r["event_id"] in fault]),
                paired_rule_named=tally([r for r in paired if r["roles"]]),
                overlapping_station_hours=sum(r["overlap"] > 1 for r in rows))


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    old = json.loads((HERE.parent / "winter_spring/sources.json").read_text())
    registration = json.loads((HERE / "sources.json").read_text())
    release = HERE / "panel/sources.json"
    expected = json.loads(release.read_text()) if release.exists() else {}
    def source(name, url):
        path = inputs(cache, manifest, name, url)
        for pinned in (old, registration, expected):
            if name in pinned and manifest[name] != pinned[name]:
                raise ValueError(f"Source changed: {name}")
        return path
    def records(name, url):
        return json.loads(source(name, url).read_bytes())

    with (HERE / "origins.csv").open() as f:
        ordered = sorted(csv.DictReader(f), key=lambda r: (datetime.fromisoformat(r["origin"]), int(r["event_id"])))
    origins = {r["event_id"]: r for r in ordered}
    first, histories = alert_history(records, registration)
    if len(origins) != len(ordered) or origins.keys() != first.keys():
        raise ValueError("Different origin ledger")
    stations = records("stations", api("39hk-dx4f"))
    dictionary = StationDictionary(stations)
    coords = coordinates(records, MAY)
    spring = timetable(source("20260227082256.zip", SPRING), stations, coords)
    summer = timetable(source("may-base.zip", registration["may-base.zip"]["url"]), stations, coords)
    bikes, city, mapped = prepare_bikes(source, {MAY: coords}, out, [date(2026,4,1), MAY, JUNE])
    rail = {}
    d = MAY - timedelta(days=7)
    while d < JUNE:
        rail.update(rail_day(records, d))
        d += timedelta(days=1)
    def departures(cid, h):
        if cid in mapped[MAY] and city.get(h, 0):
            return bikes[MAY,h][cid]
        return None

    populations = {"primary": {}, "schedule_excluded": {}}
    events, versions = [], []
    for origin in ordered:
        eid = origin["event_id"]
        raw = first[eid]
        issue = datetime.fromisoformat(origin["origin"])
        target = datetime.fromisoformat(origin["hour"])
        archived = datetime.fromisoformat(raw["date"]).replace(tzinfo=timezone.utc)
        if (raw["alert_id"] != origin["alert_id"] or archived != issue or
            raw.get("affected","") != origin["routes"] or target != archived.replace(minute=0, second=0, microsecond=0)+timedelta(hours=1) or
            not (MAY <= target.date() < JUNE)):
            raise ValueError("May ledger mismatch")
        routes = set(origin["routes"].split(" | ")) - {""}
        later = target.date() >= CUTOVER
        arm = "schedule_excluded" if later and issue < CAPTURE else "primary"
        panel = populations[arm]
        complexes, unknown, inactive = (summer if later else spring)(target, routes)
        if target.date() == CUTOVER:
            previous, unknown_before, inactive_before = spring(target, routes, service_before=CUTOVER)
            complexes |= previous
            unknown |= unknown_before
            inactive &= inactive_before
        roles = defaultdict(set)
        case = dict(routes=origin["routes"], header=raw["header"], description=raw.get("description",""))
        for name, role in rule_output(case,dictionary):
            cid = resolve(name,routes,dictionary)
            if cid:
                roles[cid].add(role)
        for n, v in sorted(histories[eid].items()):
            versions.append(dict(event_id=eid, update_number=n, alert_id=v["alert_id"],
                                 published_utc=v["date"]+"Z", used_as_input=int(n==0)))
        events.append(dict(event_id=eid, alert_id=raw["alert_id"], population=arm,
                           mapped_complexes=len(complexes), versions_in_window=len(histories[eid]),
                           inactive_routes="|".join(sorted(inactive)), unmapped_stops="|".join(sorted(unknown))))
        h = target.strftime("%Y-%m-%d %H")
        lag = (target-timedelta(days=7)).strftime("%Y-%m-%d %H")
        for cid in complexes:
            key = origin["hour"], cid
            if key in panel:
                panel[key]["overlap"] += 1
            else:
                panel[key] = dict(hour=key[0], complex_id=cid, event_id=eid, overlap=1,
                                  rail=rail.get((h,cid)), bike=departures(cid,h),
                                  weekly_rail=rail.get((lag,cid)), weekly_bike=departures(cid,lag),
                                  roles="|".join(sorted(roles[cid])))
    if sum(r["population"] == "schedule_excluded" for r in events) != 34:
        raise ValueError("Schedule exclusion differs from registration")
    write_csv(out/"events.csv", events)
    write_panel(out/"versions.csv.gz", versions)
    summary = {}
    for name, panel in populations.items():
        rows = [panel[k] for k in sorted(panel,key=lambda k: (k[0],int(k[1])))]
        write_panel(out/(name+".csv.gz"), rows)
        group = [r for r in events if r["population"] == name]
        summary[name] = dict(first_notices=len(group), notices_with_route_stations=sum(r["mapped_complexes"]>0 for r in group),
                             **summarize(rows, origins))
    summary.update(versions_of_may_origins=len(versions), coordinate_complexes=len(coords),
                   mapped_complexes_with_docks=len(mapped[MAY]),
                   observed_may_city_bike_hours=sum(h.startswith("2026-05") and n>0 for h,n in city.items()))
    if expected and manifest != expected:
        raise ValueError("Different source inventory")
    for name, data in (("sources", manifest), ("coverage", summary)):
        (out/(name+".json")).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.cache, args.output)
