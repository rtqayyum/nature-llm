"""Qualification of the NYC Aug 8–10 development cohort. Run with Python 3.9+."""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_sample import LOCAL, api, distance, inputs, read_bikes, write_csv

ROOT = Path(__file__).resolve().parent
SCHEDULE = "https://storage.googleapis.com/download/storage/v1/b/mdb-latest/o/us-new-york-mta-new-york-city-transit-mta-gtfs-516.zip?generation=1780535748331039&alt=media"
ALERT_QUERY = {
    "$where": "agency='NYCT Subway' AND date >= '2026-08-08T04:00:00' AND date < '2026-08-15T05:00:00'",
    "$order": "date,event_id,update_number,alert_id",
    "$limit": 50000,
}


def cause_screen(text):
    active = text.casefold().split("planned work reminder")[0]
    if re.search(r"maintenance|maintanence|station improvements|replacing a rail", active):
        return "maintenance or project"
    if "marine traffic" in active:
        return "marine bridge"
    if re.search(r"brake|signal problem|malfunction|mechanical|door problem|switch problem|"
                 r"unable to move track switches|power|fdny|nypd|ems\b|smoke|fire|investigat|"
                 r"in need of (cleaning|repair)|disruptive", active):
        return "fault or response wording"
    return "unclassified"


def timetable(path, station_rows, places):
    def records(name):
        with z.open(name + ".txt") as src:
            yield from csv.DictReader(io.TextIOWrapper(src, encoding="utf-8-sig"))

    with zipfile.ZipFile(path) as z:
        physical = {r["gtfs_stop_id"]: r for r in station_rows}
        calendar = list(records("calendar"))
        exceptions = list(records("calendar_dates"))
        trips = {r["trip_id"]: (r["service_id"], r["route_id"]) for r in records("trips")}
        parent = {}
        complex_ids = {}
        for r in records("stops"):
            parent[r["stop_id"]] = r["parent_station"] or r["stop_id"]
            if r["location_type"] != "1":
                continue
            station = physical.get(r["stop_id"])
            if station and station["complex_id"] in places:
                coord = float(r["stop_lat"]), float(r["stop_lon"])
                check = float(station["gtfs_latitude"]), float(station["gtfs_longitude"])
                if distance(coord, check) <= 100:
                    complex_ids[r["stop_id"]] = station["complex_id"]
        calls = defaultdict(set)
        for r in records("stop_times"):
            service, route = trips[r["trip_id"]]
            hour = int(r["departure_time"].split(":")[0])
            calls[service, hour, route].add(parent[r["stop_id"]])

    def stations(hour, routes, service_before=None):
        served, active_routes = set(), set()
        for lag in range(1 + max(key[1] for key in calls) // 24):
            day = hour.date() - timedelta(days=lag)
            if service_before is not None and day >= service_before:
                continue
            date = day.strftime("%Y%m%d")
            weekday = day.strftime("%A").lower()
            active = {r["service_id"] for r in calendar
                      if r["start_date"] <= date <= r["end_date"] and r[weekday] == "1"}
            for r in exceptions:
                if r["date"] == date:
                    if r["exception_type"] == "1":
                        active.add(r["service_id"])
                    else:
                        active.discard(r["service_id"])
            for service in active:
                for route in routes:
                    stops = calls.get((service, hour.hour + 24 * lag, route), set())
                    if stops:
                        active_routes.add(route)
                        served.update(stops)
        return {complex_ids[s] for s in served if s in complex_ids}, served - complex_ids.keys(), routes - active_routes

    return stations


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def read(name, url):
        return json.loads(inputs(cache, manifest, name, url).read_bytes())

    sample_sources = json.loads((ROOT / "sample/manifest.json").read_text())["sources"]
    places = {}
    for name in ("08", "17"):
        for r in read(name, sample_sources[name]["url"]):
            places[r["station_complex_id"]] = (float(r["latitude"]), float(r["longitude"]))
    station_rows = read("stations", api("39hk-dx4f"))
    schedule = inputs(cache, manifest, "seasonal.zip", SCHEDULE)
    if hashlib.sha1(schedule.read_bytes()).hexdigest() != "ed28414bf8c061381dd39ef7806e46d4e12d48b7":
        raise ValueError("Different seasonal timetable")
    served = timetable(schedule, station_rows, places)

    archive = read("alerts.json", api("7kct-peq7", **ALERT_QUERY))
    if len(archive) >= ALERT_QUERY["$limit"]:
        raise ValueError("Alert response requires pagination")
    versions = defaultdict(dict)
    for r in archive:
        updates = versions[r["event_id"]]
        seq = int(r["update_number"])
        if seq in updates and r != updates[seq]:
            raise ValueError("Conflicting versions for " + r["event_id"])
        updates[seq] = r

    cases = []
    for event, updates in versions.items():
        if 0 not in updates:
            continue
        first = updates[0]
        issue = datetime.fromisoformat(first["date"]).replace(tzinfo=timezone.utc).astimezone(LOCAL)
        if not 8 <= issue.day <= 10:
            continue
        target = issue.replace(minute=0, second=0) + timedelta(hours=1)
        routes = set(first.get("affected", "").split(" | ")) - {""}
        complexes, unresolved, inactive = served(target, routes)
        prior = [n for n,r in updates.items() if n and
                 datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc) < target]
        if prior and set(range(max(prior)+1)) != {0, *prior}:
            raise ValueError("Revision gap before target")
        cases.append((issue, event, target, first, complexes, prior, unresolved, inactive))

    count_query = {"$where": "agency='NYCT Subway' AND update_number=0 AND date >= '2026-08-08T04:00:00' AND date < '2026-08-11T04:00:00'",
                   "$select": "count(*) as first_notices,count(distinct event_id) as events"}
    count, = read("first_count.json", api("7kct-peq7", **count_query))
    if len(cases) != int(count["events"]) or len(cases) != int(count["first_notices"]):
        raise ValueError("Archive event count differs from independent count query")
    hours = {case[2] for case in cases}
    def rail_hour(hour):
        q = {"transit_timestamp": hour.strftime("%Y-%m-%dT%H:00:00"), "transit_mode": "subway",
             "$select": "station_complex_id,sum(ridership) as entries,count(*) as categories",
             "$group": "station_complex_id", "$order":"station_complex_id", "$limit":1000}
        rows = read(hour.strftime("rail-%Y%m%d-%H.json"), api("5wq4-mkjj", **q))
        if len(rows) >= 1000 or len({r["station_complex_id"] for r in rows}) != len(rows):
            raise ValueError("Incomplete or duplicate subway aggregate")
        return hour, {r["station_complex_id"]: int(float(r["entries"])) for r in rows}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rail = dict(pool.map(rail_hour, sorted(hours)))
    bike_path = inputs(cache, manifest, "aug.zip", sample_sources["aug.zip"]["url"])
    fmt = "%Y-%m-%d %H"
    positions, bike, duplicates = read_bikes(bike_path, [h.strftime(fmt) for h in hours])
    docks = defaultdict(list)
    for dock, counts in positions.items():
        coord = min(counts, key=lambda pt: (-counts[pt], pt))
        metres, cid = min((distance(coord, pt), int(cid)) for cid, pt in places.items())
        if metres <= 500:
            docks[str(cid)].append(dock)

    with (ROOT / "cohort/role_pairs.csv").open() as src:
        roles = {r["event_id"]: r for r in csv.DictReader(src)}
    events, panel = [], {}
    for issue,event,h,first,complexes,prior,unresolved,inactive in sorted(cases):
        text = first["header"] + "\n" + first.get("description", "")
        role = roles.get(event)
        if role and any(role[col] not in text for col in ("station_a","station_b")):
            raise ValueError("Role evidence missing in original notice")
        event_both = retained = 0
        for cid in sorted(complexes, key=int):
            key = h, cid
            b = bike[h.strftime(fmt)]
            entries = rail[h].get(cid)
            departures = sum(b[d] for d in docks[cid]) if docks[cid] else None
            event_both += entries is not None and departures is not None
            if key in panel:
                panel[key]["event_count"] += 1
                continue
            retained += 1
            panel[key] = dict(first_event=event, issue_local=issue.isoformat(),
                              hour_local=h.isoformat(), complex_id=cid, event_count=1,
                              entries=entries, docks=len(docks[cid]), departures=departures)
        events.append(dict(event_id=event, alert_id=first["alert_id"], issue_local=issue.isoformat(),
                           target_local=h.isoformat(), routes=first.get("affected",""), status=first["status_label"],
                           revision_before_hour=int(bool(prior)), supported_role_pair=int(bool(role)),
                           both_role_complexes_on_map=int({role["complex_a"],role["complex_b"]} <= complexes) if role else None,
                           planned_reminder=int("planned work reminder" in text.lower()),
                           cause_screen=cause_screen(text), routes_without_calls="|".join(sorted(inactive)),
                           unmapped_stops="|".join(sorted(unresolved)), complexes=len(complexes),
                           both_targets=event_both, retained_after_overlap=retained))
    for e in events:
        role = roles.get(e["event_id"])
        e["pair_retained_both_modes"] = None
        if role:
            hour = datetime.fromisoformat(e["target_local"])
            pair = [panel.get((hour, role[k])) for k in ("complex_a", "complex_b")]
            e["pair_retained_both_modes"] = int(all(
                r and r["first_event"] == e["event_id"] and r["entries"] is not None
                and r["departures"] is not None for r in pair))
    if not roles.keys() <= {e["event_id"] for e in events}:
        raise ValueError("Annotated event missing")
    write_csv(out/"events.csv", events)
    write_csv(out/"station_hours.csv", [panel[key] for key in sorted(panel)])
    (out/"sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cache, args.output)
