"""Reproduce the February 2026 NYC development cohort and weekly-naive baseline."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from audit_cohort import ROOT, cause_screen, timetable
from build_sample import LOCAL, api, distance, inputs, write_csv

WHEN = "date >= '2026-02-01T05:00:00' AND date < '2026-03-01T06:00:00'"
SCHEDULE = "https://web.archive.org/web/20260225205519id_/http://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
WINTER_SHA = "b3b3f88db7d48efc5c9ab52e560efca52bccc49f"
CUES = (r"\b(board|boarding|arrive and depart|transfer at)\b",
        r"\b(last stop|end at|ending at|terminate at|begin/end)\b",
        r"\b(bypassing|skipping|bypass|skip)\b",
        r"\b(from|between)\b.{2,100}?\b(to|and)\b")


def clock(row):
    return datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc).astimezone(LOCAL)


def calendar_week(day):
    return 1 + (day.day - 1) // 7


def notice(row):
    full = row["header"] + " " + row.get("description", "")
    active = re.split("planned (?:service|work) reminder", full, flags=re.I)[0]
    return " ".join(active.split())


def read_rides(path):
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith(".csv") and not member.startswith("__MACOSX/"):
                with archive.open(member) as src:
                    yield from csv.DictReader(io.TextIOWrapper(src, encoding="utf-8-sig"))


def bike_counts(january, february, places):
    positions = {k:defaultdict(Counter) for k in ("Jan 18-24", "Jan 25-31", "Jan 1-31")}
    for r in read_rides(january):
        dock = r["start_station_id"].strip()
        if not dock or not r["started_at"].startswith("2026-01-"):
            continue
        if r["start_lat"] and r["start_lng"]:
            xy = tuple(round(float(r[k]), 4) for k in ("start_lat", "start_lng"))
            if 40 < xy[0] < 42 and -75 < xy[1] < -73:
                positions["Jan 1-31"][dock][xy] += 1
                day = int(r["started_at"][8:10])
                if day >= 18:
                    window = "Jan 25-31" if day >= 25 else "Jan 18-24"
                    positions[window][dock][xy] += 1
    maps = {}
    for window, locations in positions.items():
        assigned = {}
        for dock, counts in locations.items():
            coord = min(counts, key=lambda xy: (-counts[xy], xy))
            metres, cid = min((distance(coord, xy), int(cid)) for cid, xy in places.items())
            if metres <= 500:
                assigned[dock] = str(cid)
        maps[window] = assigned
    assignments = maps["Jan 25-31"]
    result = Counter()
    city = Counter()
    seen = {}
    for r in read_rides(february):
        start = r["started_at"]
        if not start.startswith("2026-02-"):
            continue
        ride, dock = r["ride_id"], r["start_station_id"].strip()
        if not ride:
            raise ValueError("Missing ride ID")
        if ride in seen:
            if seen[ride] != (start, dock):
                raise ValueError("Conflicting ride ID " + ride)
            continue
        seen[ride] = start, dock
        hour = start[:13]
        city[hour] += 1
        if dock in assignments:
            result[hour, assignments[dock]] += 1
    return result, Counter(assignments.values()), city, maps


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def records(filename, url):
        return json.loads(inputs(cache, sources, filename, url).read_bytes())

    q = {"$where": "agency='NYCT Subway' AND " + WHEN,
         "$order": "date,event_id,update_number,alert_id", "$limit": 50000}
    raw = records("alerts.json", api("7kct-peq7", **q))
    assert len(raw) < q["$limit"]
    versions = defaultdict(dict)
    for r in raw:
        v = versions[r["event_id"]]
        n = int(r["update_number"])
        assert n not in v or v[n] == r
        v[n] = r
    firsts = sorted((clock(v[0]), event, v[0]) for event, v in versions.items()
                    if 0 in v and clock(v[0]).month == 2)
    inventory = json.loads((ROOT / "cohort/february_inventory.json").read_text())
    assert len(firsts) == sum(int(r["first_rows"]) for r in inventory["by_native_status"])
    review = {r["event_id"]: r for r in csv.DictReader((ROOT/"february/review.csv").read_text().splitlines())}
    flagged = {event for _,event,r in firsts if sum(bool(re.search(p, notice(r), re.I)) for p in CUES) >= 2}
    sampled = set()
    for week in range(1,5):
        negative = [event for issue,event,_ in firsts if event not in flagged and calendar_week(issue) == week]
        sampled.update(sorted(negative, key=lambda e: hashlib.sha256(("feb-screen-misses-v1/"+e).encode()).digest())[:20])
    assert flagged <= review.keys() <= flagged | sampled

    schedule = inputs(cache, sources, "20260225205519.zip", SCHEDULE)
    assert hashlib.sha1(schedule.read_bytes()).hexdigest() == WINTER_SHA
    station_rows = records("stations", api("39hk-dx4f"))
    places = {}
    for h in ("08","17"):
        q = {"$select":"distinct station_complex_id,station_complex,latitude,longitude", "transit_mode":"subway",
             "transit_timestamp":f"2026-01-28T{h}:00:00","$order":"station_complex_id,station_complex,latitude,longitude","$limit":5000}
        for r in records("geo"+h+".json", api("5wq4-mkjj", **q)):
            cid = r["station_complex_id"]
            coords = float(r["latitude"]),float(r["longitude"])
            assert cid not in places or places[cid] == coords
            places[cid] = coords
    served = timetable(schedule, station_rows, places)
    archives = [inputs(cache, sources, f"bike{m}.zip", f"https://s3.amazonaws.com/tripdata/2026{m}-citibike-tripdata.zip") for m in ("01","02")]
    bike, docks, city, maps = bike_counts(*archives, places)

    def day_rail(i):
        day = date(2026,2,i)
        end = day + timedelta(days=1)
        q = {"$where":f"transit_mode='subway' AND transit_timestamp >= '{day}T00:00:00' AND transit_timestamp < '{end}T00:00:00'",
             "$select":"transit_timestamp,station_complex_id,sum(ridership) as entries,count(*) as categories",
             "$group":"transit_timestamp,station_complex_id","$order":"transit_timestamp,station_complex_id", "$limit":15000}
        rows = records(str(day)+".json", api("5wq4-mkjj", **q))
        result = {(r["transit_timestamp"][:13].replace("T"," "), r["station_complex_id"]):int(float(r["entries"])) for r in rows}
        assert len(result) == len(rows) < q["$limit"]
        return result

    rail = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for day in pool.map(day_rail, range(1,29)):
            rail.update(day)

    def values(h, cid):
        key = h.strftime("%Y-%m-%d %H")
        return rail.get((key,cid)), bike[key,cid] if docks[cid] and city[key] else None

    events, panel, prior, episode = {}, {}, {}, {}
    group_gaps = (6,24,72)
    for issue, event, r in firsts:
        target = issue.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        routes = set(r.get("affected","").split(" | ")) - {""}
        key = tuple(sorted(routes)), " ".join(r["header"].lower().split())
        for gap in group_gaps:
            if key not in prior or issue - prior[key] > timedelta(hours=gap):
                episode[key,gap] = event
        prior[key] = issue
        row = dict(event_id=event,alert_id=r["alert_id"],issue_local=issue.isoformat(),target_local=target.isoformat(),
                   week=calendar_week(issue),routes=r.get("affected",""),status=r["status_label"],cause=cause_screen(notice(r)),
                   revision_before_target=int(any(n and clock(v)<target for n,v in versions[event].items())),
                   **{f"episode_{g}h":episode[key,g] for g in group_gaps})
        events[event] = row
        rev = review.get(event)
        pair = rev is not None and bool(rev["complex_a"])
        row.update(review="cue" if event in flagged else "sample" if event in sampled else "unreviewed",
                   confirmed_pair=int(pair) if event in flagged | sampled else None)
        crossing = target.month != 2
        cs, unmapped, inactive = (set(),set(),set()) if crossing else served(target, routes)
        row.update(purged_month_edge=int(crossing), complexes=len(cs), unmapped="|".join(sorted(unmapped)),
                   inactive_routes="|".join(sorted(inactive)), retained=0, pair_on_map=None, pair_both_counts=None)
        for cid in sorted(cs,key=int):
            k = target,cid
            if k in panel:
                panel[k]["n_events"] += 1
                continue
            rail_y,bike_y = values(target,cid)
            rail_lag,bike_lag = values(target-timedelta(days=7),cid)
            row["retained"] += 1
            panel[k] = dict(hour=target.isoformat(),week=calendar_week(target),complex_id=cid,first_event=event,n_events=1,
                            rail=rail_y,bike=bike_y,weekly_rail=rail_lag,weekly_bike=bike_lag)
        if pair:
            assert all(rev[k] in notice(r) for k in ("station_a","station_b"))
            ids = {rev["complex_a"],rev["complex_b"]}
            assert len(ids)==2
            row["pair_on_map"] = int(ids <= cs)
            row["pair_both_counts"] = int(ids <= cs and all(panel[target,c]["first_event"]==event and
                        panel[target,c]["rail"] is not None and panel[target,c]["bike"] is not None for c in ids))
    write_csv(out/"events.csv", list(events.values()))
    ordered = [panel[k] for k in sorted(panel)]
    with (out/"station_hours.csv.gz").open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file,filename="",mode="wb",mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=ordered[0].keys(),lineterminator="\n")
                w.writeheader()
                w.writerows(ordered)
    sensitivity = []
    for window, mapping in sorted(maps.items()):
        covered = set(mapping.values())
        available = {(h,c) for (h,c),r in panel.items()
                     if r["rail"] is not None and c in covered and city[h.strftime("%Y-%m-%d %H")]}
        pairs = 0
        for eid, role in review.items():
            if not role["complex_a"]:
                continue
            h = datetime.fromisoformat(events[eid]["target_local"])
            keys = [(h, role[k]) for k in ("complex_a","complex_b")]
            pairs += all(key in available and panel[key]["first_event"]==eid for key in keys)
        sensitivity.append(dict(january_map=window,assigned_docks=len(mapping),covered_complexes=len(covered),
                                panel_both_counts=len(available),retained_pairs=pairs))
    write_csv(out/"dock_sensitivity.csv", sensitivity)
    scores = []
    for week in (2,3,4,0):
        matched = [r for r in ordered if (not week or r["week"]==week) and
                   all(r[k] is not None for k in ("rail","bike","weekly_rail","weekly_bike"))]
        for mode in ("rail","bike"):
            error = sum(abs(r[mode]-r["weekly_"+mode]) for r in matched)
            scores.append(dict(target_week=week or "pooled",mode=mode,n=len(matched),absolute_error=error,mae=error/len(matched)))
    write_csv(out/"weekly_naive.csv", scores)
    (out/"sources.json").write_text(json.dumps(sources,sort_keys=True,indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.cache,a.output)
