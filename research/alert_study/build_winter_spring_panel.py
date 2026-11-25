"""Build the January–April NYC development panel from dated public sources."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from audit_cohort import ROOT, timetable
from audit_february import SCHEDULE, WINTER_SHA
from build_sample import LOCAL, api, distance, inputs, write_csv
from extract_alert_roles import PERIOD

START, STOP = date(2026, 1, 1), date(2026, 5, 1)
MONTHS = [date(2026, i, 1) for i in range(1, 6)]
SPRING = "https://web.archive.org/web/20260227082256id_/https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
SPRING_SHA = "f51d0e0444d677217e631b0fd1744414c2185207"


def scan_trips(path, month, next_month):
    counts, city, positions = defaultdict(Counter), Counter(), defaultdict(Counter)
    seen = {}
    duplicates = 0
    prefix = str(month)[:7]
    last_week = str(next_month-timedelta(days=7))
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith(".csv") or member.startswith("__MACOSX/"):
                continue
            with archive.open(member) as source:
                records = csv.reader(io.TextIOWrapper(source, encoding="utf-8-sig"))
                header = next(records)
                rid, start, station, lat, lon = map(header.index, (
                    "ride_id", "started_at", "start_station_id", "start_lat", "start_lng"))
                for row in records:
                    ts, dock = row[start], row[station].strip()
                    if not ts.startswith(prefix) or (month < START and ts < last_week):
                        continue
                    key = row[rid]
                    if not key:
                        raise ValueError("Missing bike ride ID")
                    if key in seen:
                        if seen[key] != (ts, dock):
                            raise ValueError("Conflicting duplicate bike start")
                        duplicates += 1
                        continue
                    seen[key] = ts, dock
                    hour = ts[:13].replace("T"," ")
                    city[hour] += 1
                    if dock:
                        counts[hour][dock] += 1
                    if next_month < STOP and ts >= last_week and dock and row[lat] and row[lon]:
                        point = (round(float(row[lat]),4), round(float(row[lon]),4))
                        if 40 < point[0] < 42 and -75 < point[1] < -73:
                            positions[dock][point] += 1
    return counts, city, positions, len(seen), duplicates


def prepare_bikes(source, places, out):
    values, city, mapped = {}, {}, {}
    maps, inventory = [], []
    previous_counts = previous_positions = None
    boundaries = [date(2025,12,1), *MONTHS]
    for month, next_month in zip(boundaries, boundaries[1:]):
        code = month.strftime("%Y%m")
        path = source(code+".zip", f"https://s3.amazonaws.com/tripdata/{code}-citibike-tripdata.zip")
        counts, total, positions, rides, duplicates = scan_trips(path, month, next_month)
        city.update(total)
        inventory.append(dict(month=str(month)[:7],unique_starts=rides,duplicates=duplicates))
        if month >= START:
            assignment = {}
            for dock, observations in sorted(previous_positions.items()):
                coords = min(observations, key=lambda pt: (-observations[pt],pt))
                metres, cid = min((distance(coords,xy),int(c)) for c,xy in places[month].items())
                if metres <= 500:
                    assignment[dock] = str(cid)
                    maps.append(dict(month=str(month)[:7],dock=dock,complex_id=cid,
                                     lat=coords[0],lon=coords[1],distance_m=round(metres,3)))
            mapped[month] = set(assignment.values())
            # Apply the target month's same fixed catchment to its target AND weekly lag.
            for hours in (previous_counts, counts):
                for hour, docks in hours.items():
                    result = Counter()
                    for dock, n in docks.items():
                        if dock in assignment:
                            result[assignment[dock]] += n
                    values[month,hour] = result
        last_week = str(next_month-timedelta(days=7))
        previous_counts = {h:v for h,v in counts.items() if h >= last_week}
        previous_positions = positions
    write_csv(out/"docks.csv",maps)
    write_csv(out/"bike_inventory.csv",inventory)
    return values, city, mapped


def write_panel(path, rows):
    with path.open("wb") as raw, gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
        with io.TextIOWrapper(gz,encoding="utf-8",newline="") as f:
            w = csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def run(cache, out):
    cache.mkdir(exist_ok=True,parents=True)
    out.mkdir(exist_ok=True,parents=True)
    manifest = {}
    expected = ROOT/"winter_spring/sources.json"
    pinned = json.loads(expected.read_text()) if expected.exists() else {}
    def source(name,url):
        path = inputs(cache, manifest, name, url)
        if pinned and pinned.get(name) != manifest[name]:
            raise ValueError(f"Source changed: {name}")
        return path
    def records(name,url):
        return json.loads(source(name,url).read_bytes())
    stations = records("stations",api("39hk-dx4f"))
    places, schedules = {}, {}
    for month in MONTHS[:-1]:
        last = month-timedelta(days=1)
        wed = last-timedelta(days=(last.weekday()-2)%7)
        coords = {}
        for h in ("08","17"):
            query = {"$select":"distinct station_complex_id,station_complex,latitude,longitude",
                     "transit_mode":"subway","transit_timestamp":f"{wed}T{h}:00:00",
                     "$order":"station_complex_id,station_complex,latitude,longitude","$limit":5000}
            rows = records(f"geo-{wed}-{h}.json",api("5wq4-mkjj",**query))
            for r in rows:
                cid = r["station_complex_id"]
                point = float(r["latitude"]),float(r["longitude"])
                if cid in coords and coords[cid] != point:
                    raise ValueError(f"Conflicting coordinates for {cid}")
                coords[cid] = point
        places[month] = coords
        spring = month.month >= 3
        name, url, digest = (("20260227082256.zip",SPRING,SPRING_SHA) if spring else
                             ("20260225205519.zip",SCHEDULE,WINTER_SHA))
        path = source(name,url)
        if hashlib.sha1(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Different base timetable")
        schedules[month] = timetable(path,stations,coords)
        if month.month == 3:
            prior_service = timetable(cache/"20260225205519.zip", stations, coords)

    bike, city, mapped = prepare_bikes(source,places,out)
    def rail_day(d):
        query = {"$where":f"transit_mode='subway' AND transit_timestamp >= '{d}T00:00:00' AND transit_timestamp < '{d+timedelta(days=1)}T00:00:00'",
                 "$select":"transit_timestamp,station_complex_id,sum(ridership) as entries,count(*) as categories",
                 "$group":"transit_timestamp,station_complex_id","$order":"transit_timestamp,station_complex_id","$limit":15000}
        rows = records(str(d)+".json",api("5wq4-mkjj",**query))
        lookup = {(r["transit_timestamp"][:13].replace("T"," "),r["station_complex_id"]):int(Decimal(r["entries"])) for r in rows}
        if len(rows) >= 15000 or len(rows) != len(lookup) or not rows:
            raise ValueError(f"Incomplete or duplicate rail keys on {d}")
        return lookup
    begin = START-timedelta(days=7)
    rail = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(rail_day,(begin+timedelta(days=i) for i in range((STOP-begin).days))):
            rail.update(rows)

    versions = defaultdict(dict)
    for start,end in zip(PERIOD,PERIOD[1:]):
        q = {"$where":f"agency='NYCT Subway' AND date >= '{start}' AND date < '{end}'",
             "$order":"date,event_id,update_number,alert_id","$limit":50000}
        rows = records("versions-"+start[:7]+".json",api("7kct-peq7",**q))
        if len(rows)>=q["$limit"]:
            raise ValueError("Truncated version history")
        for r in rows:
            v = versions[r["event_id"]]
            n = int(r["update_number"])
            if n in v and v[n] != r:
                raise ValueError("Conflicting alert versions")
            v[n] = r
    with (ROOT/"role_screen/events.csv").open() as f:
        events = sorted(csv.DictReader(f),key=lambda r:(r["issue_local"][:10],datetime.fromisoformat(r["issue_local"]).timestamp(),int(r["event_id"])))
    if {e["event_id"] for e in events} != {e for e,v in versions.items() if 0 in v}:
        raise ValueError("Different incident inventory")
    roles = defaultdict(lambda:defaultdict(set))
    with (ROOT/"role_screen/mentions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["complex_id"]:
                roles[r["event_id"]][r["complex_id"]].add(r["role"])

    panel, event_rows, version_rows = {}, [], []
    for e in events:
        eid = e["event_id"]
        issue = datetime.fromisoformat(e["issue_local"]).astimezone(timezone.utc)
        target = (issue.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)).astimezone(LOCAL)
        month = target.date().replace(day=1)
        history = versions[eid]
        if history[0]["alert_id"]!=e["alert_id"] or datetime.fromisoformat(history[0]["date"]).replace(tzinfo=timezone.utc)!=issue:
            raise ValueError("Different first notice")
        for n,r in sorted(history.items()):
            version_rows.append(dict(event_id=eid,update_number=n,alert_id=r["alert_id"],
                                     published_utc=r["date"]+"Z",used_as_input=int(n==0)))
        item = dict(event_id=eid,alert_id=e["alert_id"],issue_local=e["issue_local"],hour=target.isoformat(),
                    routes=e["routes"],versions_in_window=len(history),mapped_complexes=0,unmapped_stops="",inactive_routes="")
        event_rows.append(item)
        if month>=STOP:
            continue
        routes = set(e["routes"].split(" | "))- {""}
        complexes,unmapped,inactive = schedules[month](target,routes)
        if target.date() == date(2026,3,1):
            previous, unknown, missed = prior_service(target,routes,service_before=month)
            complexes |= previous
            unmapped |= unknown
            inactive &= missed
        item.update(mapped_complexes=len(complexes),unmapped_stops="|".join(sorted(unmapped)),inactive_routes="|".join(sorted(inactive)))
        h = target.strftime("%Y-%m-%d %H")
        lag = (target-timedelta(days=7)).strftime("%Y-%m-%d %H")
        for cid in complexes:
            key = (target.isoformat(),cid)
            if key in panel:
                panel[key]["overlap"] += 1
                continue
            def bike_value(hour):
                if cid in mapped[month] and city.get(hour,0):
                    return bike[month,hour][cid]
                return None
            panel[key] = dict(hour=key[0],complex_id=cid,event_id=eid,overlap=1,rail=rail.get((h,cid)),
                              bike=bike_value(h),weekly_rail=rail.get((lag,cid)),weekly_bike=bike_value(lag),
                              roles="|".join(sorted(roles[eid].get(cid,set()))))
    write_csv(out/"events.csv",event_rows)
    write_panel(out/"versions.csv.gz",version_rows)
    for m in MONTHS[:-1]:
        prefix = str(m)[:7]
        rows = [panel[k] for k in sorted(panel,key=lambda pair:(pair[0],int(pair[1]))) if k[0].startswith(prefix)]
        write_panel(out/(prefix+".csv.gz"),rows)
    (out/"sources.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    run(args.cache,args.output)
