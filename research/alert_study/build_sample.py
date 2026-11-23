"""Rebuild the five NYC development traces. Python 3.9+, standard library."""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# One named location per first notice. Fixed for tracing, not model eligibility.
FOCAL = {
    "262272": ("623", "Canal St", "express segment start"),
    "262324": ("616", "74 St-Broadway", "express segment end"),
    "262344": ("169", "Canal St", "express segment start"),
    "262675": ("243", "Church Av", "express segment end"),
    "262722": ("477", "72 St", "reported brake activation"),
}
LOCAL = ZoneInfo("America/New_York")
BASE = "https://data.ny.gov/resource/"


def api(dataset, **params):
    return BASE + dataset + ".json?" + urllib.parse.urlencode(params)


def inputs(cache, manifest, name, url):
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=180) as src, path.with_suffix(".part").open("wb") as dest:
            shutil.copyfileobj(src, dest)
        path.with_suffix(".part").replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
    manifest[name] = dict(url=url, bytes=path.stat().st_size, sha256=digest.hexdigest())
    return path


def distance(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    v = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742000 * math.asin(math.sqrt(v))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_bikes(path, hours):
    # Only use geography before the first incident. Do not aggregate other outcome hours.
    positions = defaultdict(Counter)
    starts = {h: Counter() for h in hours}
    seen = {}
    duplicates = Counter()
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".csv") or name.startswith("__MACOSX/"):
                continue
            with archive.open(name) as src:
                for row in csv.DictReader(io.TextIOWrapper(src, encoding="utf-8-sig")):
                    ts, dock = row["started_at"], row["start_station_id"].strip()
                    if "2026-08-01" <= ts[:10] < "2026-08-08" and dock:
                        if row["start_lat"] and row["start_lng"]:
                            coords = tuple(round(float(row[k]), 4) for k in ("start_lat","start_lng"))
                            if 40 < coords[0] < 42 and -75 < coords[1] < -73:
                                positions[dock][coords] += 1
                    hour = ts[:13]
                    if hour in starts:
                        ride = row["ride_id"]
                        if not ride:
                            raise ValueError("Missing ride ID in selected hour")
                        if ride in seen:
                            if seen[ride] != (ts, dock):
                                raise ValueError("Conflicting repeated ride ID")
                            duplicates[hour] += 1
                        else:
                            seen[ride] = ts, dock
                            starts[hour][dock] += 1
    return positions, starts, duplicates


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest, places = {}, {}
    def rows(name, url):
        return json.loads(inputs(cache, manifest, name, url).read_bytes())
    for h in ("08", "17"):
        q = {"$select":"distinct station_complex_id,station_complex,latitude,longitude",
             "transit_mode":"subway","transit_timestamp":f"2026-08-06T{h}:00:00",
             "$order":"station_complex_id,station_complex,latitude,longitude","$limit":5000}
        for row in rows(h, api("5wq4-mkjj", **q)):
            cid = row["station_complex_id"]
            point = row["station_complex"], (float(row["latitude"]),float(row["longitude"]))
            if cid in places and places[cid] != point:
                raise ValueError(f"Conflicting complex geometry: {cid}")
            places[cid] = point

    clock_row, = rows("clock538013", api("7kct-peq7", alert_id="538013"))
    assert clock_row["date"].startswith("2026-08-11T19:46")
    assert "as of 3:45pm" in clock_row["description"] and "4:30pm" in clock_row["description"]

    traces, versions, hours = [], [], {}
    for event,(cid,mention,role) in FOCAL.items():
        records = sorted(rows(event, api("7kct-peq7", event_id=event)), key=lambda r:int(r["update_number"]))
        assert [int(r["update_number"]) for r in records] == list(range(len(records)))
        assert all(r["event_id"] == event and r["agency"] == "NYCT Subway" for r in records)
        first = records[0]
        assert mention in first["header"] and mention in places[cid][0]
        assert set(first["affected"].split(" | ")).issubset(places[cid][0].rsplit("(",1)[1].rstrip(")").split(","))
        for r in records:
            v = {k:r[k] for k in ("event_id","alert_id","update_number","date","affected","status_label")}
            v["focal_name_in_header"] = mention in r["header"]
            versions.append(v)
        issue = datetime.fromisoformat(first["date"]).replace(tzinfo=timezone.utc).astimezone(LOCAL)
        start = issue.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        hours[event] = start.strftime("%Y-%m-%d %H")
        params = dict(transit_mode="subway", transit_timestamp=start.strftime("%Y-%m-%dT%H:%M:%S"),
                      station_complex_id=cid, **{"$order":"payment_method,fare_class_category","$limit":1000})
        rail = rows("rail"+event, api("5wq4-mkjj", **params))
        keys = {(r["payment_method"],r["fare_class_category"]) for r in rail}
        assert len(keys) == len(rail) < 1000
        assert all(r["station_complex_id"] == cid and datetime.fromisoformat(r["transit_timestamp"]) == start.replace(tzinfo=None) for r in rail)
        traces.append(dict(event_id=event,first_alert=first["alert_id"],issue_local=issue.isoformat(),
                           target_start_local=start.isoformat(),target_end_local=(start+timedelta(hours=1)).isoformat(),
                           routes=first["affected"],mention=mention,role=role,complex_id=cid,
                           station_complex=places[cid][0],fare_rows=len(rail),
                           reported_entries=int(sum(float(r["ridership"]) for r in rail)) if rail else None))

    path = inputs(cache, manifest, "aug.zip","https://s3.amazonaws.com/tripdata/202608-citibike-tripdata.zip")
    positions, starts, duplicates = read_bikes(path, hours.values())
    assignments = {}
    for dock,c in positions.items():
        # Most common rounded start position; lexicographic coordinate tie break.
        point = min(c, key=lambda pt:(-c[pt],pt))
        metres, numeric_id = min((distance(point, geo),int(cid)) for cid,(_,geo) in places.items())
        if metres <= 500:
            assignments[dock] = str(numeric_id), metres, point
    docks = []
    for t in traces:
        hour = hours[t["event_id"]]
        counts = starts[hour]
        local = [dock for dock,v in assignments.items() if v[0] == t["complex_id"]]
        for dock in sorted(local):
            _, metres, point = assignments[dock]
            docks.append(dict(event_id=t["event_id"],complex_id=t["complex_id"],dock_id=dock,
                              lat=point[0],lon=point[1],distance_m=round(metres,2),
                              observed_departures=counts[dock]))
        t.update(docks=len(local), recorded_bike_departures=sum(counts[d] for d in local) if local else None,
                 city_file_departures=sum(counts.values()), city_missing_dock_id=counts[""],
                 city_unknown_dock=sum(n for d,n in counts.items() if d and d not in positions),
                 city_outside_radius=sum(n for d,n in counts.items() if d in positions and d not in assignments),
                 duplicate_target_rows=duplicates[hour])
    write_csv(out/"traces.csv",traces)
    write_csv(out/"versions.csv",versions)
    write_csv(out/"docks.csv",docks)
    summary = dict(preperiod_complexes=len(places),preperiod_docks=len(positions),
                   docks_within_500m=len(assignments),sources=manifest)
    (out/"manifest.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.cache, args.output)
