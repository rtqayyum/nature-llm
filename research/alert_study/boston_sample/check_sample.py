"""Rebuild Boston's May 14, 2025 development traces from four pinned archives."""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_sample import distance, write_csv

NY = ZoneInfo("America/New_York")
START = datetime(2025,5,14,tzinfo=NY)
HEAVY = {"Red","Blue","Orange"}


def stops(path, date=START):
    stamp=int(date.strftime("%Y%m%d"))
    rows=pq.read_table(path, filters=[("gtfs_active_date","<=",stamp),("gtfs_end_date",">=",stamp)]).to_pylist()
    parents={r["parent_station"] for r in rows if r["location_type"]==0 and r["vehicle_type"] in (0,1)}
    geo={r["stop_id"]:(r["stop_name"],(r["stop_lat"],r["stop_lon"])) for r in rows
         if r["location_type"]==1 and r["stop_id"] in parents}
    return geo


def notices(path,geo):
    begin=int(START.timestamp())
    day=pq.read_table(path,filters=[("created_timestamp",">=",begin),("created_timestamp","<",begin+86400)]).to_pylist()
    by_id=defaultdict(list)
    for r in day:
        by_id[r["id"]].append(r)
    inventory,versions,traces=[],[],[]
    for eid,rows in sorted(by_id.items()):
        ordered=defaultdict(list)
        for r in rows:
            ordered[r["last_modified_timestamp"]].append(r)
        first=ordered[min(ordered)]
        routes={r["informed_entity.route_id"] for r in first}
        if not routes & HEAVY:
            continue
        facility=all(r["effect"]=="ACCESSIBILITY_ISSUE" or r["informed_entity.facility_id"] for r in first)
        inventory.append(dict(alert_id=eid,archive_rows=len(rows),modification_times=len(ordered),
                              facility_only=int(facility)))
        if facility:
            continue
        orig=first[0]
        if orig["last_modified_timestamp"]!=orig["created_timestamp"]:
            raise ValueError("Creation-version text unavailable")
        for ts,items in sorted(ordered.items()):
            content={(r["header_text.translation.text"],r["description_text.translation.text"],r["closed_timestamp"],
                      r["last_push_notification_timestamp"]) for r in items}
            if len(content)!=1:
                raise ValueError("Ambiguous version")
            header,desc,closed,push=content.pop()
            versions.append(dict(alert_id=eid,modified_local=datetime.fromtimestamp(ts,NY).isoformat(),
                push_local=datetime.fromtimestamp(push,NY).isoformat() if push else "",
                closed=int(closed is not None),rows=len(items),header=header,description=desc,
                routes="|".join(sorted({r["informed_entity.route_id"] for r in items if r["informed_entity.route_id"]})),
                stop_ids="|".join(sorted({r["informed_entity.stop_id"] for r in items if r["informed_entity.stop_id"]}))))
        text=orig["header_text.translation.text"]+" "+(orig["description_text.translation.text"] or "")
        named=[s for s,(name,_) in geo.items() if re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",text,re.I)]
        if len(named)!=1:
            raise ValueError("Trace expects one unambiguous named station on this development day")
        issue=datetime.fromtimestamp(orig["created_timestamp"],NY)
        hour=issue.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
        traces.append(dict(alert_id=eid,issue=issue.isoformat(),target=hour,complex_id=named[0]))
    return inventory,versions,traces,dict(all_feed_rows=len(day),all_created_ids=len(by_id))


def bikes(path,geo,traces,date=START):
    hours={t["target"]-timedelta(days=d) for t in traces for d in (0,7)}
    counts=Counter()
    points=defaultdict(Counter)
    city=Counter()
    seen={}
    with zipfile.ZipFile(path) as z:
        member,=[n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX/")]
        with z.open(member) as f:
            for r in csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig")):
                dt=datetime.fromisoformat(r["started_at"]).replace(tzinfo=NY)
                h=dt.replace(minute=0,second=0,microsecond=0)
                pre=date-timedelta(days=7)<=dt<date
                if not pre and h not in hours:
                    continue
                dock=r["start_station_id"].strip()
                key=r["ride_id"]
                identity=(r["started_at"],dock,r["start_lat"],r["start_lng"])
                if not key or (key in seen and seen[key]!=identity):
                    raise ValueError("Invalid or conflicting ride identity")
                if key in seen:
                    continue
                seen[key]=identity
                if h in hours:
                    city[h]+=1
                    counts[h,dock]+=1
                if pre and dock and r["start_lat"] and r["start_lng"]:
                    pt=tuple(round(float(r[k]),4) for k in ("start_lat","start_lng"))
                    if 41<pt[0]<43 and -72<pt[1]<-70:
                        points[dock][pt]+=1
    focal={t["complex_id"] for t in traces}
    docks=[]
    for dock,positions in sorted(points.items()):
        pt=min(positions,key=lambda p:(-positions[p],p))
        metres,s=min((distance(pt,center),s) for s,(_,center) in geo.items())
        if metres<=500 and s in focal:
            docks.append(dict(dock_id=dock,complex_id=s,lat=pt[0],lon=pt[1],distance_m=round(metres,2)))
    return docks,counts,city


def rail(path,traces):
    hours={(t["complex_id"],t["target"]-timedelta(days=d)) for t in traces for d in (0,7)}
    dates={str((h-timedelta(hours=3)).date()) for _,h in hours}
    rows=[]
    totals=defaultdict(dict)
    with zipfile.ZipFile(path) as z,z.open("GSE_by_year/GSE_2025.csv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig")):
            if r["service_date"] not in dates or r["stop_id"] not in {s for s,_ in hours}:
                continue
            ts=datetime.fromisoformat(r["service_date"]+"T"+r["time_period"].strip("()")).replace(tzinfo=NY)
            if ts.hour<3:
                ts+=timedelta(days=1)
            h=ts.replace(minute=0,second=0)
            s=r["stop_id"]
            if (s,h) not in hours:
                continue
            key=(ts.minute,r["route_or_line"])
            if key in totals[s,h] or ts.minute not in (0,30) or ts.second!=0:
                raise ValueError("Duplicate or unexpected gate interval")
            totals[s,h][key]=Decimal(r["gated_entries"])
            rows.append(dict(complex_id=s,interval=ts.isoformat(),line=r["route_or_line"],entries=r["gated_entries"]))
    return rows,totals


def verify(cache):
    sources=json.loads((HERE/"sources.json").read_text())
    projections=json.loads((HERE/"inputs/hashes.json").read_text())
    for name,meta in sources.items():
        h=hashlib.sha256()
        with (cache/name).open("rb") as f:
            for block in iter(lambda:f.read(2**20),b""):
                h.update(block)
        if h.hexdigest() not in {meta["sha256"],projections.get(name)}:
            raise ValueError(f"Different source snapshot: {name}")


def measure(cache,geo,traces,date=START):
    docks,starts,city=bikes(cache/"bikes.zip",geo,traces,date)
    half,gate=rail(cache/"gates.zip",traces)
    rows,contributions=[],[]
    for t in traces:
        s=t["complex_id"]
        assigned=[r["dock_id"] for r in docks if r["complex_id"]==s]
        item=dict(alert_id=t["alert_id"],issue=t["issue"],target=t["target"].isoformat(),complex_id=s,
                  station=geo[s][0],assigned_docks=len(assigned),gate_status="unknown")
        for prefix,days in (("",0),("weekly_",7)):
            h=t["target"]-timedelta(days=days)
            intervals=gate[s,h]
            lines0={l for m,l in intervals if m==0}
            lines30={l for m,l in intervals if m==30}
            item[prefix+"gate_entries"]=sum(intervals.values()) if lines0 and lines0==lines30 else None
            item[prefix+"bike_starts"]=sum(starts[h,d] for d in assigned) if assigned and city[h] else None
            for d in assigned:
                contributions.append(dict(alert_id=t["alert_id"],complex_id=s,hour=h.isoformat(),dock_id=d,starts=starts[h,d]))
        rows.append(item)
    return rows,docks,half,contributions


def run(cache,out):
    verify(cache)
    geo=stops(cache/"stops.parquet")
    inventory,versions,traces,summary=notices(cache/"alerts.parquet",geo)
    rows,docks,half,contributions=measure(cache,geo,traces)
    summary.update(rapid_transit_parents=len(geo),heavy_rail_ids=len(inventory),service_ids=len(rows),
                   service_versions=len(versions),paired_with_lags=sum(all(r[k] is not None for k in
                   ("gate_entries","bike_starts","weekly_gate_entries","weekly_bike_starts")) for r in rows))
    out.mkdir(parents=True,exist_ok=True)
    for name,table in (("inventory",inventory),("versions",versions),("docks",docks),("half_hours",half),
                       ("dock_hours",contributions),("traces",rows)):
        write_csv(out/(name+".csv"),table)
    (out/"coverage.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.cache,a.output)
