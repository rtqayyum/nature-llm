"""Preserve the source projections for Boston development checks from the pinned full downloads."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from check_sample import HERE,HEAVY,NY

ALERT_COLUMNS=["id","created_timestamp","last_modified_timestamp","last_push_notification_timestamp",
    "closed_timestamp","effect","severity","informed_entity.route_type","informed_entity.direction_id","informed_entity.activities","informed_entity.route_id","informed_entity.stop_id","informed_entity.facility_id",
    "header_text.translation.text","description_text.translation.text"]
STOP_COLUMNS=["stop_id","stop_name","stop_lat","stop_lon","parent_station","location_type","vehicle_type","gtfs_active_date","gtfs_end_date"]


def prepare(cache,out):
    manifest=json.loads((HERE/"sources.json").read_text())
    for name in ("alerts.parquet","stops.parquet","gates.zip"):
        if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=manifest[name]["sha256"]:
            raise ValueError(f"Different source capture: {name}")
    out.mkdir(parents=True,exist_ok=True)
    begin,end=[int(datetime(2025,m,1,tzinfo=NY).timestamp()) for m in (5,6)]
    table=pq.read_table(cache/"alerts.parquet",columns=ALERT_COLUMNS,filters=[("created_timestamp",">=",begin),("created_timestamp","<",end)])
    events=defaultdict(list)
    for r in table.to_pylist():
        events[r["id"]].append(r)
    with (HERE/"month_screen/review.csv").open() as f:
        immediate={int(r["alert_id"]) for r in csv.DictReader(f) if r["timing"]=="immediate"}
    keep=[]
    for rows in events.values():
        ts=min(r["last_modified_timestamp"] for r in rows)
        first=[r for r in rows if r["last_modified_timestamp"]==ts]
        check_day=datetime.fromtimestamp(first[0]["created_timestamp"],NY).day==14
        heavy=any(r["informed_entity.route_id"] in HEAVY for r in first)
        if not (check_day or heavy):
            continue
        service=heavy and not all(r["effect"]=="ACCESSIBILITY_ISSUE" or r["informed_entity.facility_id"] for r in first)
        for r in (rows if check_day or first[0]["id"] in immediate else first):
            if not service:
                r["header_text.translation.text"]=r["description_text.translation.text"]=None
            keep.append(r)
    pq.write_table(pa.Table.from_pylist(keep,schema=table.schema),out/"alerts.parquet",compression="zstd")
    table=pq.read_table(cache/"stops.parquet",columns=STOP_COLUMNS)
    keep=[r for r in table.to_pylist() if (r["location_type"]==1 or (r["location_type"]==0 and r["vehicle_type"] in (0,1)))
          and any(r["gtfs_active_date"]<=d<=r["gtfs_end_date"] for d in (20250514,20250520,20250522))]
    pq.write_table(pa.Table.from_pylist(keep,schema=table.schema),out/"stops.parquet",compression="zstd")
    # Keep every station in the selected hourly windows, before checking coverage.
    hours=set()
    for day,hour_list in ((14,(6,7,20)),(20,(13,)),(22,(19,21,22))):
        for hour in hour_list:
            for lag in (0,7):
                h=datetime(2025,5,day,hour)-timedelta(days=lag)
                hours.add((str(h.date()),h.hour))
    member="GSE_by_year/GSE_2025.csv"
    with zipfile.ZipFile(cache/"gates.zip") as archive,archive.open(member) as f:
        reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig"))
        buf=io.StringIO()
        writer=csv.DictWriter(buf,fieldnames=reader.fieldnames,lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if (row["service_date"],int(row["time_period"].strip("()").split(":")[0])) in hours:
                writer.writerow(row)
    with zipfile.ZipFile(out/"gates.zip","w") as archive:
        archive.writestr(zipfile.ZipInfo(member),buf.getvalue(),compress_type=zipfile.ZIP_DEFLATED)
    hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ("alerts.parquet","stops.parquet","gates.zip")}
    (out/"hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    prepare(a.cache,a.output)
