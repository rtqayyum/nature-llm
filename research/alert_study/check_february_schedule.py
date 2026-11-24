"""Compare reviewed station roles against the MTA's effective schedule archive."""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from audit_february import ROOT, inputs, api, write_csv


def check(cache, out):
    review = ROOT / "february/review.csv"
    roles = {r["event_id"]:r for r in csv.DictReader(review.read_text().splitlines()) if r["complex_a"]}
    events = [r for r in csv.DictReader((out/"events.csv").read_text().splitlines()) if r["event_id"] in roles]
    manifest = {}
    stations = json.loads(inputs(cache, manifest, "stations", api("39hk-dx4f")).read_bytes())
    complexes = {r["gtfs_stop_id"]: r["complex_id"] for r in stations}

    def read_event(event):
        eid = event["event_id"]
        hour = datetime.fromisoformat(event["target_local"])
        start = hour.strftime("%Y-%m-%dT%H:00:00")
        end = (hour+timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00")
        role = roles[eid]
        ids = {role["complex_a"],role["complex_b"]}
        stops = sorted(stop for stop,cid in complexes.items() if cid in ids)
        def quoted(values):
            return ",".join("'"+v.replace("'","''")+"'" for v in values)
        condition = (f"service_date between '{hour.date()-timedelta(days=1)}' and '{hour.date()+timedelta(days=1)}' "
                     f"AND gtfs_stop_id in ({quoted(stops)}) AND trip_line in ({quoted(event['routes'].split(' | '))}) "
                     f"AND revenue_service='1' AND trip_type='0' AND ((departure_time >= '{start}' AND departure_time < '{end}') "
                     f"OR (departure_time IS NULL AND arrival_time >= '{start}' AND arrival_time < '{end}'))")
        q = {"$where":condition,"$select":"gtfs_stop_id,trip_line,count(*) as calls",
             "$group":"gtfs_stop_id,trip_line","$order":"gtfs_stop_id,trip_line","$limit":1000}
        rows = json.loads(inputs(cache, manifest, "effective-"+eid+".json", api("g8es-h7gb", **q)).read_bytes())
        assert len(rows)<1000
        observed = {complexes[r["gtfs_stop_id"]] for r in rows}
        return dict(event_id=eid, hour=event["target_local"], complex_a=role["complex_a"],complex_b=role["complex_b"],
                    effective_a_calls=sum(int(r["calls"]) for r in rows if complexes[r["gtfs_stop_id"]]==role["complex_a"]),
                    effective_b_calls=sum(int(r["calls"]) for r in rows if complexes[r["gtfs_stop_id"]]==role["complex_b"]),
                    base_pair=int(event["pair_on_map"]), effective_pair=int(ids<=observed),
                    retained_base_pair=int(event["pair_both_counts"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(read_event, events))
    write_csv(out/"effective_schedule.csv",result)
    (out/"effective_sources.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    check(args.cache,args.output)
