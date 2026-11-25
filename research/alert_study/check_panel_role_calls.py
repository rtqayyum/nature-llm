"""Audit effective revenue calls for all complexes in notices with unequal rule-derived role sets."""
import argparse
import csv
import json
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from pathlib import Path

from build_sample import LOCAL, api, inputs, write_csv
from audit_cohort import ROOT


def run(cache,out):
    manifest = {}
    expected = ROOT/"winter_spring/effective_sources.json"
    pinned = json.loads(expected.read_text()) if expected.exists() else {}
    def read(name,url):
        p = inputs(cache,manifest,name,url)
        if pinned and pinned.get(name)!=manifest[name]:
            raise ValueError("Effective timetable source drift: "+name)
        return json.loads(p.read_bytes())
    stations = {s["gtfs_stop_id"]:s["complex_id"] for s in read("stations",api("39hk-dx4f"))}
    roles = defaultdict(lambda:defaultdict(set))
    with (ROOT/"role_screen/mentions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["complex_id"]:
                roles[r["event_id"]][r["complex_id"]].add(r["role"])
    with (ROOT/"role_screen/events.csv").open() as f:
        events = [r for r in csv.DictReader(f) if len({frozenset(labels) for labels in roles[r["event_id"]].values()})>=2]
    def quoted(values):
        return ",".join("'"+v.replace("'","''")+"'" for v in values)
    def check(e):
        eid = e["event_id"]
        issue = datetime.fromisoformat(e["issue_local"]).astimezone(timezone.utc)
        target = (issue.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)).astimezone(LOCAL)
        if target.month not in range(1,5):
            return []
        start = target.strftime("%Y-%m-%dT%H:00:00")
        end = (target.astimezone(timezone.utc)+timedelta(hours=1)).astimezone(LOCAL).strftime("%Y-%m-%dT%H:00:00")
        stops = sorted(s for s,c in stations.items() if c in roles[eid])
        condition = (f"service_date between '{target.date()-timedelta(days=1)}' and '{target.date()+timedelta(days=1)}' "
                     f"AND gtfs_stop_id in ({quoted(stops)}) AND trip_line in ({quoted(e['routes'].split(' | '))}) "
                     f"AND revenue_service='1' AND trip_type='0' AND ((departure_time >= '{start}' AND departure_time < '{end}') "
                     f"OR (departure_time IS NULL AND arrival_time >= '{start}' AND arrival_time < '{end}'))")
        q = {"$where":condition,"$select":"gtfs_stop_id,trip_line,count(*) as calls",
             "$group":"gtfs_stop_id,trip_line","$order":"gtfs_stop_id,trip_line","$limit":1000}
        rows = read("role-calls-"+eid+".json",api("g8es-h7gb",**q))
        if len(rows)>=1000:
            raise ValueError("Effective timetable query truncated")
        totals = Counter()
        for r in rows:
            totals[stations[r["gtfs_stop_id"]]] += int(r["calls"])
        return [dict(event_id=eid,complex_id=cid,hour=target.isoformat(),roles="|".join(sorted(tags)),
                     effective_calls=totals[cid]) for cid,tags in sorted(roles[eid].items())]
    out.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        units = [r for block in pool.map(check,events) for r in block]
    write_csv(out/"effective_calls.csv",units)
    (out/"effective_sources.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.cache,args.output)
