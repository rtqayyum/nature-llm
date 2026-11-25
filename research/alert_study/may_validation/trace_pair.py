"""Count complete rule-role contrasts and verify the earliest one against raw source records."""
import argparse
import csv
import gzip
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_sample import api, inputs, write_csv


def run(cache, out):
    panel = HERE/"panel"
    pinned = json.loads((panel/"sources.json").read_text())
    with (HERE/"origins.csv").open() as f:
        origins = {r["event_id"]:r for r in csv.DictReader(f)}
    named = defaultdict(list)
    with gzip.open(panel/"primary.csv.gz","rt") as f:
        for r in csv.DictReader(f):
            if r["roles"] and all(r[k] for k in ("rail","bike","weekly_rail","weekly_bike")):
                named[r["event_id"]].append(r)
    pairs = [eid for eid,rows in named.items() if any(set(a["roles"].split("|")).isdisjoint(b["roles"].split("|"))
                                                     for a,b in combinations(rows,2))]
    eid = min(pairs,key=lambda e:(datetime.fromisoformat(origins[e]["origin"]),int(e)))
    rows = named[eid]
    hour = datetime.fromisoformat(origins[eid]["hour"])
    places = {r["complex_id"]:r for r in rows}
    out.mkdir(exist_ok=True,parents=True)
    manifest = {}
    archive = inputs(cache,manifest,"202605.zip",pinned["202605.zip"]["url"])
    if manifest["202605.zip"] != pinned["202605.zip"]:
        raise ValueError("Different trip archive")
    with (panel/"docks.csv").open() as f:
        docks = {r["dock"]:r for r in csv.DictReader(f) if r["complex_id"] in places}
    counts,seen = Counter(),{}
    with zipfile.ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            if not member.endswith(".csv") or member.startswith("__MACOSX/"):
                continue
            with z.open(member) as stream:
                reader = csv.reader(io.TextIOWrapper(stream,encoding="utf-8-sig"))
                header = next(reader)
                rid,when,station = map(header.index,("ride_id","started_at","start_station_id"))
                for t in reader:
                    dock = t[station].strip()
                    if t[when][:13].replace("T"," ") != hour.strftime("%Y-%m-%d %H") or dock not in docks:
                        continue
                    key = t[rid]
                    if not key or (key in seen and seen[key]!=(dock,t[when])):
                        raise ValueError("Invalid or conflicting trip ID")
                    if key not in seen:
                        counts[dock] += 1
                        seen[key] = (dock,t[when])
    results = []
    for cid,r in sorted(places.items(),key=lambda k:int(k[0])):
        name = f"trace-{eid}-rail-{cid}.json"
        query = dict(transit_mode="subway", transit_timestamp=hour.strftime("%Y-%m-%dT%H:%M:%S"),
                     station_complex_id=cid, **{"$order":"payment_method,fare_class_category","$limit":1000})
        fare = json.loads(inputs(cache,manifest,name,api("5wq4-mkjj",**query)).read_bytes())
        if len(fare)>=1000 or not fare or len({v["station_complex"] for v in fare})!=1:
            raise ValueError("Incomplete or ambiguous fare records")
        n_rail = sum(Decimal(v["ridership"]) for v in fare)
        assigned = [dock for dock,d in docks.items() if d["complex_id"]==cid]
        n_bike = sum(counts[d] for d in assigned)
        if (n_rail,n_bike)!=(int(r["rail"]),int(r["bike"])):
            raise ValueError("Independent raw counts disagree")
        results.append(dict(complex_id=cid,station=fare[0]["station_complex"],rule_roles=r["roles"],
                            fare_records=len(fare),rail_entries=n_rail,assigned_docks=len(assigned),bike_starts=n_bike))
    expected = HERE/"trace/sources.json"
    if expected.exists() and json.loads(expected.read_text())!=manifest:
        raise ValueError("Trace source drift")
    with gzip.open(panel/"versions.csv.gz","rt") as f:
        versions = [r for r in csv.DictReader(f) if r["event_id"]==eid]
    write_csv(out/"docks.csv",[dict(**v,starts=counts[k]) for k,v in sorted(docks.items())])
    write_csv(out/"stations.csv",results)
    (out/"receipt.json").write_text(json.dumps(dict(complete_disjoint_incidents=len(pairs),event_id=eid,
        origin=origins[eid]["origin"],hour=hour.isoformat(),versions=versions),indent=2)+"\n")
    (out/"sources.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    run(args.cache,args.output)
