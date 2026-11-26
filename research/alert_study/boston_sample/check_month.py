"""Outcome-blind inventory followed by a targeted development join for immediate instructions."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from pathlib import Path

import pyarrow.parquet as pq
from check_sample import HERE, HEAVY, NY, measure, stops, verify, write_csv

CUES=re.compile(r"shuttl|express|single.track|bypass|divert|turn.back|terminat|suspend|transfer|board at",re.I)


def run(cache,out):
    verify(cache)
    start,end=(int(datetime(2025,m,1,tzinfo=NY).timestamp()) for m in (5,6))
    records=pq.read_table(cache/"alerts.parquet",filters=[("created_timestamp",">=",start),("created_timestamp","<",end)]).to_pylist()
    events=defaultdict(list)
    for r in records:
        events[r["id"]].append(r)
    with (HERE/"month_screen/review.csv").open() as f:
        review={int(r["alert_id"]):r for r in csv.DictReader(f)}
    annotations=defaultdict(dict)
    with (HERE/"month_screen/station_readings.csv").open() as f:
        for r in csv.DictReader(f):
            annotations[int(r["alert_id"])][r["station"]]=r["instruction"]
    inventory,screened=[],set()
    history=[]
    by_day=defaultdict(list)
    for eid,rows in sorted(events.items()):
        ts=min(r["last_modified_timestamp"] for r in rows)
        first=[r for r in rows if r["last_modified_timestamp"]==ts]
        routes=sorted({r["informed_entity.route_id"] for r in first if r["informed_entity.route_id"]})
        if not set(routes)&HEAVY or all(r["effect"]=="ACCESSIBILITY_ISSUE" or r["informed_entity.facility_id"] for r in first):
            continue
        texts={(r["header_text.translation.text"],r["description_text.translation.text"] or "",r["effect"]) for r in first}
        if len(texts)!=1 or {r["created_timestamp"] for r in first}!={ts}:
            raise ValueError("Incomplete or ambiguous original")
        header,description,effect=texts.pop()
        text=header+"\n"+description
        candidate=effect!="OTHER_EFFECT" or bool(CUES.search(text))
        issue=datetime.fromtimestamp(ts,NY)
        inventory.append(dict(alert_id=eid,issue=issue.isoformat(),routes="|".join(routes),effect=effect,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),screened=int(candidate),first_entity_period_rows=len(first),
            header=header if candidate else ""))
        if candidate:
            screened.add(eid)
            if review[eid]["timing"]=="immediate":
                for modified in sorted({r["last_modified_timestamp"] for r in rows}):
                    block=[r for r in rows if r["last_modified_timestamp"]==modified]
                    state={(r["header_text.translation.text"],r["description_text.translation.text"],
                            r["closed_timestamp"],r["effect"],r["severity"]) for r in block}
                    if len(state)!=1:
                        raise ValueError("Conflicting version content")
                    head,desc,closed,native,severity=state.pop()
                    entities={json.dumps([r["informed_entity."+k] for k in (
                        "route_id","route_type","direction_id","stop_id","facility_id","activities")],
                        ensure_ascii=False,separators=(",",":")) for r in block}
                    signature=hashlib.sha256("\n".join(sorted(entities)).encode()).hexdigest()
                    history.append(dict(alert_id=eid,modified_local=datetime.fromtimestamp(modified,NY).isoformat(),
                        closed=int(closed is not None),effect=native,severity=severity,entity_period_rows=len(block),
                        entity_count=len(entities),entity_sha256=signature,header=" ".join(head.split()),description=" ".join((desc or "").split())))
                for name,reading in annotations[eid].items():
                    if name not in text:
                        raise ValueError("Station span absent")
                    by_day[issue.replace(hour=0,minute=0,second=0)].append((dict(alert_id=eid,issue=issue.isoformat(),
                        target=issue.replace(minute=0,second=0)+timedelta(hours=1)),name,reading))
    if screened!=set(review) or set(annotations)!={e for e,r in review.items() if r["timing"]=="immediate"}:
        raise ValueError("Review inventory mismatch")
    joined,dock_rows,half_hours,contributions=[],[],[],[]
    for date,items in sorted(by_day.items()):
        geo=stops(cache/"stops.parquet",date)
        traces,meaning=[],{}
        for trace,name,reading in items:
            cid,=[cid for cid,(label,_) in geo.items() if label==name]
            traces.append(dict(**trace,complex_id=cid))
            meaning[trace["alert_id"],cid]=reading
        rows,docks,half,counts=measure(cache,geo,traces,date)
        for r in rows:
            r["instruction"]=meaning[r["alert_id"],r["complex_id"]]
        joined.extend(rows)
        dock_rows.extend(dict(map_date=str(date.date()),**r) for r in docks)
        half_hours.extend(half)
        contributions.extend(counts)
    out.mkdir(parents=True,exist_ok=True)
    for name,table in (("origins",inventory),("version_history",history),("traces",joined),("docks",dock_rows),
                       ("half_hours",half_hours),("dock_hours",contributions)):
        write_csv(out/(name+".csv"),table)
    complete=[r for r in joined if all(r[k] is not None for k in ("gate_entries","bike_starts","weekly_gate_entries","weekly_bike_starts"))]
    totals=dict(service_ids=len(inventory),first_at_created=len(inventory),screened_ids=len(review),immediate_ids=len(annotations),
                immediate_dates=len(by_day),reviewed_station_hours=len(joined),complete_station_hours=len(complete),
                complete_by_id=dict(Counter(r["alert_id"] for r in complete)))
    (out/"coverage.json").write_text(json.dumps(totals,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.cache,a.output)
