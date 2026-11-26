"""Describe May notice revisions visible strictly between original issue and target start."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_panel import HERE, StationDictionary, resolve, rule_output


def verified_json(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Source mismatch for {path.name}")
    return json.loads(raw)


def role_set(row, directory):
    case = dict(routes=row.get("affected",""), header=row["header"], description=row.get("description",""))
    routes = set(case["routes"].split(" | ")) - {""}
    return {(cid,role) for name,role in rule_output(case,directory)
            if (cid := resolve(name,routes,directory))}


def run(archives, stations, out):
    sources = json.loads((HERE/"panel/sources.json").read_text())
    directory = StationDictionary(verified_json(stations,sources["stations"]["sha256"]))
    versions = defaultdict(list)
    for row in verified_json(archives,sources["versions-2026-05.json"]["sha256"]):
        ts = datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc)
        versions[row["event_id"]].append((ts,int(row["update_number"]),row))
    with (HERE/"panel/events.csv").open() as f:
        eligible = {r["event_id"] for r in csv.DictReader(f) if r["population"]=="primary"}
    paired = defaultdict(set)
    with gzip.open(HERE/"controls/operator_rail.csv.gz","rt") as f:
        for r in csv.DictReader(f):
            paired[r["event_id"]].add(r["complex_id"])
    ledger = []
    with (HERE/"origins.csv").open() as f:
        for origin in csv.DictReader(f):
            eid = origin["event_id"]
            if eid not in eligible:
                continue
            start, target = map(datetime.fromisoformat,(origin["origin"],origin["hour"]))
            records = sorted(versions[eid],key=lambda v: v[:2])
            first_ts, n, first = next(v for v in records if v[1]==0)
            assert first["alert_id"]==origin["alert_id"] and first_ts==start and start<target
            seen = [v for v in records if v[1]>0 and start<v[0]<target]
            at_start = sum(v[1]>0 and v[0]==target for v in records)
            named = set()
            before,after = set(),set()
            latest = None
            same_native = False
            if seen:
                ts, _, latest = seen[-1]
                same_native = all(first.get(k,"")==latest.get(k,"") for k in ("affected","status_label"))
                before,after = role_set(first,directory),role_set(latest,directory)
                named = {c for c,_ in before^after}
            ledger.append(dict(event_id=eid,original_alert_id=first["alert_id"],issue=origin["origin"],
                target_start=origin["hour"],paired_stations=len(paired[eid]),earlier_updates=len(seen),
                updates_at_target_start=at_start,latest_alert_id=latest["alert_id"] if latest else "",
                latest_utc=ts.isoformat() if latest else "",same_native=int(same_native) if latest else "",
                removed=json.dumps(sorted(before-after)),added=json.dumps(sorted(after-before)),
                changed_paired_stations=len(named & paired[eid])))
    assert len(ledger)==len(eligible) and sum(r["paired_stations"] for r in ledger)==46621
    def summary(rows):
        updated = [r for r in rows if r["earlier_updates"]]
        changed = [r for r in updated if r["added"]!="[]" or r["removed"]!="[]"]
        same = [r for r in changed if r["same_native"]]
        return dict(origins=len(rows),paired_station_hours=sum(r["paired_stations"] for r in rows),
                    revised_origins=len(updated),revised_paired_station_hours=sum(r["paired_stations"] for r in updated),
                    at_target_start_only=sum(bool(r["updates_at_target_start"] and not r["earlier_updates"]) for r in rows),
                    revised_same_native=sum(r["same_native"] for r in updated),changed_roles=len(changed),
                    changed_roles_same_native=len(same),
                    changes_on_paired_stations=sum(r["changed_paired_stations"]>0 for r in changed),
                    changed_paired_rows=sum(r["changed_paired_stations"] for r in changed),
                    same_native_changes_on_paired_stations=sum(r["changed_paired_stations"]>0 for r in same),
                    same_native_changed_paired_rows=sum(r["changed_paired_stations"] for r in same))
    out.mkdir(parents=True,exist_ok=True)
    with (out/"events.csv").open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=ledger[0],lineterminator="\n")
        w.writeheader()
        w.writerows(ledger)
    report = {"primary":summary(ledger),"paired":summary([r for r in ledger if r["paired_stations"]])}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archives","stations","output"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    run(a.archives,a.stations,a.output)
