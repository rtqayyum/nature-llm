"""Freeze first-notice origins and collect CPU 7B outputs from exact-prompt shards."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT.parent/"may_parser"))
from infer import infer
from select_cases import SOURCE_SHA
from audit_cohort import cause_screen
from audit_february import notice

PROMPT_SHA="82afac0b8f25ea2723d06b6dd2284f6a31579c3d2e47f34b043d4fb05f8b1eb5"
MODEL="Qwen/Qwen2.5-7B-Instruct-GGUF@bb5d59e06d9551d752d08b292a50eb208b07ab1f/Q4_K_M/llama.cpp-b12721"
N_SHARDS=96
NY=ZoneInfo("America/New_York")

def digest(b):
    return hashlib.sha256(b).hexdigest()

def inputs(cache,may_path):
    source=json.loads((ROOT.parent/"role_screen/sources.json").read_text())
    cases=[]
    for m in range(1,6):
        p=cache/f"2026-{m:02}.json" if m<5 else may_path
        b=p.read_bytes()
        if digest(b)!=(source[p.name]["sha256"] if m<5 else SOURCE_SHA):
            raise ValueError("Changed alert archive: "+p.name)
        for row in json.loads(b):
            if int(row["update_number"])!=0:
                raise ValueError("Expected first notice")
            c=dict(event_id=row["event_id"],alert_id=row["alert_id"],routes=row.get("affected",""),
                   header=row["header"],description=row.get("description",""),date_utc=row["date"])
            payload=json.dumps([c[k] for k in ("routes","header","description")],ensure_ascii=False,separators=(",",":")).encode()
            c["key"]=digest(PROMPT_SHA.encode()+b"\0"+MODEL.encode()+b"\0"+payload)
            c["shard"]=int(c["key"][:8],16)%N_SHARDS
            cases.append(c)
    if len({c["event_id"] for c in cases})!=len(cases):
        raise ValueError("Incident crosses inventory")
    return sorted(cases,key=lambda c:(c["date_utc"],int(c["event_id"])))

def ledger(cases,out):
    audit={r["event_id"] for name in ["cases","challenge","followup"]
           for r in json.loads((ROOT.parent/f"may_parser/{name}.json").read_text())}
    keys={c["key"] for c in cases if c["event_id"] in audit}
    previous,episodes={},{}
    rows=[]
    for c in cases:
        h=" ".join(c["header"].lower().split())
        key=(c["routes"],h)
        ts=datetime.fromisoformat(c["date_utc"]).replace(tzinfo=timezone.utc)
        if key not in previous or ts-previous[key]>timedelta(hours=24):
            episodes[key]=c["event_id"]
        previous[key]=ts
        t=(ts.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)).astimezone(NY)
        rows.append(dict(event_id=c["event_id"],alert_id=c["alert_id"],origin=ts.astimezone(NY).isoformat(),
                         hour=t.isoformat(),episode=episodes[key],routes=c["routes"],class_cue=cause_screen(notice(c)),
                         audit_case=int(c["event_id"] in audit),audit_text=int(c["key"] in keys),
                         parser_key=c["key"],shard=c["shard"]))
    aud_episodes={r["episode"] for r in rows if r["audit_case"]}
    may=[dict(r,audit_episode=int(r["episode"] in aud_episodes)) for r in rows if r["origin"].startswith("2026-05")]
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=may[0],lineterminator="\n")
        w.writeheader()
        w.writerows(may)
    return may

def batch(cases,prompt,url,shard,out):
    group={}
    for c in cases:
        if c["shard"]==shard:
            group.setdefault(c["key"],c)
    ordered=[group[key] for key in sorted(group)]
    if not ordered:
        raise ValueError("Empty shard")
    if out.exists():
        with gzip.open(out,"rt") as f: existing=json.load(f)
        if [r["key"] for r in existing]!=sorted(group) or any(r["event_id"]!=group[r["key"]]["event_id"] or "raw" not in r for r in existing):
            raise ValueError("Existing shard differs")
        print("Already collected",shard,len(existing))
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        raw=Path(tmp)/"raw.json"
        infer(ordered,prompt,url,raw)
        rows=json.loads(raw.read_text())
        if len(rows)!=len(ordered):
            raise ValueError("Incomplete batch")
        for row,c in zip(rows,ordered):
            if row["event_id"]!=c["event_id"] or row["alert_id"]!=c["alert_id"]:
                raise ValueError("Output identity mismatch")
            row["key"]=c["key"]
        ready=Path(tmp)/"shard.gz"
        ready.write_bytes(gzip.compress((json.dumps(rows,ensure_ascii=False,sort_keys=True)+"\n").encode(),mtime=0))
        os.replace(ready,out)
    print("Collected",shard,len(ordered))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("may_source",type=Path)
    p.add_argument("--ledger",type=Path)
    p.add_argument("--shard",type=int,choices=range(N_SHARDS))
    p.add_argument("--server",default="http://127.0.0.1:18443")
    p.add_argument("--out",type=Path,default=ROOT/"parser_shards")
    args=p.parse_args()
    prompt=(ROOT.parent/"may_parser/prompt_examples.txt").read_bytes()
    if digest(prompt)!=PROMPT_SHA:
        raise ValueError("Prompt differs")
    cases=inputs(args.cache,args.may_source)
    print("Notices",len(cases),"unique model inputs",len({c["key"] for c in cases}),flush=True)
    if args.ledger:
        ledger(cases,args.ledger)
    if args.shard is not None:
        batch(cases,prompt.decode().strip(),args.server,args.shard,args.out/f"{args.shard:02}.json.gz")
