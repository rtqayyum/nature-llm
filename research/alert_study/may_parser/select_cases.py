"""Recreate the May text-only stratified sample without reading demand or predictions."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE_SHA = "445a8c24ddb86c507f61c4aa0ba10f0f1b1a46861a741f9d386f22d2d1ee275a"
CUES = [
    r"\b(?:board|boarding|platform|upper level|lower level|arrive and depart)\b",
    r"\b(?:last stop|end at|ending at|turn back|terminate)\b",
    r"\b(?:express|local track|running local|rerouted|via the|skipping|bypassing)\b",
    r"\b(?:no .*?service between|no trains between|suspended|not stopping)\b",
    r"\b(?:alternatives?|transfer|instead|for service to/from)\b",
]

def select(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("Different source snapshot")
    strata = defaultdict(list)
    for r in json.loads(raw):
        text = r["header"]+"\n"+r.get("description","")
        active = re.split(r"planned (?:service|work) reminder",text,maxsplit=1,flags=re.I)[0]
        day = datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York")).day
        block = min((day-1)//7+1,4)
        cue = min(sum(bool(re.search(c,active,re.I)) for c in CUES),2)
        strata[block,cue].append(dict(event_id=r["event_id"],alert_id=r["alert_id"],routes=r.get("affected",""),
                                      header=r["header"],description=r.get("description",""),date_utc=r["date"],block=block,cue=cue))
    selected = []
    for (block,cue),cases in sorted(strata.items()):
        take = (2,2,3)[cue]
        cases.sort(key=lambda r: hashlib.sha256(("NYC-May-station-audit-v1/"+r["event_id"]).encode()).digest())
        for r in cases[:take]:
            r.update(stratum_size=len(cases),stratum_draw=min(take,len(cases)))
            selected.append(r)
    return selected

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(select(a.source),ensure_ascii=False,indent=2)+"\n")
