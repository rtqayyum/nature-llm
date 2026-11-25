"""Fixed follow-up after the first prompt failed: four cases per wording stratum."""
import hashlib
import json
import re
from pathlib import Path
from select_cases import SOURCE_SHA

def draw(raw, excluded):
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("Source mismatch")
    cases={}
    priority=[("platform",r"\b(?:boarding change|boarding changes|board|platform)\b"),
              ("short turn",r"\b(?:last stop|turn back|end at|terminate at)\b"),
              ("suspension",r"\b(?:no .*?service between|no trains between|suspended|not stopping)\b"),
              ("other instruction",r"\b(?:express|running local|rerouted|skipping|bypassing|transfer|alternative|instead)\b")]
    for row in json.loads(raw):
        if row["event_id"] in excluded:
            continue
        text=row["header"]+"\n"+row.get("description","")
        text=re.split(r"planned (?:service|work) reminder",text,flags=re.I,maxsplit=1)[0]
        name=next((k for k,p in priority if re.search(p,text,re.I)),"other")
        cases.setdefault(name,[]).append(dict(event_id=row["event_id"],alert_id=row["alert_id"],routes=row.get("affected",""),header=row["header"],description=row.get("description",""),date_utc=row["date"],stratum=name))
    result=[]
    for name in sorted(cases):
        rank=sorted(cases[name],key=lambda r:hashlib.sha256(("NYC-May-prompt-check-v1/"+r["event_id"]).encode()).digest())
        for r in rank[:4]:
            result.append(dict(r,stratum_size=len(rank)))
    return result

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    folder=Path(__file__).parent
    excluded={r["event_id"] for f in ("cases.json","challenge.json") for r in json.loads((folder/f).read_text())}
    args.output.write_text(json.dumps(draw(args.source.read_bytes(),excluded),ensure_ascii=False,indent=2)+"\n")
