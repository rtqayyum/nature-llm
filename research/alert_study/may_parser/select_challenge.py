"""Select an additional May challenge set using lexical triggers and frozen hash order."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from select_cases import SOURCE_SHA

TRIGGERS = {
    "transfer": r"\btransfer (?:at|between|to)\b",
    "substitute": r"\b(?:take|use) (?:a nearby |the nearby |the |an? )?(?:[A-Z]+\d+(?:-SBS)? buses?|shuttle buses?)\b",
    "boarding": r"\b(?:board .*? (?:platform|level)|boarding change|boarding changes)\b",
    "turnback": r"\b(?:last stop|turn back|end at|terminate at)\b",
    "suspension": r"\bno [^.!?\n]{0,65}(?:service|trains) [^.!?\n]{0,25}between\b",
}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("primary",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("Source drift")
    seen={r["event_id"] for r in json.loads(a.primary.read_text())}
    used_text=set()
    extra=[]
    for category,pattern in TRIGGERS.items():
        ranked=sorted(json.loads(raw),key=lambda r:hashlib.sha256(
            ("NYC-May-rare-v1/"+category+"/"+r["event_id"]).encode()).digest())
        taken=0
        for r in ranked:
            s=r["header"]+"\n"+r.get("description","")
            active=re.split(r"planned (?:service|work) reminder",s,maxsplit=1,flags=re.I)[0]
            if r["event_id"] in seen or active in used_text or not re.search(pattern,active,re.I):
                continue
            seen.add(r["event_id"])
            used_text.add(active)
            extra.append(dict(event_id=r["event_id"],alert_id=r["alert_id"],routes=r.get("affected",""),
                              header=r["header"],description=r.get("description",""),date_utc=r["date"],trigger=category))
            taken+=1
            if taken==2:
                break
    a.output.write_text(json.dumps(extra,ensure_ascii=False,indent=2)+"\n")
