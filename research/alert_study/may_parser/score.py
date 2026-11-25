"""Compare locked extraction outputs at mention-role and complex-role levels."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from extract_alert_roles import StationDictionary
from build_sample import write_csv


def norm(s):
    return " ".join(s.casefold().split())


def resolve(station, routes, dictionary):
    choices=dictionary.aliases.get(station.casefold().replace("/","-"),{})
    if len(choices)==1:
        return next(iter(choices))
    compatible=[cid for cid,lines in choices.items() if lines & routes]
    return compatible[0] if len(compatible)==1 else ""


def rule_output(case,dictionary):
    rows=[]
    for field in ["header","description"]:
        chunk=re.split(r"planned (?:service|work) reminder",case[field],maxsplit=1,flags=re.I)
        rows += [(r["station_text"],r["role"]) for r in dictionary.match(chunk[0],set(case["routes"].split(" | ")))]
        if len(chunk)>1:
            break
    return rows


def decode(row):
    text=row["raw"].strip()
    fenced=re.fullmatch(r"```(?:json)?\s*(.*?)\s*```",text,flags=re.S|re.I)
    if fenced:
        text=fenced.group(1)
    try:
        pairs=json.loads(text)
        if row["finish_reason"]!="stop" or not isinstance(pairs,list):
            raise ValueError("incomplete or wrong type")
        if not all(isinstance(pair,list) and len(pair)==2 and all(isinstance(x,str) and x.strip() for x in pair) for pair in pairs):
            raise ValueError("bad pair")
        return pairs,True
    except (ValueError,TypeError):
        return [],False


def evaluate(station_file,out):
    if hashlib.sha256(station_file.read_bytes()).hexdigest()!="40f829816b08434dfd23ddd6e2b75a98ad7bd9bc2c00861c9a257726435cb0cb":
        raise ValueError("Station directory differs from the earlier rule baseline")
    dictionary=StationDictionary(json.loads(station_file.read_bytes()))
    scores,detail,errors=[],[],[]
    for sample,cases_name,prefix in (("stratified","cases",""),("challenge","challenge","challenge_"),
                                    ("followup","followup","followup_")):
        cases=json.loads((ROOT/(cases_name+".json")).read_text())
        gold=defaultdict(list)
        with (ROOT/(prefix+"reference.csv")).open() as f:
            for r in csv.DictReader(f):
                gold[r["event_id"]].append((r["station"],r["role"],r["complex_id"]))
        qwen={r["event_id"]:r for r in json.loads((ROOT/(prefix+"qwen.json")).read_text())}
        methods = {"rule":None,"qwen":qwen}
        if sample=="followup":
            methods["qwen_examples"]={r["event_id"]:r for r in json.loads((ROOT/"followup_examples.json").read_text())}
        if any(set(v)!={r["event_id"] for r in cases} for v in methods.values() if v is not None):
            raise ValueError("Different prediction inventory")
        for c in cases:
            eid=c["event_id"]
            text=c["header"]+"\n"+c["description"]
            if any(station not in text for station,_,_ in gold[eid]) or qwen[eid]["alert_id"]!=c["alert_id"]:
                raise ValueError("Source mismatch")
            for method,outputs in methods.items():
                pairs,valid=(rule_output(c,dictionary),True) if method=="rule" else decode(outputs[eid])
                for level in ("mention","complex"):
                    reference={(norm(s) if level=="mention" else cid,role) for s,role,cid in gold[eid] if level=="mention" or cid}
                    ambiguous={(norm(s),role) for s,role,cid in gold[eid] if not cid}
                    pred=set()
                    for s,role in pairs:
                        if level=="complex" and (norm(s),norm(role)) in ambiguous:
                            continue
                        if norm(s) not in norm(text):
                            key="<ungrounded> "+norm(s)
                        elif level=="mention":
                            key=norm(s)
                        else:
                            key=resolve(s,set(c["routes"].split(" | ")),dictionary) or "<unresolved> "+norm(s)
                        pred.add((key,norm(role)))
                    tp,fp,fn=len(pred&reference),len(pred-reference),len(reference-pred)
                    detail.append(dict(sample=sample,event_id=eid,method=method,level=level,tp=tp,fp=fp,fn=fn,
                                       valid=int(valid),exact=int(valid and pred==reference)))
                    for kind,diff in (("extra",pred-reference),("miss",reference-pred)):
                        for station,role in sorted(diff):
                            errors.append(dict(sample=sample,event_id=eid,method=method,level=level,error=kind,station=station,role=role))
        for method in methods:
            for level in ("mention","complex"):
                d=[r for r in detail if (r["sample"],r["method"],r["level"])==(sample,method,level)]
                tp,fp,fn=[sum(r[k] for r in d) for k in ("tp","fp","fn")]
                scores.append(dict(sample=sample,method=method,level=level,notices=len(d),
                                   exact=sum(r["exact"] for r in d),invalid=sum(1-r["valid"] for r in d),
                                   tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp),recall=tp/(tp+fn),
                                   f1=2*tp/(2*tp+fp+fn)))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"scores.csv",scores)
    write_csv(out/"case_scores.csv",detail)
    write_csv(out/"errors.csv",errors)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("station_file",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    evaluate(a.station_file,a.output)
