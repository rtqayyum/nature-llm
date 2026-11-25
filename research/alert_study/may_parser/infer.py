"""Run the frozen May text-only comparison against a local llama.cpp chat server."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent
LABELS=("express boundary","local boundary","reroute boundary","suspension boundary","short turn",
        "boarding platform","skipped","transfer","alternative service")

def infer(cases, prompt, url, out):
    def predict(row):
        source = f"Affected routes: {row['routes']}\nHeader: {row['header']}\nDescription: {row['description']}"
        messages=[{"role":"system","content":prompt},{"role":"user","content":source}]
        body={"messages":messages,"temperature":0,"seed":20261125,"max_tokens":384,
              "model":"nyc-qwen","stream":False}
        request=Request(url.rstrip("/")+"/v1/chat/completions",data=json.dumps(body).encode(),
                        headers={"Content-Type":"application/json"})
        data=json.load(urlopen(request,timeout=600))
        answer=data["choices"][0]
        return dict(event_id=row["event_id"],alert_id=row["alert_id"],raw=answer["message"]["content"],
                    finish_reason=answer["finish_reason"],usage=data.get("usage",{}))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(predict,cases))
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("output",type=Path)
    p.add_argument("--cases",type=Path,default=ROOT/"cases.json")
    args=p.parse_args()
    infer(json.loads(args.cases.read_text()),(ROOT/"prompt.txt").read_text().strip(),args.url,args.output)
