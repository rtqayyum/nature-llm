"""Verify saved Qwen outputs and resolve their station roles without reading travel outcomes."""
import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from collect_text import MODEL, N_SHARDS, ROOT, inputs
from extract_alert_roles import StationDictionary
from infer import LABELS
from score import norm, resolve


def valid_roles(row, case, dictionary, counts):
    text = row["raw"].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```",text,re.S|re.I)
    try:
        pairs = json.loads(fenced.group(1) if fenced else text)
        if row["finish_reason"] != "stop" or not isinstance(pairs,list):
            raise ValueError("Incomplete array")
    except (ValueError,TypeError):
        counts["invalid_responses"] += 1
        return {}
    source = norm(case["header"]+" "+case["description"])
    routes = set(case["routes"].split(" | "))
    roles = defaultdict(set)
    for pair in pairs:
        if not isinstance(pair,list) or len(pair)!=2 or not all(isinstance(s,str) and s.strip() for s in pair):
            counts["malformed_pairs"] += 1
            continue
        station, role = map(norm,pair)
        if role not in LABELS:
            counts["unknown_role_pairs"] += 1
            continue
        if not re.search(r"(?<![\w/-])"+re.escape(station)+r"(?![\w/-])",source):
            counts["ungrounded_pairs"] += 1
            continue
        cid = resolve(station,routes,dictionary)
        if not cid:
            counts["unresolved_pairs"] += 1
            continue
        roles[int(cid)].add(role)
    counts["accepted_complex_role_pairs"] += sum(map(len,roles.values()))
    counts["inputs_with_accepted_pairs"] += int(bool(roles))
    return dict(roles)


def read(cases, stations, shards, require_complete=True):
    if hashlib.sha256(stations.read_bytes()).hexdigest()!="40f829816b08434dfd23ddd6e2b75a98ad7bd9bc2c00861c9a257726435cb0cb":
        raise ValueError("Different frozen station directory")
    dictionary = StationDictionary(json.loads(stations.read_bytes()))
    expected = defaultdict(dict)
    for c in cases:
        expected[c["shard"]].setdefault(c["key"],c)
    inventory, outputs, counts = {}, {}, Counter()
    for shard in range(N_SHARDS):
        path = shards/f"{shard:02}.json.gz"
        if not path.exists():
            if require_complete:
                raise ValueError(f"Missing extraction shard {shard:02}")
            continue
        raw = path.read_bytes()
        rows = json.loads(gzip.decompress(raw))
        group = expected[shard]
        if [r["key"] for r in rows] != sorted(group):
            raise ValueError(f"Incomplete, repeated or foreign extraction keys in {path.name}")
        for r in rows:
            c = group[r["key"]]
            if (r["event_id"],r["alert_id"]) != (c["event_id"],c["alert_id"]):
                raise ValueError(f"Wrong representative in {path.name}")
            outputs[c["key"]] = valid_roles(r,c,dictionary,counts)
        inventory[path.name] = dict(sha256=hashlib.sha256(raw).hexdigest(),inputs=len(rows))
    by_event = {int(c["event_id"]):outputs[c["key"]] for c in cases if c["key"] in outputs}
    receipt = dict(model=MODEL, files=inventory, completed_shards=len(inventory),total_shards=N_SHARDS,
                   verified_inputs=len(outputs),total_inputs=sum(map(len,expected.values())),
                   linked_notices=len(by_event),total_notices=len(cases),counts_per_unique_input=dict(sorted(counts.items())))
    return by_event,receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("may_source",type=Path)
    p.add_argument("stations",type=Path)
    p.add_argument("--shards",type=Path,default=ROOT/"parser_shards")
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true")
    a = p.parse_args()
    _, receipt = read(inputs(a.cache,a.may_source),a.stations,a.shards,not a.allow_partial)
    a.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
