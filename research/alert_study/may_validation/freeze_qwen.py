"""Freeze the preregistered Qwen predictions after every exact model input has returned."""
import argparse
import gzip
import io
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from collect_text import inputs
from freeze_controls import Design, HERE, KEY, MODES, boost, get_split, sha
from read_extractions import read


def station_roles(frame, mapping):
    return [mapping[r.event_id].get(r.complex_id,set()) for r in frame.itertuples()]


def exchange_indices(frame, tags):
    pairs = []
    named = frame[KEY].reset_index(drop=True)
    named["role_set"] = tags
    named = named.loc[named.role_set.map(bool)]
    for _,g in named.groupby("event_id",sort=False):
        ordered = g.sort_values("complex_id").index
        for a,b in combinations(ordered,2):
            if tags[a].isdisjoint(tags[b]):
                pairs.append((a,b))
                break
    return np.asarray(pairs,dtype=int).reshape(-1,2)


def indicators(tags, vocab):
    return np.asarray([[role in record for role in vocab] for record in tags],dtype=float)


def run(cache, notice_cache, out):
    cases = inputs(notice_cache,cache/"may_notices.json")
    mapping, extraction = read(cases,cache/"stations",HERE/"parser_shards")
    train,test,docs,splits = get_split(cache)
    controls = HERE/"controls"
    frozen = json.loads((controls/"receipt.json").read_text())
    if any(sha(controls/name)!=r["sha256"] for name,r in frozen["files"].items()):
        raise ValueError("Registered comparison predictions changed")
    if any(splits[fit]!=frozen[fit] for fit in train) or len(test)!=frozen["evaluation_rows"]:
        raise ValueError("Different comparison population")
    out.mkdir(parents=True,exist_ok=True)
    test_tags = station_roles(test,mapping)
    pairs = exchange_indices(test,test_tags)
    pair_rows = pairs.ravel()
    exchange_rows = pairs[:,::-1].ravel()
    receipt = dict(extraction=extraction,splits=splits,files={},network_rows=len(test),
                   named_rows=sum(map(bool,test_tags)),exchange_incidents=len(pairs),
                   exchange_dates=test.iloc[pair_rows].hour.str[:10].nunique())
    for fit, frame in train.items():
        design = Design(frame,test,docs)
        train_tags = station_roles(frame,mapping)
        roles = sorted(set().union(*train_tags))
        train_roles,test_roles = indicators(train_tags,roles),indicators(test_tags,roles)
        train_names = np.array(list(map(bool,train_tags)))
        test_names = np.array(list(map(bool,test_tags)))
        for mode in MODES:
            x = np.column_stack((design.matrix(frame,mode,fit,"text_rules"),train_names))
            v = np.column_stack((design.matrix(test,mode,fit,"text_rules"),test_names))
            pred = test[KEY].copy()
            pred["qwen_named"] = test_names.astype(int)
            y = frame[mode].to_numpy()
            offset = np.zeros(len(test))
            if fit=="operator":
                y = y-frame["weekly_"+mode].to_numpy()
                offset = test["weekly_"+mode].to_numpy()
            pred["names"] = np.maximum(0,offset+boost(x,y,v,fit))
            # The second evaluation block contains just the two selected stations per incident.
            # Exchange their Qwen roles, leaving each station's own name, rule and count inputs in place.
            full_v = np.column_stack((v,test_roles))
            swap_v = np.column_stack((v[pair_rows],test_roles[exchange_rows]))
            forecasts = boost(np.column_stack((x,train_roles)),y,np.vstack((full_v,swap_v)),fit)
            pred["roles"] = np.maximum(0,offset+forecasts[:len(test)])
            exchanged = np.maximum(0,offset[pair_rows]+forecasts[len(test):])
            if not np.isfinite(pred[["names","roles"]]).all().all() or not np.isfinite(exchanged).all():
                raise ValueError("Nonfinite Qwen prediction")
            pred["roles_exchanged"] = np.nan
            pred.iloc[pair_rows,pred.columns.get_loc("roles_exchanged")] = exchanged
            path = out/f"{fit}_{mode}.csv.gz"
            with path.open("wb") as raw,gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding="utf-8",newline="") as dst:
                    pred.to_csv(dst,index=False,float_format="%.10f",lineterminator="\n")
            receipt["files"][path.name] = dict(sha256=sha(path),bytes=path.stat().st_size)
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("notice_cache",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        run(a.cache,a.notice_cache,a.output)
