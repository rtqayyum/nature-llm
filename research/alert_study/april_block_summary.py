"""Descriptive paired April uncertainty using circular blocks of complete calendar days."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(predictions,out):
    frame = pd.read_csv(predictions)
    if frame.duplicated(["mode","complex_id","hour"]).any():
        raise ValueError("Repeated evaluation observation")
    for arm in ("weekly","native","text","text_names","text_roles"):
        frame[arm] = (frame[arm]-frame.actual).abs()
    dates = frame.hour.str[:10]
    sums = frame.groupby([dates,frame["mode"]])[["weekly","native","text","text_names","text_roles"]].sum()
    days = sorted(set(dates))
    if days != [f"2026-04-{i:02}" for i in range(1,31)]:
        raise ValueError("Expected thirty consecutive April dates")
    rows = []
    for length in (3,7):
        rng = np.random.default_rng(20261125)
        starts = rng.integers(0,len(days),size=(5000,(len(days)+length-1)//length))
        idx = ((starts[:,:,None]+np.arange(length))%len(days)).reshape(5000,-1)[:,:len(days)]
        for comparator, candidate in (("weekly","text"),("native","text"),("text_names","text_roles")):
            estimates, draws = [],[]
            for mode in ("rail","bike"):
                m = sums.xs(mode,level=1).reindex(days)
                a,b = m[comparator].to_numpy(),m[candidate].to_numpy()
                point = 100*(1-b.sum()/a.sum())
                sampled = 100*(1-b[idx].sum(axis=1)/a[idx].sum(axis=1))
                estimates.append(point)
                draws.append(sampled)
                lo,hi = np.quantile(sampled,[0.025,0.975])
                rows.append([length,comparator,candidate,mode,point,lo,hi])
            lo,hi = np.quantile(np.mean(draws,axis=0),[0.025,0.975])
            rows.append([length,comparator,candidate,"equal_mode",np.mean(estimates),lo,hi])
    pd.DataFrame(rows,columns=["block_days","comparator","candidate","mode","mae_reduction_pct",
                              "development_p025","development_p975"]).to_csv(out,index=False,float_format="%.8f")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.predictions,args.output)
