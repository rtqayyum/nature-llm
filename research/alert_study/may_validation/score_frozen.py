"""Score the complete frozen May forecasts in matched natural units and paired day blocks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ["hour","complex_id","event_id"]
MODES = ("rail","bike")
DAYS = [f"2026-05-{i:02d}" for i in range(1,32)]


def checked_files(path, receipt):
    expected = {f"{fit}_{mode}.csv.gz" for fit in ("operator","lag_free") for mode in MODES}
    if set(receipt["files"]) != expected:
        raise ValueError("Missing forecast file")
    for name,r in receipt["files"].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=r["sha256"]:
            raise ValueError(f"Forecast file changed: {path/name}")


def paired_frame(folder, qwen, fit, mode, targets):
    control = pd.read_csv(folder/f"{fit}_{mode}.csv.gz").set_index(KEY,verify_integrity=True)
    lang = pd.read_csv(qwen/f"{fit}_{mode}.csv.gz").set_index(KEY,verify_integrity=True)
    if not all(frame.index.equals(targets.index) for frame in (control,lang)):
        raise ValueError("Forecasts and observed targets must have exactly the same ordered keys")
    if not lang.qwen_named.isin([0,1]).all():
        raise ValueError("Invalid named-station indicator")
    full = targets.join(control).join(lang)
    base = "weekly" if fit=="operator" else "median"
    if list(control)!=[base,"native","text","text_rules"] or list(lang)!=["qwen_named","names","roles","roles_exchanged"]:
        raise ValueError("Unexpected frozen forecast columns")
    arms = list(control)+["names","roles"]
    if not np.isfinite(full[arms].to_numpy()).all():
        raise ValueError("A forecast is missing on a paired row")
    swap = full.roles_exchanged.notna()
    if not np.isfinite(full.loc[swap,"roles_exchanged"]).all() or (swap & full.qwen_named.eq(0)).any():
        raise ValueError("Invalid exchanged-role predictions")
    if not full.loc[swap].groupby(level="event_id").size().eq(2).all():
        raise ValueError("Exchange must have two retained stations per incident")
    for arm in [*arms,"roles_exchanged"]:
        full[arm]=(full[arm]-full[mode]).abs()
    return full,arms


def block_indices(length):
    starts=np.random.default_rng(20261125).integers(0,len(DAYS),size=(5000,(len(DAYS)+length-1)//length))
    return ((starts[:,:,None]+np.arange(length))%len(DAYS)).reshape(5000,-1)[:,:len(DAYS)]


def reduction(a,b):
    return np.divide(100*(a-b),a,out=np.full(np.shape(a),np.nan),where=a>0)


def contrasts(frames, refs, candidate, mask, subset, fit, name, indices):
    daily = []
    for mode in MODES:
        frame = frames[mode].loc[mask]
        totals = frame.groupby("day")[[refs[mode],candidate]].sum().reindex(DAYS,fill_value=0)
        daily.append(totals.to_numpy())
    daily = np.stack(daily,axis=1)  # day, mode, [reference error sum, candidate error sum]
    totals = daily.sum(axis=0)
    point = reduction(totals[:,0],totals[:,1])
    rows=[]
    for length,idx in indices.items():
        totals=daily[idx].sum(axis=1)
        sampled=reduction(totals[:,:,0],totals[:,:,1])
        draws=np.column_stack([sampled,sampled.mean(axis=1)])
        for i,(mode,p) in enumerate(zip((*MODES,"equal_mode"),[*point,point.mean()])):
            valid=draws[np.isfinite(draws[:,i]),i]
            lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
            rows.append(dict(fit=fit,subset=subset,comparison=name,mode=mode,block_days=length,
                             reduction_pct=p,lower_pct=lo,upper_pct=hi,defined_draws=len(valid),requested_draws=len(idx)))
    return rows


def run(qwen,out,shards=None):
    if shards is None:
        shards=HERE/"parser_shards"
    ctrl=HERE/"controls"
    cr=json.loads((ctrl/"receipt.json").read_text())
    qr=json.loads((qwen/"receipt.json").read_text())
    extraction=qr["extraction"]
    if (extraction["completed_shards"],extraction["verified_inputs"],extraction["linked_notices"])!=(96,8310,9050):
        raise ValueError("Incomplete Qwen extraction")
    for n in range(96):
        name=f"{n:02}.json.gz"
        if hashlib.sha256((shards/name).read_bytes()).hexdigest()!=extraction["files"][name]["sha256"]:
            raise ValueError("Frozen extraction hash mismatch: "+name)
    checked_files(qwen,qr)
    checked_files(ctrl,cr)
    panel=pd.read_csv(HERE/"panel/primary.csv.gz").dropna(subset=["rail","bike","weekly_rail","weekly_bike"])
    origins=pd.read_csv(HERE/"origins.csv").set_index("event_id",verify_integrity=True)
    panel["audit"]=panel.event_id.map(origins[["audit_case","audit_text","audit_episode"]].any(axis=1))
    panel["cue"]=panel.event_id.map(origins.class_cue)
    panel["day"]=panel.hour.str[:10]
    targets=panel.drop(columns="roles").set_index(KEY,verify_integrity=True)
    if targets[["audit","cue"]].isna().any().any() or sorted(targets.day.unique())!=DAYS:
        raise ValueError("Incomplete validation calendar or origin ledger")
    if len(targets)!=qr["network_rows"] or len(targets)!=cr["evaluation_rows"]:
        raise ValueError("Unexpected paired population")
    indices={length:block_indices(length) for length in (7,3)}
    maes, effects, selected=[],[],[]
    for fit in ("operator","lag_free"):
        frames={}
        for mode in MODES:
            frames[mode],arms=paired_frame(ctrl,qwen,fit,mode,targets)
        a,b=frames.values()
        if not a.qwen_named.equals(b.qwen_named) or not a.roles_exchanged.notna().equals(b.roles_exchanged.notna()):
            raise ValueError("Named or exchange masks differ by mode")
        exchanged=a.loc[a.roles_exchanged.notna()]
        if (int(a.qwen_named.sum())!=qr["named_rows"] or len(exchanged)!=2*qr["exchange_incidents"]
                or exchanged.day.nunique()!=qr["exchange_dates"]):
            raise ValueError("Reported Qwen support differs from retained forecast rows")
        masks={"network":np.ones(len(a),dtype=bool),"audit_excluded":~a.audit,"fault_response":a.cue.eq("fault or response wording"),
               "qwen_named":a.qwen_named.eq(1),"role_exchange":a.roles_exchanged.notna()}
        for subset,mask in masks.items():
            for mode,frame in frames.items():
                f=frame.loc[mask]
                for arm in (arms+["roles_exchanged"] if subset=="role_exchange" else arms):
                    maes.append(dict(fit=fit,subset=subset,mode=mode,arm=arm,station_hours=len(f),
                                     incidents=f.index.get_level_values("event_id").nunique(),dates=f.day.nunique(),mae=f[arm].mean()))
            refs={"rail":"weekly","bike":"text"} if fit=="operator" else dict.fromkeys(MODES,"median")
            effects.extend(contrasts(frames,refs,"roles",mask,subset,fit,"roles_vs_fixed_controls",indices))
            effects.extend(contrasts(frames,dict.fromkeys(MODES,"names"),"roles",mask,subset,fit,"roles_vs_names",indices))
            if subset=="role_exchange":
                effects.extend(contrasts(frames,dict.fromkeys(MODES,"roles_exchanged"),"roles",mask,subset,fit,"assigned_vs_exchanged",indices))
        if fit=="operator":
            for mode,frame in frames.items():
                eligible=["weekly","native","text","text_rules"]
                best=min(eligible,key=lambda arm:frame[arm].sum())
                selected.append(dict(mode=mode,selected=best,mae=frame[best].mean(),purpose="later confirmation"))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("mae",maes),("contrasts",effects),("may_selected_controls",selected)):
        pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False,float_format="%.10f")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("qwen",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--shards",type=Path,help="Raw extraction directory used to freeze the Qwen forecasts")
    args=p.parse_args()
    run(args.qwen,args.output,args.shards)
