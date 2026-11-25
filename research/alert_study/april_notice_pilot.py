"""Train earlier fixed ridge baselines and score the April development period."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from audit_cohort import cause_screen
from audit_february import notice
from february_ablation import ALPHA, ROOT, MODES


def load_split(cache,folder):
    events = pd.read_csv(folder/"events.csv",keep_default_na=False)
    events["issue"] = pd.to_datetime(events.issue_local,utc=True)
    docs, groups, prior, episodes, causes, statuses = {}, {}, {}, {}, {}, {}
    manifest = json.loads((folder/"sources.json").read_text())
    for month in range(1,5):
        name = f"versions-2026-{month:02}.json"
        raw = (cache/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[name]["sha256"]:
            raise ValueError("Alert source mismatch")
        for r in json.loads(raw):
            if int(r["update_number"])!=0:
                continue
            eid = int(r["event_id"])
            docs[eid] = r["header"]+"\n"+r.get("description","")
            key = r.get("affected",""), " ".join(r["header"].lower().split())
            ts = pd.Timestamp(r["date"])
            if key not in prior or ts-prior[key]>pd.Timedelta(days=1):
                episodes[key] = eid
            prior[key] = ts
            groups[eid] = episodes[key]
            causes[eid],statuses[eid] = cause_screen(notice(r)),r["status_label"]
    if set(docs)!=set(events.event_id):
        raise ValueError("Unmatched notice inventory")
    events["episode"] = events.event_id.map(groups)
    panel = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("2026-??.csv.gz"))],ignore_index=True)
    panel = panel.dropna(subset=["rail","bike","weekly_rail","weekly_bike"]).merge(
        events[["event_id","issue","episode","routes"]],on="event_id",validate="many_to_one")
    panel["tags"] = panel.roles.fillna("")
    target = pd.to_datetime(panel.hour,utc=True).dt.tz_convert("America/New_York")
    panel["end"] = target+pd.Timedelta(hours=1)
    panel["clock"] = target.dt.hour.astype(str)
    panel["weekday"] = target.dt.dayofweek.astype(str)
    panel["how"] = (target.dt.dayofweek*24+target.dt.hour).astype(str)
    panel["status"] = panel.event_id.map(statuses)
    panel["cue"] = panel.event_id.map(causes)
    start = pd.Timestamp("2026-04-01",tz="America/New_York")
    stop = pd.Timestamp("2026-05-01",tz="America/New_York")
    test = panel.loc[panel.issue.ge(start)&panel.end.le(stop)].copy()
    crossing = events.loc[events.issue.ge(start)&events.issue.lt(stop),"episode"]
    pre = panel.loc[panel.end.le(start)]
    train = pre.loc[~pre.episode.isin(crossing)].copy()
    if set(train.episode)&set(test.episode) or train.end.max()>test.issue.min():
        raise ValueError("Leaking split")
    return train, test, docs, len(pre)-len(train)


def run(cache,folder,out):
    out.mkdir(parents=True,exist_ok=True)
    train, test, docs, purged = load_split(cache,folder)
    vector = TfidfVectorizer(ngram_range=(1,2),min_df=2,token_pattern=r"(?u)\b\w+\b")
    vector.fit([docs[e] for e in sorted(set(train.event_id))])
    def words(frame):
        ids,idx = np.unique(frame.event_id,return_inverse=True)
        return vector.transform([docs[e] for e in ids])[idx]
    train_ids = sorted(set(train.event_id))
    fit_text, score_text = words(train), words(test)
    forecasts=[]
    for mode in MODES:
        def records(frame,arm):
            for r in frame.itertuples():
                v = {"weekly_log_count":np.log1p(getattr(r,"weekly_"+mode)),
                     "complex":str(r.complex_id),"hour":r.clock,"weekday":r.weekday,"hour_of_week":r.how}
                if arm!="history":
                    v.update(routes=r.routes,status=r.status,cue=r.cue)
                if arm in ("text_names","text_roles") and r.tags:
                    v.update(mentioned=1.0,named_complex=str(r.complex_id))
                    if arm=="text_roles":
                        v.update({"role_"+t:1.0 for t in r.tags.split("|")})
                yield v
        pred = test[["event_id","complex_id","hour","tags"]].copy()
        pred["mode"],pred["actual"],pred["weekly"] = mode,test[mode],test["weekly_"+mode]
        y = (np.log1p(train[mode])-np.log1p(train["weekly_"+mode])).to_numpy()
        for arm in ("history","native_cue","text","text_names","text_roles"):
            v = DictVectorizer()
            a,b = v.fit_transform(records(train,arm)),v.transform(records(test,arm))
            if arm.startswith("text"):
                a = sparse.hstack([a,fit_text],format="csr")
                b = sparse.hstack([b,score_text],format="csr")
            fit = Ridge(alpha=ALPHA,solver="lsqr",tol=1e-10).fit(a,y)
            residual = fit.predict(b)
            pred[arm] = np.maximum(0,np.expm1(np.log1p(pred.weekly)+residual))
        forecasts.append(pred)
    predictions = pd.concat(forecasts,ignore_index=True)
    if predictions.duplicated(["hour","complex_id","mode"]).any():
        raise ValueError("Repeated scored key")
    rows = []
    for mode,frame in predictions.groupby("mode"):
        for scope,cases in (("network",frame),("rule_named",frame[frame.tags.ne("")])):
            for arm in ("weekly","history","native_cue","text","text_names","text_roles"):
                error = (cases[arm]-cases.actual).abs()
                if not np.isfinite(error).all():
                    raise ValueError("Nonfinite prediction")
                rows.append(dict(mode=mode,scope=scope,arm=arm,station_hours=len(cases),
                                 incidents=cases.event_id.nunique(),dates=cases.hour.str[:10].nunique(),mae=error.mean()))
    pd.DataFrame(rows).to_csv(out/"april_scores.csv",index=False,float_format="%.9f")
    predictions.to_csv(out/"april_predictions.csv.gz",index=False,float_format="%.10f")
    predictions.loc[predictions.tags.ne("")].to_csv(out/"april_named_predictions.csv",index=False,float_format="%.10f")
    print(f"fit {len(train)}, purge {purged}, evaluation {len(test)}, training notices {len(train_ids)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("panel",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.cache,args.panel,args.output)
