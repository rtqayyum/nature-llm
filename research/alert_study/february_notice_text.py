"""Add full first-notice TF-IDF and automated station roles to fixed February folds."""
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

from february_ablation import ALPHA, MODES, ROOT, features, get_fold, load_panel


def run(cache, output):
    output.mkdir(parents=True, exist_ok=True)
    folder = ROOT/"role_screen"
    raw = (cache/"2026-02.json").read_bytes()
    digest = json.loads((folder/"sources.json").read_text())["2026-02.json"]["sha256"]
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("February notice archive differs from the extraction source")
    notices = {int(a["alert_id"]): a for a in json.loads(raw)}
    panel, inventory = load_panel(ROOT/"february")
    docs = {}
    for r in inventory.itertuples():
        a = notices[r.alert_id]
        if int(a["event_id"]) != r.event_id or int(a["update_number"]) != 0:
            raise ValueError("Notice identity mismatch")
        docs[r.event_id] = a["header"] + "\n" + a.get("description", "")
    hits = pd.read_csv(folder/"mentions.csv").dropna(subset=["complex_id"])
    roles = hits.groupby(["event_id","complex_id"]).role.agg(lambda s: tuple(sorted(set(s))))
    panel["tags"] = [roles.get((r.event_id,r.complex_id), ()) for r in panel.itertuples()]
    frames = []
    for day in (15,22):
        train, valid, _ = get_fold(panel, inventory, day)
        tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, token_pattern=r"(?u)\b\w+\b")
        ids = sorted(set(train.event_id))
        tfidf.fit([docs[e] for e in ids])
        def words(frame):
            unique, indexes = np.unique(frame.event_id, return_inverse=True)
            return tfidf.transform([docs[e] for e in unique])[indexes]
        text_train, text_valid = words(train), words(valid)
        for mode in MODES:
            pred = valid[["event_id","complex_id","target","tags",mode]].copy()
            pred = pred.rename(columns={mode:"observed"})
            pred["fold"], pred["mode"] = day, mode
            lag = "weekly_"+mode
            y = np.log1p(train[mode].to_numpy()) - np.log1p(train[lag].to_numpy())
            pred["weekly"] = valid[lag]
            for arm in ("native_cue","text","text_names","text_roles"):
                def design(frame):
                    records = features(frame,mode,"native_cue")
                    if arm in ("text_names","text_roles"):
                        for record,row in zip(records,frame.itertuples()):
                            if row.tags:
                                record.update(mentioned=1.0,named_complex=str(row.complex_id))
                                if arm=="text_roles":
                                    record.update({"role_"+tag:1.0 for tag in row.tags})
                    return records
                vector = DictVectorizer()
                x = vector.fit_transform(design(train))
                v = vector.transform(design(valid))
                if arm!="native_cue":
                    x = sparse.hstack([x,text_train],format="csr")
                    v = sparse.hstack([v,text_valid],format="csr")
                fit = Ridge(alpha=ALPHA,solver="lsqr",tol=1e-8).fit(x,y)
                pred[arm] = np.maximum(0,np.expm1(np.log1p(valid[lag])+fit.predict(v)))
            frames.append(pred)
    frame = pd.concat(frames,ignore_index=True)
    if frame.duplicated(["mode","complex_id","target"]).any():
        raise ValueError("Repeated score key")
    rows = []
    for fold,group in [*frame.groupby("fold"),("pooled",frame)]:
        for mode, subset in group.groupby("mode"):
            for scope, cases in [("network",subset),("rule_named",subset[subset.tags.str.len()>0])]:
                for arm in ("weekly","native_cue","text","text_names","text_roles"):
                    errors = (cases.observed-cases[arm]).abs()
                    if not np.isfinite(errors).all():
                        raise ValueError("Incomplete scores")
                    rows.append(dict(fold=fold,mode=mode,scope=scope,arm=arm,rows=len(cases),
                                     events=cases.event_id.nunique(),dates=cases.target.dt.date.nunique(),
                                     mae=errors.mean()))
    pd.DataFrame(rows).to_csv(output/"scores.csv",index=False,float_format="%.9f")
    frame["tags"] = frame.tags.map("|".join)
    frame.to_csv(output/"predictions.csv.gz",index=False,float_format="%.10f")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.cache,args.output)
