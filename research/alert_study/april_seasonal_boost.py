"""Exploratory seasonal-residual boosting control on the existing April development split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from april_notice_pilot import load_split, MODES


def run(cache, panel, output):
    output.mkdir(exist_ok=True,parents=True)
    train,test,docs,purged = load_split(cache,panel)
    train_ids = np.sort(train.event_id.unique())
    ids = np.union1d(train_ids,test.event_id.unique())
    words = TfidfVectorizer(ngram_range=(1,2),min_df=2,token_pattern=r"(?u)\b\w+\b")
    train_text = words.fit_transform([docs[e] for e in train_ids])
    reducer = TruncatedSVD(n_components=16,random_state=20261125)
    reducer.fit(train_text)
    all_text = pd.DataFrame(reducer.transform(words.transform([docs[e] for e in ids])),index=ids)
    # Use stable geography instead of the arbitrary numerical order of complex IDs.
    manifest = json.loads((panel/"sources.json").read_text())
    coords = {}
    for month,day in enumerate(["2025-12-31","2026-01-28","2026-02-25","2026-03-25"],1):
        for hour in ("08","17"):
            name = f"geo-{day}-{hour}.json"
            path = cache/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
                raise ValueError("Different preperiod station coordinates")
            geo = pd.read_json(path)
            for r in geo.itertuples():
                coords[month,int(r.station_complex_id)] = (r.latitude,r.longitude)
    role_vocab = sorted({t for tags in train.tags for t in tags.split("|") if t})
    categories = ["clock","weekday","routes","status","cue"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1).fit(train[categories])
    def features(frame,mode):
        geo = np.asarray([coords[int(r.hour[5:7]),r.complex_id] for r in frame.itertuples()])
        cat = enc.transform(frame[categories])
        counts = np.log1p(frame["weekly_"+mode]).to_numpy()
        base = np.column_stack([counts,geo,cat])
        text = all_text.loc[frame.event_id].to_numpy()
        names = frame.tags.ne("").to_numpy(dtype=float)[:,None]
        labels = [set(s.split("|")) for s in frame.tags]
        roles = np.asarray([[t in record for t in role_vocab] for record in labels],dtype=float)
        return (base, np.column_stack([base,text]),np.column_stack([base,text,names]),
                np.column_stack([base,text,names,roles]))
    arms = ("native","text","text_names","text_roles")
    rows, predictions = [], []
    for mode in MODES:
        fit_features, eval_features = features(train,mode),features(test,mode)
        lag = "weekly_"+mode
        pred = test[["event_id","complex_id","hour","tags"]].copy()
        pred["mode"],pred["actual"],pred["weekly"] = mode,test[mode],test[lag]
        for arm,x,v in zip(arms,fit_features,eval_features):
            # Category columns follow the lag and the two geographical coordinates.
            model = HistGradientBoostingRegressor(loss="absolute_error",max_iter=100,learning_rate=0.05,
                max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,early_stopping=False,
                categorical_features=list(range(3,3+len(categories))),random_state=20261125)
            model.fit(x,train[mode]-train[lag])
            pred[arm] = np.maximum(0,test[lag]+model.predict(v))
        predictions.append(pred)
        for scope,cases in (("network",pred),("rule_named",pred[pred.tags.ne("")])):
            for arm in ("weekly",*arms):
                err = (cases.actual-cases[arm]).abs()
                if not np.isfinite(err).all():
                    raise ValueError("Incomplete scores")
                rows.append(dict(mode=mode,scope=scope,arm=arm,rows=len(cases),events=cases.event_id.nunique(),
                                 dates=cases.hour.str[:10].nunique(),mae=err.mean()))
    result = pd.concat(predictions,ignore_index=True)
    pd.DataFrame(rows).to_csv(output/"boost_scores.csv",index=False,float_format="%.9f")
    result.to_csv(output/"boost_predictions.csv.gz",index=False,float_format="%.10f")
    result.loc[result.tags.ne("")].to_csv(output/"boost_named_predictions.csv",index=False,float_format="%.10f")
    print(f"fit {len(train)}, purge {purged}, score {len(test)}, tags {len(role_vocab)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("cache","panel","output"):
        p.add_argument(arg,type=Path)
    args=p.parse_args()
    run(args.cache,args.panel,args.output)
