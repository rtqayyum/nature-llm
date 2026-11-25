"""Fit the registered May non-Qwen controls and freeze predictions without scoring them."""
import argparse
import gzip
import hashlib
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from audit_cohort import cause_screen
from audit_february import notice

CATEGORIES = ["clock","weekday","routes","status","cue"]
KEY = ["hour","complex_id","event_id"]
MODES = ("rail","bike")
SEED = 20261125


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_split(cache):
    winter = HERE.parent / "winter_spring"
    manifest = {**json.loads((winter/"sources.json").read_text()),
                **json.loads((HERE/"panel/sources.json").read_text())}
    def read(name):
        path = cache/name
        if sha(path) != manifest[name]["sha256"]:
            raise ValueError("Different source: "+name)
        return json.loads(path.read_bytes())
    rows, coords = [], {}
    for m,day in enumerate(("2025-12-31","2026-01-28","2026-02-25","2026-03-25","2026-04-29"),1):
        rows.extend(r for r in read(f"versions-2026-{m:02}.json") if int(r["update_number"]) == 0)
        for h in ("08","17"):
            for r in read(f"geo-{day}-{h}.json"):
                coords[m,int(r["station_complex_id"])] = float(r["latitude"]),float(r["longitude"])
    episodes, previous, events = {}, {}, []
    for row in sorted(rows, key=lambda r: (r["date"],int(r["event_id"]))):
        eid = int(row["event_id"])
        ts = datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc)
        routes = row.get("affected","")
        key = routes, " ".join(row["header"].lower().split())
        if key not in previous or ts - previous[key] > timedelta(days=1):
            episodes[key] = eid
        previous[key] = ts
        events.append(dict(event_id=eid,episode=episodes[key],routes=routes,status=row["status_label"],
                           cue=cause_screen(notice(row)),doc=row["header"]+"\n"+row.get("description","")))
    events = pd.DataFrame(events).set_index("event_id", verify_integrity=True)
    ledger = pd.read_csv(HERE/"origins.csv",keep_default_na=False).set_index("event_id")
    for left, right in (("episode","episode"),("routes","routes"),("cue","class_cue")):
        if not events.loc[ledger.index,left].equals(ledger[right]):
            raise ValueError("Ledger mismatch for "+left)
    def enrich(frame):
        frame = frame.merge(events.drop(columns="doc"),how="left",left_on="event_id",right_index=True,validate="many_to_one")
        if frame.status.isna().any() or frame.duplicated(KEY[:2]).any():
            raise ValueError("Unmatched origin or duplicate station-hour")
        target = pd.to_datetime(frame.hour,utc=True).dt.tz_convert("America/New_York")
        frame["end"] = target + pd.Timedelta(hours=1)
        frame["clock"] = target.dt.hour.astype(str)
        frame["weekday"] = target.dt.dayofweek.astype(str)
        frame["how"] = 24*target.dt.dayofweek+target.dt.hour
        frame["tags"] = frame.roles.fillna("")
        xy = [coords[int(r.hour[5:7]),r.complex_id] for r in frame.itertuples()]
        frame[["lat","lon"]] = np.asarray(xy)
        return frame
    history = enrich(pd.concat([pd.read_csv(f) for f in sorted(winter.glob("2026-??.csv.gz"))],ignore_index=True))
    mask = ["rail","bike","weekly_rail","weekly_bike"]
    test = enrich(pd.read_csv(HERE/"panel/primary.csv.gz").dropna(subset=mask))
    # No May targets are passed to feature construction, fitting or serialization.
    test = test.drop(columns=list(MODES))
    train, receipt = {}, {}
    for fit, end, fields in (("operator","2026-05-01",mask),("lag_free","2026-04-01",list(MODES))):
        cutoff = pd.Timestamp(end,tz="America/New_York")
        pre = history.loc[history.end.le(cutoff)].dropna(subset=fields)
        frame = pre.loc[~pre.episode.isin(ledger.episode)].copy()
        if set(frame.episode)&set(ledger.episode) or frame.end.max()>pd.to_datetime(ledger.origin,utc=True).min():
            raise ValueError("Overlapping train and validation")
        train[fit] = frame
        receipt[fit] = dict(training_rows=len(frame),training_origins=frame.event_id.nunique(),
                            purged_rows=len(pre)-len(frame), complete_label_cutoff=cutoff.isoformat())
    return train,test,events.doc,receipt


class Design:
    def __init__(self, train, test, docs):
        train_ids = np.sort(train.event_id.unique())
        ids = np.union1d(train_ids,test.event_id.unique())
        words = TfidfVectorizer(ngram_range=(1,2),min_df=2,token_pattern=r"(?u)\b\w+\b")
        training = words.fit_transform(docs.loc[train_ids])
        svd = TruncatedSVD(n_components=16,random_state=SEED).fit(training)
        self.text = pd.DataFrame(svd.transform(words.transform(docs.loc[ids])),index=ids)
        self.categories = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1).fit(train[CATEGORIES])
        self.roles = sorted({tag for tags in train.tags for tag in tags.split("|") if tag})

    def matrix(self, frame, mode, fit, arm):
        columns = []
        if fit == "operator":
            columns.append(np.log1p(frame["weekly_"+mode]))
        columns.extend((frame[["lat","lon"]],self.categories.transform(frame[CATEGORIES])))
        if arm != "native":
            columns.append(self.text.loc[frame.event_id])
        if arm == "text_rules":
            columns.append(frame.tags.ne(""))
            tags = [set(s.split("|")) for s in frame.tags]
            columns.append(np.asarray([[t in r for t in self.roles] for r in tags]))
        return np.column_stack(columns)


def boost(x, y, v, fit):
    offset = 3 if fit == "operator" else 2
    model = HistGradientBoostingRegressor(loss="absolute_error",max_iter=100,learning_rate=0.05,
        max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,early_stopping=False,
        categorical_features=list(range(offset,offset+len(CATEGORIES))),random_state=SEED)
    return model.fit(x,y).predict(v)


def historical_median(train, test, mode):
    by_station_hour = train.groupby(["complex_id","how"])[mode].median()
    by_station = train.groupby("complex_id")[mode].median()
    keys = pd.MultiIndex.from_frame(test[["complex_id","how"]])
    value = pd.Series(by_station_hour.reindex(keys).to_numpy(),index=test.index)
    return value.fillna(test.complex_id.map(by_station)).fillna(train[mode].median())


def run(cache, out):
    out.mkdir(parents=True,exist_ok=True)
    train,test,docs,receipt = get_split(cache)
    receipt.update(evaluation_rows=len(test), files={}, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__)
    for fit, frame in train.items():
        design = Design(frame,test,docs)
        for mode in MODES:
            pred = test[KEY].copy()
            y = frame[mode]
            if fit == "operator":
                pred["weekly"] = test["weekly_"+mode]
                y = y-frame["weekly_"+mode]
            else:
                pred["median"] = historical_median(frame,test,mode)
            for arm in ("native","text","text_rules"):
                x = design.matrix(frame,mode,fit,arm)
                v = design.matrix(test,mode,fit,arm)
                value = boost(x,y,v,fit)
                if fit == "operator":
                    value += pred.weekly
                pred[arm] = np.maximum(0,value)
            if not np.isfinite(pred.iloc[:,len(KEY):]).all().all():
                raise ValueError("Incomplete forecast")
            path = out/f"{fit}_{mode}.csv.gz"
            with path.open("wb") as raw, gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding="utf-8",newline="") as dst:
                    pred.to_csv(dst,index=False,float_format="%.10f",lineterminator="\n")
            receipt["files"][path.name] = dict(sha256=sha(path),bytes=path.stat().st_size)
            print(f"Frozen {fit} {mode}, {len(pred)} station-hours",flush=True)
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.cache,args.output)
