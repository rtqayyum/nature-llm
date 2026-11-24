"""Chronological development comparison on the released February NYC alert panel."""
import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
INPUT_SHA = {
    "station_hours.csv.gz": "0b20a8f51d87e401fcb0318ec8a32d10d116f8a9",
    "events.csv": "6857d4fa890a2dcac6fcbc6bb37ac655878fa82d",
    "review.csv": "819d38e8b31bf956771403c495a63833962beae7",
}
MODES = ("rail", "bike")
ARMS = ("weekly", "history", "native_cue", "names", "roles")
ALPHA = 10.0


def load_panel(folder):
    for name, expected in INPUT_SHA.items():
        data = (folder / name).read_bytes()
        header = f"blob {len(data)}\0".encode()
        if hashlib.sha1(header + data).hexdigest() != expected:
            raise ValueError(f"Input differs from the released February panel: {name}")
    events = pd.read_csv(folder / "events.csv", keep_default_na=False)
    panel = pd.read_csv(folder / "station_hours.csv.gz")
    panel = panel.merge(events[["event_id", "issue_local", "target_local", "episode_24h",
                                "status", "routes", "cause"]],
                        left_on="first_event", right_on="event_id", validate="many_to_one")
    if (panel.duplicated(["hour", "complex_id"]).any() or
            not panel["hour"].eq(panel["target_local"]).all()):
        raise ValueError("Complex-hour keys or event target times disagree")
    panel["issue"] = pd.to_datetime(panel["issue_local"])
    panel["target"] = pd.to_datetime(panel["target_local"])
    panel["end"] = panel["target"] + pd.Timedelta(hours=1)
    if not ((panel["target"] - panel["issue"] > pd.Timedelta(0)) &
            (panel["target"] - panel["issue"] <= pd.Timedelta(hours=1))).all():
        raise ValueError("Invalid forecast chronology")

    review = pd.read_csv(folder / "review.csv").dropna(subset=["complex_a", "complex_b"])
    annotations = []
    for row in review.itertuples():
        annotations.extend([(row.event_id, int(row.complex_a), row.role_a, row.role_b),
                            (row.event_id, int(row.complex_b), row.role_b, row.role_a)])
    mentions = pd.DataFrame(annotations, columns=["event_id", "complex_id", "role", "swapped"])
    panel = panel.merge(mentions, how="left", on=["event_id", "complex_id"], validate="many_to_one")
    panel[["role", "swapped"]] = panel[["role", "swapped"]].fillna("")

    # Missing targets and weekly lags for either mode are masked identically for every arm.
    panel = panel.dropna(subset=["rail", "bike", "weekly_rail", "weekly_bike"]).copy()
    mentioned = panel.loc[panel.role.ne("")].groupby("event_id").size()
    panel["pair"] = panel.role.ne("") & panel.event_id.isin(mentioned[mentioned.eq(2)].index)
    return panel.sort_values(["target", "complex_id"]).reset_index(drop=True), events


def get_fold(panel, inventory, day):
    cutoff = pd.Timestamp(f"2026-02-{day}T00:00:00-05:00")
    stop = cutoff + pd.Timedelta(days=7)
    valid = panel.loc[panel.issue.ge(cutoff) & panel.end.le(stop)].copy()
    # Use the entire notice inventory for recurrence purging, even if an alert lost all rows.
    issues = pd.to_datetime(inventory.issue_local)
    episodes = inventory.loc[issues.ge(cutoff) & issues.lt(stop), "episode_24h"]
    train_before_purge = panel.loc[panel.end.le(cutoff)]
    train = train_before_purge.loc[~train_before_purge.episode_24h.isin(episodes)].copy()
    if train.end.max() > valid.issue.min() or set(train.episode_24h) & set(valid.episode_24h):
        raise ValueError("Overlapping fit and evaluation periods or incident groups")
    return train, valid, len(train_before_purge) - len(train)


def features(frame, mode, arm, swap=False):
    lag = "weekly_" + mode
    records = []
    for row in frame.itertuples():
        counts = np.log1p(getattr(row, lag))
        record = {"weekly_log_count": counts, "complex": str(row.complex_id),
                  "hour": str(row.target.hour), "weekday": str(row.target.dayofweek),
                  "hour_of_week": str(row.target.dayofweek * 24 + row.target.hour)}
        if arm in {"native_cue", "names", "roles"}:
            record.update(status=row.status, routes=row.routes, cue=row.cause)
        if arm in {"names", "roles"} and row.role:
            record.update(mentioned=1.0, named_complex=str(row.complex_id))
        if arm == "roles" and row.role:
            record["station_role"] = row.swapped if swap else row.role
        records.append(record)
    return records


def forecasts(train, valid, mode):
    train_lag = np.log1p(train["weekly_" + mode].to_numpy())
    target = np.log1p(train[mode].to_numpy()) - train_lag
    lag = valid["weekly_" + mode].to_numpy()
    predictions = {"weekly": lag}
    for arm in ARMS[1:]:
        dictionary = DictVectorizer()
        x = dictionary.fit_transform(features(train, mode, arm))
        fit = Ridge(alpha=ALPHA, solver="lsqr", tol=1e-8).fit(x, target)
        def predict(frame, swap=False):
            design = dictionary.transform(features(frame, mode, arm, swap))
            base = np.log1p(frame["weekly_" + mode].to_numpy())
            return np.maximum(0, np.expm1(base + fit.predict(design)))
        predictions[arm] = predict(valid)
        if arm == "roles":
            swapped = np.full(len(valid), np.nan)
            pair = valid.pair.to_numpy()
            if pair.any():
                swapped[pair] = predict(valid.loc[pair], swap=True)
            predictions["swapped"] = swapped
    return predictions


def score(frame):
    records = []
    for fold, subset in [*frame.groupby("fold"), ("pooled", frame)]:
        for scope, rows in [("all", subset), ("complete_named_pairs", subset.loc[subset.pair])]:
            for mode, sample in rows.groupby("mode"):
                for arm in (*ARMS, "swapped"):
                    if arm == "swapped" and scope == "all":
                        continue
                    error = (sample[mode] - sample[arm]).abs()
                    if not np.isfinite(error).all():
                        raise ValueError("Missing paired predictions")
                    records.append(dict(fold=fold, scope=scope, mode=mode, arm=arm,
                                        rows=len(sample), events=sample.event_id.nunique(),
                                        days=sample.target.dt.date.nunique(),
                                        sum_absolute_error=error.sum(), mae=error.mean()))
    return pd.DataFrame(records)


def run(inputs, output, keep_predictions=False):
    panel, inventory = load_panel(inputs)
    output.mkdir(parents=True, exist_ok=True)
    frames, support = [], []
    for day in (15, 22):
        train, valid, purged = get_fold(panel, inventory, day)
        support.append(dict(fold=day, fit_rows=len(train), fit_rows_purged=purged,
                            validation_rows=len(valid), fit_named=int(train.role.ne("").sum()),
                            validation_named=int(valid.role.ne("").sum()),
                            fit_complete_pairs=train.loc[train.pair, "event_id"].nunique(),
                            validation_complete_pairs=valid.loc[valid.pair, "event_id"].nunique()))
        for mode in MODES:
            pred = valid[["event_id", "complex_id", "target", "role", "swapped", "pair", "rail", "bike"]].copy()
            pred = pred.rename(columns={"swapped": "exchanged_role"})
            pred["fold"], pred["mode"] = day, mode
            for arm, values in forecasts(train, valid, mode).items():
                pred[arm] = values
            frames.append(pred)
    results = pd.concat(frames, ignore_index=True)
    if results.duplicated(["target", "complex_id", "mode"]).any():
        raise ValueError("Duplicate evaluation units")
    pd.DataFrame(support).to_csv(output / "folds.csv", index=False)
    score(results).to_csv(output / "scores.csv", index=False, float_format="%.10f")
    results.loc[results.pair].to_csv(output / "pair_predictions.csv", index=False, float_format="%.10f")
    if keep_predictions:
        # Reproducible gzip bytes, independent of output path and run timestamp.
        import io
        with (output / "predictions.csv.gz").open("wb") as raw, \
                gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped, \
                io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
            results.to_csv(stream, index=False, float_format="%.10f", quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "february")
    parser.add_argument("--output", type=Path, default=ROOT / "february_pilot")
    parser.add_argument("--keep-predictions", action="store_true")
    args = parser.parse_args()
    run(args.inputs, args.output, args.keep_predictions)
