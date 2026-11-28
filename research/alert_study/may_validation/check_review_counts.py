"""Recover target and weekly counts for the three text-reviewed exchange cases."""
import argparse
import csv
import gzip
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_sample import api, inputs, write_csv


def verify(cache, output):
    panel = HERE / "panel"
    review = HERE / "name_support"
    source_files = json.loads((panel / "sources.json").read_text())
    with (review / "review_traces.csv").open() as stream:
        traces = list(csv.DictReader(stream))
    dock_map = defaultdict(set)
    with (panel / "docks.csv").open() as stream:
        for row in csv.DictReader(stream):
            dock_map[row["complex_id"]].add(row["dock"])

    needed, hourly = defaultdict(set), {}
    for row in traces:
        complex_id = row["complex_id"]
        if set(row["assigned_docks"].split("|")) != dock_map[complex_id]:
            raise ValueError("Trace dock IDs differ from the fixed map")
        target = datetime.fromisoformat(row["target_hour"])
        for period, time in (("target", target), ("weekly", target - timedelta(days=7))):
            hour = time.strftime("%Y-%m-%d %H")
            needed[hour].add(complex_id)
            hourly[(row["event_id"], complex_id, period)] = hour

    docks_by_hour = {h: set.union(*(dock_map[c] for c in ids)) for h, ids in needed.items()}
    seen, starts, fares, manifest = {}, Counter(), defaultdict(Decimal), {}
    for month in sorted({h[:7].replace("-", "") for h in needed}):
        name = month + ".zip"
        archive = inputs(cache, manifest, name, source_files[name]["url"])
        if manifest[name] != source_files[name]:
            raise ValueError(f"Different source archive for {month}")
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not member.endswith(".csv") or member.startswith("__MACOSX/"):
                    continue
                with z.open(member) as data:
                    records = csv.reader(io.TextIOWrapper(data, encoding="utf-8-sig"))
                    header = next(records)
                    ride, when, dock = map(header.index, ("ride_id", "started_at", "start_station_id"))
                    for r in records:
                        h = r[when][:13].replace("T", " ")
                        if h not in needed or r[dock].strip() not in docks_by_hour[h]:
                            continue
                        key, observation = r[ride], (r[when], r[dock].strip())
                        if not key or (key in seen and seen[key] != observation):
                            raise ValueError("Missing or conflicting ride identity")
                        if key not in seen:
                            seen[key] = observation
                            starts[h, observation[1]] += 1

    fare_rows = Counter()
    for h, ids in sorted(needed.items()):
        stamp = h.replace(" ", "T") + ":00:00"
        members = ",".join("'" + c + "'" for c in sorted(ids))
        query = api("5wq4-mkjj", transit_mode="subway", transit_timestamp=stamp,
                    **{"$where": f"station_complex_id in ({members})",
                       "$order": "station_complex_id,payment_method,fare_class_category", "$limit": 1000})
        name = "review-fares-" + h.replace(" ", "-") + ".json"
        rows = json.loads(inputs(cache, manifest, name, query).read_bytes())
        if len(rows) >= 1000 or {r["station_complex_id"] for r in rows} != ids:
            raise ValueError(f"Incomplete fare query at {h}")
        for r in rows:
            if r["transit_mode"] != "subway" or datetime.fromisoformat(r["transit_timestamp"]) != datetime.fromisoformat(stamp):
                raise ValueError("Fare query returned a different time or mode")
            key = h, r["station_complex_id"]
            fare_rows[key] += 1
            fares[key] += Decimal(r["ridership"])

    results, dock_counts = [], []
    with gzip.open(panel / "primary.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            event, cid = r["event_id"], r["complex_id"]
            if (event, cid, "target") not in hourly:
                continue
            for period, prefix in (("target", ""), ("weekly", "weekly_")):
                h = hourly[event, cid, period]
                bike = sum(starts[h, dock] for dock in dock_map[cid])
                if (fares[h, cid], bike) != (Decimal(r[prefix + "rail"]), int(r[prefix + "bike"])):
                    raise ValueError(f"Panel mismatch for {event}, {cid}, {period}")
                results.append(dict(event_id=event, complex_id=cid, period=period, local_hour=h,
                                    fare_components=fare_rows[h, cid], rail_entries=int(fares[h, cid]),
                                    assigned_docks=len(dock_map[cid]), bike_starts=bike))
                dock_counts.extend(dict(event_id=event, complex_id=cid, period=period, dock=dock,
                                        bike_starts=starts[h, dock]) for dock in sorted(dock_map[cid]))
    if len(results) != len(hourly) or len(hourly) != 2 * len(traces):
        raise ValueError("Missing or duplicated review keys")
    receipt = review / "raw_counts/sources.json"
    if receipt.exists() and manifest != json.loads(receipt.read_text()):
        raise ValueError("Source captures differ from the published verification")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "counts.csv", results)
    write_csv(output / "docks.csv", dock_counts)
    (output / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    verify(a.cache, a.output)
