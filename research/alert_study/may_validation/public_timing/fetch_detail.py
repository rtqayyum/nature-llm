"""Record creation and update times for the keys that pass the citywide timing screen."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from analyze import STUDY, as_time
from fetch import FIELDS, RAIL


def table(group, condition):
    url = RAIL + "?" + urlencode({"$select": group + "," + FIELDS, "$where": "transit_mode='subway' AND " + condition,
                                 "$group": group, "$order": group, "$limit": 15000})
    with urlopen(url, timeout=120) as response:
        raw = response.read()
    rows = json.loads(raw)
    if not 0 < len(rows) < 15000:
        raise ValueError("Empty or truncated response")
    return rows, dict(url=url, sha256=hashlib.sha256(raw).hexdigest(), groups=len(rows))


def collect(metadata, out):
    m = json.loads(metadata.read_text())
    wanted = {}
    with (STUDY/"origins.csv").open() as f:
        origins = {r["event_id"]: as_time(r["origin"]) for r in csv.DictReader(f)}
    with gzip.open(STUDY/"controls/operator_rail.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            lag = as_time(r["hour"]) - timedelta(days=7)
            d = str(lag.date())
            if origins[r["event_id"]] >= as_time(m["rail"][d]["first_created"]):
                wanted.setdefault(d,set()).add((lag.strftime("%Y-%m-%dT%H"), r["complex_id"]))
    sources = {}
    records = []
    def lag_day(d):
        cond = f"transit_timestamp >= '{d}T00:00:00' AND transit_timestamp < '{date.fromisoformat(d)+timedelta(days=1)}T00:00:00'"
        return d, table("transit_timestamp,station_complex_id", cond)
    with ThreadPoolExecutor(3) as pool:
        for d, (rows, source_receipt) in pool.map(lag_day, sorted(wanted)):
            sources[d] = source_receipt
            used = {(r["transit_timestamp"][:13],r["station_complex_id"]): r for r in rows
                    if (r["transit_timestamp"][:13],r["station_complex_id"]) in wanted[d]}
            if set(used) != wanted[d]:
                raise ValueError(f"Missing weekly keys on {d}")
            records.extend(used.values())
    out.mkdir(parents=True, exist_ok=True)
    with (out/"weekly_source_times.csv.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as gz:
        with io.TextIOWrapper(gz, newline="") as f:
            w = csv.DictWriter(f,fieldnames=["transit_timestamp","station_complex_id","fare_rows","first_created","last_created","first_updated","last_updated"],lineterminator="\n")
            w.writeheader()
            w.writerows(records)
    manifest = {}
    for path in (STUDY.parent/"winter_spring/sources.json", STUDY/"panel/sources.json"):
        manifest.update(json.loads(path.read_text()))
    locations = []
    for d in ("2026-03-25","2026-04-29"):
        cond = f"transit_timestamp IN ('{d}T08:00:00','{d}T17:00:00')"
        rows, sources[d+"-geometry"] = table("station_complex_id,latitude,longitude", cond)
        expected = set()
        for hour in ("08", "17"):
            meta = manifest[f"geo-{d}-{hour}.json"]
            with urlopen(meta["url"], timeout=120) as response:
                body = response.read()
            if hashlib.sha256(body).hexdigest() != meta["sha256"]:
                raise ValueError(f"Different coordinate snapshot on {d} at {hour}")
            expected.update((r["station_complex_id"], float(r["latitude"]), float(r["longitude"]))
                            for r in json.loads(body))
        actual = {(r["station_complex_id"], float(r["latitude"]), float(r["longitude"])) for r in rows}
        if actual != expected or len({key[0] for key in actual}) != len(rows):
            raise ValueError(f"Coordinates differ from the frozen map for {d}")
        locations.extend(dict(source_day=d, **r) for r in rows)
    with (out/"geometry_source_times.csv").open("w", newline="") as f:
        w = csv.DictWriter(f,fieldnames=["source_day","station_complex_id","latitude","longitude","fare_rows","first_created","last_created","first_updated","last_updated"],lineterminator="\n")
        w.writeheader()
        w.writerows(locations)
    (out/"detail_sources.json").write_text(json.dumps(sources,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metadata",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    collect(a.metadata,a.output)
