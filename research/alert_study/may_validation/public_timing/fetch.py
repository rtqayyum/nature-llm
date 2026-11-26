"""Capture public archive metadata for the frozen May input-timing audit."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
RAIL = "https://data.ny.gov/resource/5wq4-mkjj.json"
FIELDS = ("min(:created_at) as first_created,max(:created_at) as last_created,"
          "min(:updated_at) as first_updated,max(:updated_at) as last_updated,count(*) as fare_rows")


def rail_record(item):
    label, condition = item
    url = RAIL + "?" + urlencode({"$select": FIELDS, "$where": "transit_mode='subway' AND " + condition})
    with urlopen(url, timeout=120) as response:
        raw = response.read()
    row, = json.loads(raw)
    if int(row["fare_rows"]) < 1:
        raise ValueError(f"No fare records for {label}")
    return label, dict(url=url, sha256=hashlib.sha256(raw).hexdigest(), **row)


def bike_record(item):
    name, expected = item
    h = hashlib.sha256()
    size = 0
    with urlopen(expected["url"], timeout=180) as source:
        for buf in iter(lambda: source.read(2**20), b""):
            h.update(buf)
            size += len(buf)
        modified = source.headers["Last-Modified"]
        etag = source.headers["ETag"]
    if not modified or not etag or h.hexdigest() != expected["sha256"] or size != expected["bytes"]:
        raise ValueError(f"Trip archive differs from study manifest for {name}")
    return name, dict(**expected, last_modified=modified, etag=etag)


def collect(out):
    manifest = {}
    for sources in [HERE.parents[1]/"winter_spring/sources.json", HERE.parent/"panel/sources.json"]:
        manifest.update(json.loads(sources.read_text()))
    days = [date(2026, 4, 24) + timedelta(days=n) for n in range(31)]
    queries = [(str(d), f"transit_timestamp >= '{d}T00:00:00' AND "
                        f"transit_timestamp < '{d + timedelta(days=1)}T00:00:00'") for d in days]
    queries.append(("geometry", "transit_timestamp IN ('2026-04-29T08:00:00','2026-04-29T17:00:00')"))
    with ThreadPoolExecutor(3) as bikes, ThreadPoolExecutor(4) as rail:
        bike_data = bikes.map(bike_record, ((m, manifest[m]) for m in ("202603.zip","202604.zip","202605.zip")))
        rail_data = dict(rail.map(rail_record, queries))
        result = dict(bike=dict(bike_data), rail=rail_data)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    collect(parser.parse_args().output)
