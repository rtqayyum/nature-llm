"""Bound May station-role exchange support using first-notice text and paired keys."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from collect_text import ROOT
from extract_alert_roles import StationDictionary
from score import norm, resolve


def verified(path, manifest, key):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[key]["sha256"]:
        raise ValueError(f"Source mismatch for {key}")
    return data


def run(notices, stations, output):
    manifest = json.loads((ROOT / "panel/sources.json").read_text())
    alerts = {int(row["event_id"]): row for row in json.loads(verified(notices, manifest, "may_notices.json"))}
    dictionary = StationDictionary(json.loads(verified(stations, manifest, "stations")))
    # Search each alias independently. A longer matching name may contain another valid name.
    patterns = {name: re.compile(r"(?<![\w/-])" + re.escape(name).replace(r"\-", "[-/]") + r"(?![\w/-])")
                for name in dictionary.aliases}
    control = ROOT / "controls/operator_rail.csv.gz"
    receipt = json.loads((control.parent / "receipt.json").read_text())
    verified(control, receipt["files"], control.name)
    paired = defaultdict(set)
    with gzip.open(control, "rt") as stream:
        for row in csv.DictReader(stream):
            paired[int(row["event_id"])].add(int(row["complex_id"]))
    with (ROOT / "origins.csv").open() as stream:
        origins = {int(row["event_id"]): row for row in csv.DictReader(stream)}
    if set(alerts) != set(origins) or sum(map(len, paired.values())) != receipt["evaluation_rows"]:
        raise ValueError("Different May population")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as dst:
        writer = csv.writer(dst, lineterminator="\n")
        writer.writerow(["event_id", "date", "episode", "fault_response", "audit_exposed",
                         "paired_complexes", "eligible_complexes", "eligible_complex_ids"])
        for event, assigned in sorted(paired.items()):
            alert, origin = alerts[event], origins[event]
            source = norm(alert["header"] + " " + alert.get("description", ""))
            routes = set(alert.get("affected", "").split(" | "))
            matches = {int(cid) for alias, pattern in patterns.items()
                       if pattern.search(source) and (cid := resolve(alias, routes, dictionary))}
            retained = sorted(assigned & matches)
            exposed = any(int(origin[k]) for k in ("audit_case", "audit_text", "audit_episode"))
            writer.writerow([event, origin["hour"][:10], origin["episode"],
                             int(origin["class_cue"] == "fault or response wording"), int(exposed),
                             len(assigned), len(retained), "|".join(map(str, retained))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("notices", "stations", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    run(args.notices, args.stations, args.output)
