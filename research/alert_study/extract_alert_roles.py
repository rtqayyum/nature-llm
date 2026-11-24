"""Outcome-blind template extraction on the January–April 2026 NYC first-notice archive."""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from audit_february import CUES, notice
from build_sample import LOCAL, api, inputs, write_csv

ROOT = Path(__file__).resolve().parent
TOKEN = r"(@\d+@)"
# Capturing groups are only station placeholders. Keep links within a single clause.
RULES = [
    ("express boundary", r"(?:run(?:ning)? express|express (?:track|service)) from " + TOKEN + " to " + TOKEN),
    ("local boundary", r"(?:run(?:ning)? local|local track) from " + TOKEN + " to " + TOKEN),
    ("reroute boundary", r"(?:running via|run via|running on|run on) the [\w /]+? (?:line|lines) from " + TOKEN + " to " + TOKEN),
    ("suspension boundary", r"no [^.!?\n]{0,75}?service [^.!?\n]{0,35}?between " + TOKEN + " and " + TOKEN),
    ("short turn", r"(?:last stop on [^.!?\n]{1,75}? (?:is|will be)|(?:trains? (?:will |are )?)?(?:end|ending|terminate|terminating) at) " + TOKEN),
    ("boarding platform", r"at " + TOKEN + r", (?:board|boarding|catch)\b"),
    ("boarding platform", r"\bboard [^.!?\n]{1,75}?platform at " + TOKEN),
    ("transfer", r"\btransfer (?:between [^.!?\n]{1,55}? at |at )" + TOKEN),
    ("skipped", r"\b(?:trains (?:are |will )?(?:skip|skipping|bypass|bypassing)) " + TOKEN),
]
PATTERNS = [(role, re.compile(rule, re.I)) for role, rule in RULES]
PERIOD = ["2026-01-01T05:00:00", "2026-02-01T05:00:00", "2026-03-01T05:00:00",
          "2026-04-01T04:00:00", "2026-05-01T04:00:00"]


class StationDictionary:
    def __init__(self, rows):
        self.aliases = defaultdict(lambda: defaultdict(set))
        for row in rows:
            self.aliases[row["stop_name"].casefold().replace("/", "-")][row["complex_id"]].update(row["daytime_routes"].split())
        names = sorted(self.aliases, key=lambda s: (-len(s), s))
        variants = [re.escape(name).replace(r"\-", "[-/]") for name in names]
        self.pattern = re.compile(r"(?<![\w/-])(?:" + "|".join(variants) + r")(?![\w/-])", re.I)

    def match(self, source, routes):
        matches = list(self.pattern.finditer(source))
        index = iter(range(len(matches)))
        masked = self.pattern.sub(lambda _: f"@{next(index)}@", source)
        for role, rule in PATTERNS:
            for clause in rule.finditer(masked):
                for token in clause.groups():
                    hit = matches[int(token[1:-1])]
                    choices = self.aliases[hit[0].casefold().replace("/", "-")]
                    filtered = [key for key, lines in choices.items() if lines & routes]
                    selected = list(choices) if len(choices) == 1 else filtered
                    cid = selected[0] if len(selected) == 1 else ""
                    yield dict(complex_id=cid, role=role, station_text=hit[0], start=hit.start(),
                               end=hit.end(), candidates="|".join(sorted(choices)),
                               resolution=("unique" if len(choices)==1 else
                                           "daytime route" if cid else "unresolved"))


def contrast(roles):
    return any(a.isdisjoint(b) for a,b in combinations(roles.values(), 2))


def run(cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    def read(name, url):
        return json.loads(inputs(cache, manifest, name, url).read_bytes())
    stations = StationDictionary(read("stations", api("39hk-dx4f")))
    inventory, mentions, seen, feb = [], [], set(), {}
    miss_sample = []
    for begin, end in zip(PERIOD, PERIOD[1:]):
        month = begin[:7]
        query = {"$where": f"agency='NYCT Subway' AND update_number=0 AND date >= '{begin}' AND date < '{end}'",
                 "$order":"date,event_id,alert_id", "$limit":20000}
        alerts = read(month + ".json", api("7kct-peq7", **query))
        if len(alerts) >= query["$limit"]:
            raise ValueError("Incomplete archive query")
        candidates = []
        for alert in alerts:
            eid = alert["event_id"]
            issue = datetime.fromisoformat(alert["date"]).replace(tzinfo=timezone.utc).astimezone(LOCAL)
            if eid in seen or issue.strftime("%Y-%m") != month or int(alert["update_number"]) != 0:
                raise ValueError("Repeated incident, out-of-period time or noninitial notice")
            seen.add(eid)
            roles = defaultdict(set)
            routes = set(alert.get("affected","").split(" | "))
            # A planned-work reminder can be embedded inside an unrelated first notice.
            for field in ("header", "description"):
                raw = alert.get(field,"")
                parts = re.split(r"planned (?:service|work) reminder", raw, flags=re.I, maxsplit=1)
                for hit in stations.match(parts[0], routes):
                    mentions.append(dict(event_id=eid, alert_id=alert["alert_id"], field=field, **hit))
                    if hit["complex_id"]:
                        roles[hit["complex_id"]].add(hit["role"])
                if len(parts)>1:
                    break
            inventory.append(dict(event_id=eid, alert_id=alert["alert_id"], issue_local=issue.isoformat(),
                                  status=alert["status_label"], routes=alert.get("affected",""),
                                  resolved_complexes=len(roles), distinct_role_pair=int(contrast(roles))))
            if month in ("2026-03", "2026-04") and not contrast(roles):
                if sum(bool(re.search(cue, notice(alert), re.I)) for cue in CUES) >= 2:
                    candidates.append(eid)
            if month=="2026-02":
                feb[eid] = roles
        candidates.sort(key=lambda e: hashlib.sha256(("role-screen-miss-v1/"+e).encode()).digest())
        for eid in candidates[:6]:
            miss_sample.append(dict(month=month, event_id=eid, stratum_size=len(candidates)))
    audit = []
    with (ROOT/"february/review.csv").open() as f:
        for row in csv.DictReader(f):
            eid = row["event_id"]
            if row["complex_a"]:
                for slot in ("a", "b"):
                    cid, expected = row["complex_"+slot], row["role_"+slot]
                    actual = sorted(feb[eid].get(cid, set()))
                    audit.append(dict(event_id=eid, complex_id=cid, reviewed_role=expected,
                                      parsed_roles="|".join(actual), recovered=int(expected in actual)))
    expected = ROOT/"role_screen/sources.json"
    if expected.exists() and json.loads(expected.read_text()) != manifest:
        raise ValueError("Source drift. Reconcile the archive before replacing any result.")
    write_csv(out/"miss_sample.csv", miss_sample)
    write_csv(out/"events.csv", inventory)
    write_csv(out/"mentions.csv", mentions)
    write_csv(out/"february_comparison.csv", audit)
    (out/"sources.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cache, args.output)
