"""Summarize availability and weekly seasonal error on identical station-hours."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from audit_cohort import ROOT


def supported(tags, disjoint):
    sets = [set(s.split("|")) for s in tags if s]
    return any((a.isdisjoint(b) if disjoint else a != b) for a,b in combinations(sets,2))


def run(panel_dir, output):
    panel = pd.concat([pd.read_csv(p,keep_default_na=False,na_values={k:[""] for k in ["rail","bike","weekly_rail","weekly_bike"]})
                       for p in sorted(panel_dir.glob("2026-??.csv.gz"))],ignore_index=True)
    if panel.duplicated(["hour","complex_id"]).any():
        raise ValueError("Duplicate complex-hour")
    events = pd.read_csv(panel_dir/"events.csv")
    effective = pd.read_csv(panel_dir/"effective_calls.csv")
    raw = pd.read_csv(ROOT/"role_screen/mentions.csv").dropna(subset=["complex_id"])
    role_sets = raw.groupby(["event_id","complex_id"]).role.agg(lambda a:"|".join(sorted(set(a))))
    panel["month"] = panel.hour.str[:7]
    summary, cases = [], []
    for month, rows in panel.groupby("month"):
        observed = rows.dropna(subset=["rail","bike"])
        matched = observed.dropna(subset=["weekly_rail","weekly_bike"])
        m_events = events.loc[events.hour.str.startswith(month)]
        summary.append(dict(month=month,first_origins=int(events.issue_local.str.startswith(month).sum()),
                            unique_hours=rows.hour.nunique(),station_hours=len(rows),with_both=len(observed),
                            weekly_matched=len(matched),complexes=rows.complex_id.nunique(),overlapping=int((rows.overlap>1).sum()),
                            weekly_rail_mae=(matched.rail-matched.weekly_rail).abs().mean(),
                            weekly_bike_mae=(matched.bike-matched.weekly_bike).abs().mean()))
        for eid in set(m_events.event_id) & set(effective.event_id):
            event = m_events.loc[m_events.event_id==eid].iloc[0]
            query = effective.loc[effective.event_id==eid]
            assigned = rows[(rows.event_id==eid)&rows.roles.ne("")]
            both = assigned.dropna(subset=["rail","bike"])
            lagged = both.dropna(subset=["weekly_rail","weekly_bike"])
            calls = set(query.loc[query.effective_calls.gt(0),"complex_id"])
            for label,mode in (("disjoint",True),("unequal",False)):
                if not supported(role_sets.loc[eid],mode):
                    continue
                cases.append(dict(event_id=eid,month=month,hour=event.hour,definition=label,
                                  assigned=int(supported(assigned.roles,mode)),
                                  both_targets=int(supported(both.roles,mode)),
                                  weekly_matched=int(supported(lagged.roles,mode)),
                                  matched_with_calls=int(supported(lagged.loc[lagged.complex_id.isin(calls),"roles"],mode))))
    pd.DataFrame(summary).to_csv(output/"monthly.csv",index=False,float_format="%.8f")
    pd.DataFrame(cases).sort_values(["event_id","definition"]).to_csv(output/"pair_support.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel_dir",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.panel_dir,args.output)
