"""Summarize frozen job 3100357; convexity bounds use archived MAEs only."""
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    groups = defaultdict(list)
    with (ROOT/'results/development/window_scores.csv').open() as f:
        for row in csv.DictReader(f):
            groups[row['dataset'],row['objective']].append(row)
    print('| Domain | Objective | Local MAE | Full-window switch MAE | Half-correction MAE upper bound | Observed relative gain of switch |')
    print('|---|---|---:|---:|---:|---:|')
    for (domain,arm),rows in groups.items():
        if len({(r['window'],r['seed']) for r in rows})!=12 or len(rows)!=12:
            raise ValueError('Incomplete development window/seed matrix')
        local = mean(float(r['local_mae']) for r in rows)
        switch = mean(float(r['deployed_mae']) for r in rows)
        print(f'| {domain} | {arm} | {local:.3f} | {switch:.3f} | {(local+switch)/2:.3f} | {100*(1-switch/local):.2f}% |')


if __name__=='__main__':
    main()
