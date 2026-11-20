"""Describe archived Beijing verification traces from window collector 3100357."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median


ARMS = ('suffix_mae', 'prefix_mae', 'verification_margin')
LOCKED, FIXED, MIXTURE = 'locked_two_sided', 'fixed_budget_mean', 'one_sided_mixture'
RULES = (LOCKED, FIXED, MIXTURE)
PREFIXES = (12, 18, 24, 36)


def read(path):
    return json.loads(path.read_bytes())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def inspect(arms, start, outcome, eta, delta, minimum):
    """Positive gain is better; positive boundary gap means evidence is short."""
    locked, fixed, mixture = (arms[k]['trace'] for k in RULES)
    check(all(len(trace) == 36 for trace in (locked, fixed, mixture)), 'Expected 36 archived verification rows')
    check(not arms[LOCKED]['accepted'] and not arms[MIXTURE]['accepted'], 'Unexpected accepted sequential trace')
    favorable, upper_gaps, log_gaps = [], [], []
    total = 0.
    for n, (lrow, frow, mrow) in enumerate(zip(locked, fixed, mixture), 1):
        for field in ('difference', 'bound', 'origin'):
            check(lrow[field] == frow[field] == mrow[field], 'Rule inputs differ at row %d' % n)
        check(lrow['origin'] == start+n-1, 'Nonchronological verification origin')
        diff, bound = lrow['difference'], lrow['bound']
        check(math.isfinite(diff) and math.isfinite(bound) and abs(diff) <= bound+1e-12, 'Invalid difference or bound')
        total += diff
        gain = -total/n
        check(math.isclose(gain-eta, frow['statistic'], abs_tol=1e-10), 'Empirical gain differs from fixed-rule trace')
        upper, lower = lrow['upper'], lrow['lower']
        check(lower <= upper and math.isclose((lower+upper)/2, -gain, abs_tol=1e-10), 'Interval differs from mean')
        gaps = (upper+eta, -math.log(delta)-mrow['statistic'])
        check(all(math.isfinite(x) for x in gaps), 'Nonfinite boundary gap')
        if n >= minimum:
            check(min(gaps) >= 0, 'Archived sequential decision disagrees with boundary')
        favorable.append(frow['statistic'] > 0)
        upper_gaps.append(gaps[0])
        log_gaps.append(gaps[1])
    check((frow['statistic'] > 0) == arms[FIXED]['accepted'], 'Unexpected fixed-budget decision')
    passed = [n for n in range(minimum, 37) if favorable[n-1]]
    nearest = min(range(minimum, 37), key=lambda n: upper_gaps[n-1])
    nearest_log = min(range(minimum, 37), key=lambda n: log_gaps[n-1])
    return dict(test_gain=outcome, first=min(passed, default=None), favorable=favorable,
                nearest=nearest, gap=upper_gaps[nearest-1], log_n=nearest_log, log_gap=log_gaps[nearest_log-1])


def analyze(project):
    report = project/'results/slurm_jobs/3100357/report'
    summary = read(report/'transfer_result.json')
    ident = summary['identity']
    check(len(ident) == 64 and all(c in '0123456789abcdef' for c in ident), 'Invalid campaign identity')
    check(summary['scientific_status'] == 'ECTT_WINDOW_CAMPAIGN_COMPLETE' and
          summary['diagnostic_cases_checked'] == 216 and summary['missing_tasks'] == [], 'Incomplete collector')
    root = project/'outputs/ectt/verification_windows'/ident
    preflight = read(root/'preflight.json')
    check(preflight['identity'] == ident, 'Preflight identity differs')
    protocol = read(Path(__file__).resolve().parents[1]/'config/ectt_protocol_v1.json')
    eta = protocol['training_design']['mechanism_coefficients']['epsilon']
    delta = protocol['verifier']['delta_per_stream']
    minimum = protocol['partitions']['minimum_verification_origins']
    with (report/'window_metrics.csv').open(newline='') as stream:
        records = list(csv.DictReader(stream))
    def key(r):
        return (r['dataset'], int(r['window']), int(r['seed']), r['objective'], r['rule'])
    scores = {key(row): row for row in records}
    expected = set(itertools.product(('pems08', 'bj'), range(1, 5), (11, 22, 33), ARMS, RULES))
    check(len(records) == 216 and set(scores) == expected, 'Unmatched collector CSV')
    output = {}
    for task in preflight['tasks']:
        if task['dataset'] != 'bj':
            continue
        case = root/'cases'/str(task['index'])
        marker = read(case/'COMPLETE.json')
        blob = (case/'decisions.json').read_bytes()
        check(marker['identity'] == ident and marker['artifacts']['decisions.json'] ==
              hashlib.sha256(blob).hexdigest(), 'Decision checksum mismatch at '+str(case))
        record = json.loads(blob)
        check(record['identity'] == ident and record['task'] == task, 'Decision record mismatch')
        for arm in ARMS:
            ident_key = (task['window'], task['seed'], arm)
            check(ident_key not in output, 'Duplicate test case')
            row = scores[('bj', *ident_key, FIXED)]
            arms = record['decisions'][arm]
            check(str(arms[FIXED]['accepted']) == row['accepted'], 'Collector decision mismatch')
            outcome = float(row['local_mae'])-float(row['always_transfer_mae'])
            output[ident_key] = inspect(arms, task['verification'][0], outcome, eta, delta, minimum)
    check(set(output) == set(itertools.product(range(1, 5), (11, 22, 33), ARMS)), 'Unmatched Beijing cases')
    return output, eta


def report_results(cases, eta, details):
    print('Beijing | 4 development windows | 36 candidate traces | 36 verification rows per case')
    print('Empirical verification gain = (local MAE - transfer MAE) / support scale.')
    print('For each prefix: test-improving / other candidates whose verification gain exceeds', eta)
    print('Window  test-improving/total   row12    row18    row24    row36')
    for w in range(1, 5):
        group = [r for (window, _, _), r in cases.items() if window == w]
        counts = []
        for n in PREFIXES:
            hit = [r for r in group if r['favorable'][n-1]]
            wins = sum(r['test_gain'] > 0 for r in hit)
            counts.append(f'{wins}/{len(hit)-wins}')
        print(f"{w:>6}  {sum(r['test_gain'] > 0 for r in group):>12}/{len(group):<3}", *(x.rjust(8) for x in counts))
    print('By objective among candidates that improved test MAE')
    for arm in ARMS:
        winners = [r for (_, _, a), r in cases.items() if a == arm and r['test_gain'] > 0]
        halves = [sum(r['first'] is not None and lo <= r['first'] <= hi for r in winners)
                  for lo, hi in ((12, 18), (19, 36))]
        print(f"{arm}: useful={len(winners)}, first favorable at 12-18={halves[0]}, "
              f"first at 19-36={halves[1]}, never={len(winners)-sum(halves)}")
    if details:
        print('window seed objective test_MAE_gain first_favorable closest_two_sided_row upper_gap closest_one_sided_row log_gap')
        for (window, seed, arm), r in sorted(cases.items()):
            print(window, seed, arm, f"{r['test_gain']:+.3f}", r['first'] or '-', r['nearest'],
                  f"{r['gap']:.5f}", r['log_n'], f"{r['log_gap']:.5f}")
        print('Two-sided gap = upper limit + margin in normalized MAE units.')
        print('One-sided gap = required log evidence minus observed log evidence. Each minimum is over rows 12-36.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    report_results(*analyze(args.project), args.details)
