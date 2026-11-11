"""Describe frozen development forecast errors without replaying verification."""
import argparse
import csv
import fcntl
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

from ectt_common import inside, load_contract, read, write
import verification_pilot as pilot


STAGE = 'shared-transfer-ectt-candidate-audit'


def admission(project, code):
    plan = pilot.foundation(project, code)
    jobs = pilot.jobs(project, plan)
    report = pilot.job_report(project, jobs['report'], pilot.REPORT)
    result = read(report/'transfer_result.json')
    if (result.get('identity') != plan['identity'] or result.get('failed')
            or result.get('scientific_status') != 'ECTT_PILOT_CANDIDATES_READY'
            or result.get('diagnostic_cases_checked') != len(plan['tasks'])):
        raise ValueError('Complete frozen pilot candidate collection is required')
    return plan


def run(project, code, report):
    import numpy as np
    from shared_baselines import independent_scores
    from transfer_worker import raw_arrays, raw_paths
    from verification_pilot_score import frozen_cases

    plan = admission(project, code)
    contract, _ = load_contract(code)
    cases = frozen_cases(project, plan, contract)
    raw = raw_arrays(raw_paths(project, code, contract), plan['recipe']['development_targets'], contract)
    truths = {}
    rows = []
    for case in cases:
        task = case['task']
        domain = task['dataset']
        bounds = contract['datasets'][domain]['test_targets']
        if domain not in truths:
            truths[domain] = np.asarray(raw[domain][:, slice(*bounds)].T, dtype=np.float64)
        truth = truths[domain]
        row = {key: task[key] for key in ('family', 'dataset', 'seed', 'arm')}
        for name in ('local', 'transfer'):
            prediction = case['forecasts'][name]
            if truth.shape != prediction.shape or not np.isfinite(truth).all() or np.any(truth < 0):
                raise ValueError('Invalid frozen development test truth')
            for metric, value in independent_scores(prediction, truth).items():
                row[f'{name}_{metric}'] = value
        row.update(test_origins=len(truth), frozen_case_sha256=case['completion_sha256'])
        rows.append(row)
    with (report/'candidate_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_case = {(row['dataset'], row['seed'], row['arm']): row for row in rows}
    arms = plan['recipe']['arms']
    seeds = plan['recipe']['seeds']
    lines = ['# ECTT frozen candidate accuracy audit', '',
             f'Scored {len(rows)}/{len(plan["tasks"])} development test forecast pairs.',
             'These MAEs describe unconditional forecast accuracy. S01 review is pending.', '',
             '| Domain | Local MAE | Suffix MAE | Prefix MAE | Margin MAE | Margin wins vs prefix | Margin wins vs local |',
             '|---|---:|---:|---:|---:|---:|---:|']
    comparison = {}
    for domain in plan['recipe']['development_targets']:
        means = {arm: mean(by_case[domain, seed, arm]['transfer_MAE'] for seed in seeds) for arm in arms}
        local = mean(by_case[domain, seed, arms[0]]['local_MAE'] for seed in seeds)
        deltas = {str(seed): {
            'vs_suffix_MAE': by_case[domain, seed, arms[-1]]['transfer_MAE']-by_case[domain, seed, arms[0]]['transfer_MAE'],
            'vs_prefix_MAE': by_case[domain, seed, arms[-1]]['transfer_MAE']-by_case[domain, seed, arms[1]]['transfer_MAE'],
            'vs_local_MAE': by_case[domain, seed, arms[-1]]['transfer_MAE']-by_case[domain, seed, arms[-1]]['local_MAE'],
        } for seed in seeds}
        prefix_wins = sum(delta['vs_prefix_MAE'] < 0 for delta in deltas.values())
        local_wins = sum(delta['vs_local_MAE'] < 0 for delta in deltas.values())
        comparison[domain] = {'local_MAE': local, 'transfer_MAE': means, 'margin_paired_differences': deltas}
        lines.append(f'| {domain} | {local:.3f} | {means[arms[0]]:.3f} | {means[arms[1]]:.3f} | '
                     f'{means[arms[2]]:.3f} | {prefix_wins}/{len(seeds)} | {local_wins}/{len(seeds)} |')
    lines += ['', 'Per-seed MAE, RMSE and WMAPE are in candidate_metrics.csv.',
              'The formal verifier comparison follows independent S01 review.', '']
    (report/'SUMMARY.md').write_text('\n'.join(lines))
    write(report/'transfer_result.json', {
        'scientific_status': 'ECTT_DEVELOPMENT_CANDIDATE_AUDIT_COMPLETE',
        'identity': plan['identity'], 'diagnostic_cases_checked': len(rows),
        'method_evaluated': False, 'S01_independent_review': 'PENDING',
        'verification_labels_scored': 0, 'new_optimizer_steps': 0,
        'comparison': comparison})
    return 0


def dispatch(project, code, report):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/verification_pilot_candidate_audit.py'),
                            str(project), str(code), str(report)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    report = inside(project, args.report)
    try:
        sys.exit(run(project, inside(project, args.code), report))
    except Exception as exc:
        traceback.print_exc()
        (report/'SUMMARY.md').write_text(f'# Candidate audit failed\n\n{exc}\n')
        write(report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
