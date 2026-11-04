#!/usr/bin/env python3
"""One Slurm job: existing development-artifact review and Qwen recovery."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from ectt_common import read, write, digest, inside, completed, finish
import compact_grid as grid

STAGE = 'shared-transfer-check'


def config(code):
    recipe = read(code/'config/pretransfer_checks_v1.json')
    contract = read(code/'config/ectt_protocol_v1.json')
    if grid.identity(code) != recipe['grid_identity'] or recipe['development_targets'] != contract['development_targets']:
        raise ValueError('Completed baseline computation/development panel differs')
    q = recipe['qwen_recovery']
    if q != {'dataset': 'pems08', 'budget': '0.05', 'seed': 11, 'support_origins': 2, 'updates': 6, 'interrupt_after': 3}:
        raise ValueError('Unexpected Qwen engineering recovery recipe')
    return recipe


def identity(code):
    return hashlib.sha256(json.dumps({'grid': grid.identity(code), 'recipe': digest(code/'config/pretransfer_checks_v1.json'),
        'worker': digest(code/'scripts/pretransfer_checks.py'), 'recovery': digest(code/'scripts/qwen_recovery.py'),
        'qwen_fit': digest(code/'scripts/ectt_pilot.py'), 'qwen_model': digest(code/'scripts/ectt_model.py'),
        'environment': digest(code/'scripts/ectt_env.py')}, sort_keys=True).encode()).hexdigest()


def foundation(project, code, full=False):
    recipe = config(code); evidence = grid.foundation(project, code, full=full)
    collector = inside(project, project/'results/slurm_jobs'/recipe['completed_collector'])
    status = read(collector/'job_status.json'); progress = read(collector/'report/compact_progress.json')
    if status.get('stage') != grid.REPORT_STAGE or status.get('execution_state') != 'COMPLETED' or status.get('shell_exit_code') != 0 or status.get('scientific_status') != 'COMPACT_LOCAL_GRID_COMPLETE':
        raise ValueError('Required compact grid collector is incomplete/failed')
    if progress.get('completed_cases') != 96 or progress.get('failed_cases') != 0:
        raise ValueError('Required compact grid has missing/failed cases')
    parent = inside(project, project/'results/slurm_jobs'/recipe['completed_array'])
    submission = read(parent/'submission.json')
    if submission.get('stage') != grid.STAGE or submission.get('grid_identity') != recipe['grid_identity'] or submission.get('array_tasks') != grid.plan(code)['tasks']:
        raise ValueError('Completed compact source array identity differs')
    return {'grid': evidence, 'array_job': recipe['completed_array'], 'collector_job': recipe['completed_collector'],
            'full_artifact_verification': full}


def _curve(path, expected_updates, window):
    info = read(path/'fit.json')
    if not completed(path, info['identity']):
        raise ValueError('Missing completed development fit')
    losses = info['losses']
    if info.get('optimizer_steps') != expected_updates or len(losses) != expected_updates or not info.get('completed'):
        raise ValueError('Incomplete development training trajectory')
    if not all(math.isfinite(x) and x >= 0 for x in losses):
        raise ValueError('Nonfinite/negative recorded training loss')
    first = math.fsum(losses[:window])/window; last = math.fsum(losses[-window:])/window
    return {'first_window_mean_MSE': first, 'last_window_mean_MSE': last, 'minimum_sampled_MSE': min(losses),
            'relative_change_last_vs_first': (last-first)/first if first else None,
            'optimizer_steps': expected_updates, 'parameters': info['parameters'],
            'elapsed_training_seconds': info['elapsed_training_seconds'], 'losses': losses}


def review_development(project, code, store, ident, progress):
    if completed(store, ident):
        return read(store/'result.json')
    recipe = config(code); numerical = grid.config(code)[1]
    contract = read(code/'config/ectt_protocol_v1.json'); evidence = grid.foundation(project, code, full=True)
    records = [r for r in grid.plan(code)['records'] if r['dataset'] in recipe['development_targets']]
    if len(records) != recipe['expected_development_records']:
        raise ValueError('Development record panel differs')
    store.mkdir(parents=True, exist_ok=True); rows = []; curves = []; fallback = []
    for task in records:
        if task['kind'] == 'DETERMINISTIC_FALLBACK':
            fallback.append(dict(task, reviewed_as='NO_SUPERVISED_FIT')); continue
        if task['kind'] == 'PILOT_REUSE':
            outer = Path(evidence['pilot_store'])/'cases'/task['key']; expected = evidence['pilot_identity']; source = outer
        else:
            outer = grid.case_path(project, code, task); expected = recipe['grid_identity']; source = outer/'kernel'
        if not completed(outer, expected):
            raise ValueError('Missing/corrupt completed development case: '+task['key'])
        selection = read(source/'selection.json'); validation = read(source/'compact_validation.json')
        access = read(source/'access_ledger.json')
        profile = contract['datasets'][task['dataset']]; bounds = profile['budgets'][task['budget']]
        start, end = bounds['support_raw']; split = start+int(.7*(end-start))
        if (selection['inner_split'] != split or selection['inner_scaler']['fit_rows'] != [start, split]
                or selection['final_scaler']['fit_rows'] != [start, end]):
            raise ValueError('Development support-scaler/selection timing differs')
        if (access['verification_labels_used_for_fitting_or_selection'] != 0
                or access['test_labels_used_for_fitting_or_selection'] != 0
                or access['unique_supervised_values'] != (end-start)*profile['shape'][0]):
            raise ValueError('Development access ledger differs')
        if [v['index'] for v in validation] != list(range(len(numerical['configurations']))):
            raise ValueError('Incomplete development configuration review')
        best = min(validation, key=lambda v: v['validation_MAE'])
        if best['index'] != selection['best_compact_configuration']:
            raise ValueError('Compact support-only configuration selection differs')
        if [r['family'] for r in selection['candidates']] != numerical['selection_order']:
            raise ValueError('Development candidate ordering differs')
        selected = min(selection['candidates'], key=lambda r: r['validation_MAE'])
        if selected['family'] != selection['selected_family'] or selection['support_scale'] != max(selected['validation_MAE'], 1.):
            raise ValueError('Local support-only family/scale selection differs')
        details = []
        for v in validation:
            if v['architecture'] != numerical['configurations'][v['index']] or not math.isfinite(v['validation_MAE']):
                raise ValueError('Changed/invalid development architecture or validation score')
            details.append(('inner_'+str(v['index']), source/'inner'/str(v['index'])))
        details.append(('refit', source/'refit'))
        fit_rows = []
        for name, path in details:
            curve = _curve(path, numerical['updates'], recipe['loss_summary_window'])
            for step, loss in enumerate(curve.pop('losses'), 1):
                curves.append({'dataset': task['dataset'], 'budget': task['budget'], 'seed': task['seed'],
                               'fit': name, 'step': step, 'normalized_sampled_batch_MSE': loss})
            fit_rows.append({'fit': name, **curve})
        values = {r['family']: r['validation_MAE'] for r in selection['candidates']}
        rows.append({**task, 'selected_family': selected['family'], 'best_compact_index': best['index'],
            'best_compact_architecture': best['architecture'], 'support_validation_MAE': values,
            'compact_validation_beats_persistence': values['compact'] < values['persistence'],
            'compact_validation_beats_ridge': values['compact'] < values['ridge'], 'training_curves': fit_rows,
            'source_store': str(source), 'new_optimizer_updates': 0})
        progress(phase='DEVELOPMENT_REVIEW', reviewed_trained_cases=len(rows), expected_trained_cases=recipe['expected_development_fits'])
    if len(rows) != recipe['expected_development_fits'] or len(curves) != recipe['expected_training_curves']*numerical['updates']:
        raise ValueError('Incomplete development curve inventory')
    result = {'stage': STAGE, 'reviewed_trained_cases': len(rows), 'reviewed_fallback_records': len(fallback),
        'training_curves_reviewed': recipe['expected_training_curves'], 'cases': rows, 'fallback_records': fallback,
        'new_optimizer_steps': 0, 'new_inference_performed': False, 'new_test_evaluation_performed': False,
        'architecture_or_protocol_changed': False,
        'interpretation': 'Recorded resampled training-minibatch MSE and fixed support-validation MAE; these do not identify underfitting, overfitting or a defect by themselves'}
    with (store/'training_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    write(store/'result.json', result); finish(store, ident)
    return result


def run(project, code, report):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    inside(project, code); inside(project, report)
    ident = identity(code); store = inside(project, project/'outputs/ectt/pretransfer_checks'/ident)
    store.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    state = {'stage': STAGE, 'identity': ident, 'store': str(store), 'new_optimizer_steps': 0,
             'benchmark_experiments_completed': 0, 'method_evaluated': False}
    def progress(**values):
        state.update(values, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(report/'transfer_check_progress.json', state); print('TRANSFER_CHECK_PROGRESS', json.dumps(state), flush=True)
    began = time.monotonic()
    with (store/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress(phase='VERIFYING_COMPLETED_BASELINES')
            write(report/'transfer_check_foundation.json', foundation(project, code, full=True))
            review = review_development(project, code, store/'development_review', ident, progress)
            write(report/'development_review.json', review)
            shutil.copyfile(store/'development_review/training_curves.csv', report/'development_training_curves.csv')
            from qwen_recovery import run as recover
            outcome = recover(project, code, report, store/'qwen_recovery', ident, progress)
            if not outcome.get('passed') or outcome.get('engineering_optimizer_steps') != 12:
                raise ValueError('Qwen recovery gate did not pass the declared 12-update engineering check')
            write(report/'qwen_recovery_result.json', outcome)
            result = {'stage': STAGE, 'identity': ident, 'scientific_status': 'PRETRANSFER_ENGINEERING_CHECKS_PASSED',
                'development_trained_cases_reviewed': review['reviewed_trained_cases'],
                'training_curves_reviewed': review['training_curves_reviewed'], 'qwen_recovery_passed': outcome['passed'],
                'qwen_full_size_fixed_support_optimizer_resume_verified': True,
                'engineering_optimizer_steps': outcome['engineering_optimizer_steps'],
                'new_optimizer_steps': outcome['new_optimizer_steps'], 'training_performed': outcome['new_optimizer_steps'] > 0,
                'benchmark_experiments_completed': 0, 'method_evaluated': False, 'S01_independent_review': 'PENDING',
                'elapsed_seconds': time.monotonic()-began,
                'scope': 'Development review and full-size fixed-support recovery only; B03/B04 benchmark trainers and their episode/adaptation recovery remain pending'}
            write(report/'transfer_check_result.json', result); write(store/'result.json', result)
            lines = ['# Pre-transfer development and recovery checks', '',
                'Reviewed 15 completed development cases and 75 recorded training trajectories; no refits.', '',
                '| Dataset | Budget | Seed | Selected family | Compact support MAE | Ridge support MAE | Persistence support MAE |',
                '|---|---|---:|---|---:|---:|---:|']
            for r in review['cases']:
                v = r['support_validation_MAE']
                lines.append('| %s | %s | %s | %s | %.6f | %.6f | %.6f |' %
                    (r['dataset'], r['budget'], r['seed'], r['selected_family'], v['compact'], v['ridge'], v['persistence']))
            lines += ['', 'Full-size Qwen fixed-support process-restart check passed: model, Adam, CPU/CUDA RNG, loss trajectory, cursor and support predictions match bitwise.',
                'Controlled interruption after update 3; both paths finish update 6. Twelve total engineering updates on a fresh run.',
                'No new test evaluation, compact retuning, benchmark source training or ECTT objective.',
                'Future episodic/adaptation benchmark trainers require their own cursor recovery tests. Independent S01 remains pending.',
                'Loss summaries describe resampled training minibatches; they do not prove a cause of weak test performance.',
                'Existing baseline, pilot and cache artifacts are preserved.', 'Store: '+str(store)]
            (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n'); shutil.copyfile(report/'SUMMARY.md', store/'SUMMARY.md')
            progress(phase='COMPLETED', new_optimizer_steps=outcome['new_optimizer_steps'])
            print((report/'SUMMARY.md').read_text(), flush=True)
            return 0
        except Exception as exc:
            segment_path = report/'qwen_segment_progress.json'
            if segment_path.exists():
                state['new_optimizer_steps'] = max(state['new_optimizer_steps'], read(segment_path).get('cumulative_new_optimizer_steps', 0))
            progress(phase='FAILED', error=str(exc))
            (report/'failure_traceback.txt').write_text(traceback.format_exc())
            (report/'SUMMARY.md').write_text('# Pre-transfer check failed\n\n'+str(exc)+'\n\nPartial artifacts: '+str(store)+'\n')
            return 2


def dispatch(project, code, report):
    foundation(project, code, full=False)
    from ectt_env import ensure
    with (project/'work/ectt_environment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/pretransfer_checks.py'), '--mode', 'run',
        '--project', str(project), '--code', str(code), '--report', str(report)],
        env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--code', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True); p.add_argument('--mode', choices=['run', 'qwen-segment'], required=True)
    p.add_argument('--store', type=Path); p.add_argument('--identity'); p.add_argument('--segment', choices=['full', 'interrupt', 'resume'])
    p.add_argument('--manifest-sha'); args = p.parse_args()
    inside(args.project, args.code); inside(args.project, args.report)
    if not os.environ.get('SLURM_JOB_ID', '').isdigit() or not (args.code/'.source_snapshot.json').is_file():
        p.error('Use a real Slurm allocation and a verified source snapshot')
    if args.mode == 'run':
        return run(args.project, args.code, args.report)
    if not all((args.store, args.identity, args.segment, args.manifest_sha)):
        p.error('Qwen child requires contained store, identity, segment and verified manifest')
    from qwen_recovery import worker
    return worker(args.project.resolve(), args.code.resolve(), args.report.resolve(), inside(args.project, args.store),
                  args.identity, args.segment, args.manifest_sha)


if __name__ == '__main__':
    sys.exit(main())
