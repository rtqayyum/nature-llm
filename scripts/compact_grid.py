#!/usr/bin/env python3
"""Independent compact cases plus an artifact-only, after-any collector."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from ectt_common import read, write, digest, inside, completed, finish
import compact_baselines as pilot
import shared_baselines as base

STAGE = 'shared-compact-baselines'
REPORT_STAGE = 'shared-compact-report'


def config(code):
    recipe = read(code/'config/compact_grid_v1.json')
    p = pilot.config(code)
    if digest(code/'config/compact_local_v1.json') != recipe['pilot_recipe_sha256'] or pilot.identity(code) != recipe['pilot_identity']:
        raise ValueError('Admitted compact computation changed; do not reuse pilot evidence')
    if recipe['budgets'] != ['0.0', '0.01', '0.05', '0.1'] or recipe['seeds'] != [11, 22, 33]:
        raise ValueError('Changed full compact budget/seed matrix')
    if recipe['max_concurrent_tasks'] != 4 or recipe['dependency'] != 'afterany':
        raise ValueError('Unexpected compact execution policy')
    return recipe, p


def identity(code):
    return hashlib.sha256(json.dumps({'pilot': pilot.identity(code),
        'grid_recipe': digest(code/'config/compact_grid_v1.json'),
        'grid_worker': digest(code/'scripts/compact_grid.py')}, sort_keys=True).encode()).hexdigest()


def plan(code):
    recipe, p = config(code); contract = read(code/'config/ectt_protocol_v1.json')
    records, tasks = [], []
    for slug, profile in contract['datasets'].items():
        for budget in recipe['budgets']:
            for seed in recipe['seeds']:
                fallback = not profile['budgets'][budget]['eligible_for_learned_adaptation'] or float(budget) == 0
                reused = not fallback and slug in p['datasets'] and budget == p['budget'] and seed == p['seed']
                row = {'dataset': slug, 'budget': budget, 'seed': seed, 'key': '%s/%s/%s' % (slug, budget, seed),
                       'kind': 'DETERMINISTIC_FALLBACK' if fallback else 'PILOT_REUSE' if reused else 'GPU_FIT'}
                if row['kind'] == 'GPU_FIT':
                    row['task_index'] = len(tasks); tasks.append(dict(row))
                records.append(row)
    counts = {kind: sum(r['kind'] == kind for r in records) for kind in ('GPU_FIT', 'PILOT_REUSE', 'DETERMINISTIC_FALLBACK')}
    if (len(records), counts['GPU_FIT'], counts['PILOT_REUSE'], counts['DETERMINISTIC_FALLBACK']) != (
        recipe['expected_records'], recipe['expected_new_gpu_cases'], recipe['expected_pilot_reuse_cases'],
        recipe['expected_deterministic_fallback_records']):
        raise ValueError('Unexpected compact run matrix; preserve ineligible and reused cases')
    return {'schema': recipe['schema'], 'identity': identity(code), 'records': records, 'tasks': tasks,
            'counts': counts, 'recipe_sha256': digest(code/'config/compact_grid_v1.json')}


def foundation(project, code, full=False):
    recipe, p = config(code)
    evidence = pilot.foundation(project, code, full=full)
    job = inside(project, project/'results/slurm_jobs'/recipe['pilot_job'])
    status = read(job/'job_status.json'); result = read(job/'report/compact_result.json')
    if status.get('stage') != pilot.STAGE or status.get('execution_state') != 'COMPLETED' or status.get('shell_exit_code') != 0:
        raise ValueError('Compact development pilot has not completed successfully')
    if result.get('identity') != recipe['pilot_identity'] or result.get('completed_cases') != 2:
        raise ValueError('Compact pilot identity/case count differs')
    runtime = read(job/'report/compact_runtime.json')
    if runtime.get('allocated_cpus') != 5 or runtime.get('visible_gpus') != 1 or runtime.get('torch') != '2.4.1+cu118':
        raise ValueError('Compact pilot CUDA allocation/environment differs')
    store = inside(project, project/'outputs/ectt/compact_local'/recipe['pilot_identity'])
    for slug in p['datasets']:
        row = result['cases'][slug]
        if not row.get('recovery_probe_passed') or row.get('posttraining_invariance_passed') != 4:
            raise ValueError('Required compact recovery/invariance gate failed')
        case = store/'cases'/slug/p['budget']/str(p['seed'])
        if full and (not completed(case, recipe['pilot_identity']) or read(case/'result.json') != row):
            raise ValueError('Missing or inconsistent compact pilot case: '+slug)
    return {'baseline': evidence, 'pilot_store': str(store), 'pilot_identity': recipe['pilot_identity'],
            'pilot_job': recipe['pilot_job'], 'full_artifact_verification': full}


def store_path(project, code):
    return inside(project, project/'outputs/ectt/compact_grid'/identity(code))


def case_path(project, code, task):
    root = store_path(project, code)
    return inside(root, root/'cases'/task['key'])


def run_task(project, code, report, index, device='cuda'):
    import numpy as np
    import torch
    from compact_training import configure
    from ectt_data import verify_profile
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    inside(project, code); inside(project, report)
    matrix = plan(code)
    if not 0 <= index < len(matrix['tasks']):
        raise ValueError('Unknown compact array task index')
    task = matrix['tasks'][index]; case = case_path(project, code, task)
    case.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    state = {'stage': STAGE, 'identity': matrix['identity'], 'store': str(case), 'task': task,
             'completed_cases': 0, 'total_cases': 1, 'new_optimizer_steps': 0, 'training_performed': False}
    new_by_fit = {}; began = time.monotonic()
    def progress(**values):
        key = values.get('fit_key')
        if key is not None and 'new_optimizer_steps' in values:
            new_by_fit[key] = values.pop('new_optimizer_steps')
        state.update(values, new_optimizer_steps=sum(new_by_fit.values()), training_performed=sum(new_by_fit.values()) > 0,
                     updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(report/'compact_progress.json', state)
        print('COMPACT_GRID_PROGRESS', json.dumps(state), flush=True)
    with (case/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress(phase='VERIFYING_FOUNDATION')
            evidence = foundation(project, code, full=True)
            write(report/'compact_foundation.json', evidence)
            write(report/'compact_store.json', {'identity': matrix['identity'], 'store': str(case), 'task': task})
            if device == 'cuda':
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or os.environ.get('SLURM_CPUS_PER_TASK') != '5':
                    raise ValueError('Compact task requires one visible GPU and five CPUs')
                if torch.__version__ != '2.4.1+cu118' or np.__version__ != '1.26.4':
                    raise ValueError('Pinned compact GPU environment differs')
                torch.cuda.reset_peak_memory_stats()
            configure()
            recipe = dict(config(code)[1], budget=task['budget'], seed=task['seed'])
            contract = read(code/'config/ectt_protocol_v1.json'); profile = contract['datasets'][task['dataset']]
            verify_profile(profile)
            source = read(code/'config/sources.lock.json')['reference_commit']
            path = inside(project, project/'work/reference/snapshots'/source/profile['path'])
            if digest(path) != profile['container_sha256']:
                raise ValueError('Canonical compact data digest differs')
            raw = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(raw.shape) != profile['shape'] or raw.dtype.kind not in 'fiu':
                raise ValueError('Canonical compact data shape/type differs')
            raw = raw[..., 0]
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError('Invalid compact raw channel')
            reused = completed(case, matrix['identity'])
            if reused:
                result = read(case/'result.json')
            else:
                kernel = case/'kernel'
                original = read(kernel/'result.json') if completed(kernel, matrix['identity']) else pilot.save_case(
                    raw, profile, task['dataset'], recipe,
                    Path(evidence['baseline']['store'])/'cases'/task['dataset']/task['budget'], kernel,
                    matrix['identity'], device, progress)
                result = dict(original, development_only=task['dataset'] in contract['development_targets'],
                              forecast_file='kernel/forecasts.npz', access_ledger_file='kernel/access_ledger.json')
                write(case/'result.json', result); finish(case, matrix['identity'])
            output = {'stage': STAGE, 'identity': matrix['identity'], 'task': task, 'case_result': result,
                      'completed_cases': 1, 'benchmark_experiments_completed': 1,
                      'new_optimizer_steps': state['new_optimizer_steps'], 'training_performed': state['training_performed'],
                      'optimizer_steps': result['optimizer_steps']+result['engineering_probe_optimizer_steps'],
                      'scientific_status': 'COMPACT_LOCAL_CASE_ONLY', 'method_evaluated': False,
                      'development_only': task['dataset'] in contract['development_targets'], 'S01_independent_review': 'PENDING',
                      'reused_case': reused, 'store': str(case)}
            write(report/'compact_result.json', output)
            import resource
            metadata = read(code/'.source_snapshot.json') if (code/'.source_snapshot.json').exists() else {}
            write(report/'compact_runtime.json', {'job_id': os.environ.get('TRAFFIC_JOB_KEY', os.environ.get('SLURM_JOB_ID')),
                'code_commit': metadata.get('commit', 'SYNTHETIC_LOCAL_TEST'), 'task': task, 'torch': torch.__version__,
                'numpy': np.__version__, 'cuda': torch.version.cuda, 'allocated_cpus': 5,
                'visible_gpus': torch.cuda.device_count() if device == 'cuda' else 0,
                'gpu': torch.cuda.get_device_name() if device == 'cuda' else 'SYNTHETIC_CPU',
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                'peak_host_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'elapsed_seconds': time.monotonic()-began, 'new_optimizer_steps': state['new_optimizer_steps'],
                'scientific_contract_sha256': recipe['scientific_contract_sha256']})
            (report/'SUMMARY.md').write_text('# Compact local case completed\n\n'+json.dumps(task)+'\n\n'
                +'Support-selected family: '+result['selection']['selected_family']+'\n\nStore: '+str(case)
                +'\n\nFull grid collection is separate; S01 and ECTT remain pending.\n')
            progress(phase='COMPLETED', completed_cases=1, reused_case=reused)
            return 0
        except Exception as exc:
            progress(phase='FAILED', error=str(exc))
            (report/'failure_traceback.txt').write_text(traceback.format_exc())
            (report/'SUMMARY.md').write_text('# Compact grid case failed\n\n'+str(exc)+'\n\nStore: '+str(case)+'\n')
            return 2


def collect(project, code, report, array_id):
    """Verify every record, retaining explicit failed/missing/censored cases."""
    import csv
    import numpy as np
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    inside(project, code); inside(project, report); report.mkdir(parents=True, exist_ok=True)
    if not str(array_id).isdigit():
        raise ValueError('A numeric recorded compact source array is required')
    matrix = plan(code); root = store_path(project, code); root.mkdir(parents=True, exist_ok=True)
    evidence = foundation(project, code, full=True)
    parent = inside(project, project/'results/slurm_jobs'/array_id)
    submission = read(parent/'submission.json')
    if submission.get('stage') != STAGE or submission.get('grid_identity') != matrix['identity'] or submission.get('array_tasks') != matrix['tasks']:
        raise ValueError('Collector source array does not match this compact contract')
    results, failed, rows = {}, {}, []
    with (root/'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in matrix['records']:
            try:
                if task['kind'] == 'DETERMINISTIC_FALLBACK':
                    store = Path(evidence['baseline']['store'])/'cases'/task['dataset']/task['budget']
                    expected = evidence['baseline']['baseline_identity']
                    source = read(store/'result.json')
                    metrics = {name: source['metrics']['persistence'] for name in ('persistence', 'ridge', 'compact', 'support_selected_local')}
                    record = {'metrics': metrics, 'selection': {'selected_family': 'persistence'},
                              'independent_seed_replication': False, 'state': source['state']}
                    columns = {name: 'persistence' for name in metrics}
                else:
                    if task['kind'] == 'PILOT_REUSE':
                        store = Path(evidence['pilot_store'])/'cases'/task['key']; expected = evidence['pilot_identity']
                    else:
                        store = case_path(project, code, task); expected = matrix['identity']
                    record = read(store/'result.json'); metrics = record['metrics']; columns = {name: name for name in metrics}
                    if any(record.get(k) != task[k] for k in ('dataset', 'budget', 'seed')):
                        raise ValueError('Compact case dataset/budget/seed differs')
                    if not record.get('recovery_probe_passed') or record.get('posttraining_invariance_passed') != 4:
                        raise ValueError('Compact case admission gates are missing/failed')
                    if task['kind'] == 'GPU_FIT':
                        child = parent.parent/(array_id+'_'+str(task['task_index']))
                        state = read(child/'job_status.json')
                        if state.get('execution_state') != 'COMPLETED' or state.get('shell_exit_code') != 0:
                            raise ValueError('Array task has no successful final status: '+child.name)
                        reported = read(child/'report/compact_result.json')
                        if reported.get('case_result') != record or reported.get('identity') != expected:
                            raise ValueError('Array task report differs from completed case')
                if not completed(store, expected):
                    raise ValueError('Missing case completion marker')
                ledger_file = record.get('access_ledger_file', 'access_ledger.json')
                ledger = read(inside(store, store/ledger_file))
                if ledger.get('verification_labels_used_for_fitting_or_selection') != 0 or ledger.get('test_labels_used_for_fitting_or_selection') != 0:
                    raise ValueError('Compact access ledger does not establish support-only fitting/selection')
                # Saved truths/origin IDs must equal the fixed CPU foundation.
                prior = Path(evidence['baseline']['store'])/'cases'/task['dataset']/task['budget']
                forecast_file = record.get('forecast_file', 'forecasts.npz')
                with np.load(inside(store, store/forecast_file), allow_pickle=False) as payload, np.load(prior/'forecasts.npz', allow_pickle=False) as truth:
                    if not np.array_equal(payload['origins'], truth['origins']) or not np.array_equal(payload['truth'], truth['truth']):
                        raise ValueError('Compact forecast cohort/truth differs from foundation')
                    for name, column in columns.items():
                        check = base.independent_scores(payload[column], payload['truth'])
                        for metric, value in check.items():
                            old = metrics[name][metric]
                            if (value is None) != (old is None) or (value is not None and not np.isclose(value, old, rtol=1e-11, atol=1e-11)):
                                raise ValueError('Independent collection metric differs')
                results[task['key']] = {**task, 'state': 'VERIFIED', 'artifact_store': str(store),
                    'artifact_identity': expected, 'forecast_columns': columns, 'metrics': metrics,
                    'forecast_file': forecast_file, 'access_ledger_file': ledger_file,
                    'selected_family': record['selection']['selected_family'],
                    'independent_seed_replication': task['kind'] != 'DETERMINISTIC_FALLBACK'}
                for method, metrics_row in metrics.items():
                    rows.append({**{k: task[k] for k in ('dataset', 'budget', 'seed', 'key', 'kind')},
                        'task_index': task.get('task_index'), 'method': method,
                        'selected_family': record['selection']['selected_family'], **metrics_row})
            except Exception as exc:
                failed[task['key']] = str(exc)
                results[task['key']] = {**task, 'state': 'FAILED_OR_MISSING', 'error': str(exc)}
            write(report/'compact_progress.json', {'stage': REPORT_STAGE, 'phase': 'COLLECTING',
                'completed_cases': len(results)-len(failed), 'failed_cases': len(failed), 'total_cases': len(matrix['records'])})
        output = {'stage': REPORT_STAGE, 'identity': matrix['identity'], 'source_array': array_id,
                  'expected_cases': len(matrix['records']), 'completed_cases': len(results)-len(failed), 'failed_cases': failed,
                  'records': results, 'scientific_status': 'COMPACT_LOCAL_GRID_COMPLETE' if not failed else 'COMPACT_LOCAL_GRID_INCOMPLETE',
                  'training_performed': False, 'method_evaluated': False, 'S01_independent_review': 'PENDING',
                  'pilot_cases_reused': 2, 'deterministic_fallback_records': 30,
                  'fallback_seed_references_are_independent_replications': False}
        write(report/'compact_grid_result.json', output)
        write(parent/'report/compact_grid_result.json', output)
        with (report/'compact_grid_results.csv').open('w', newline='') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        lines = ['# Compact local baseline collection', '', 'Verified %s/%s records; failed/missing %s.' %
                 (output['completed_cases'], output['expected_cases'], len(failed)), '',
                 '64 remaining GPU cases; two verified pilot cases reused; 30 deterministic fallback seed references.',
                 'No automatic scientific interpretation. Full saved records and per-method metrics: compact_grid_result.json / compact_grid_results.csv.',
                 'Failed/infeasible records retained. Zero/infeasible cases use persistence without added labels.',
                 'B03/B04, ECTT and independent S01 remain pending. Ordered rows; calendar/provenance limits unchanged.',
                 '', 'Source array: '+array_id, 'Store: '+str(root)]
        lines += [key+': '+value for key, value in failed.items()]
        (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
        write(report/'compact_progress.json', {'stage': REPORT_STAGE, 'phase': 'FAILED' if failed else 'COMPLETED',
            'completed_cases': output['completed_cases'], 'failed_cases': len(failed), 'total_cases': len(matrix['records'])})
        return 2 if failed else 0


def dispatch(project, code, report, stage):
    if stage == STAGE:
        from ectt_env import ensure
        with (project/'work/ectt_environment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            python = ensure(project, code, report)
    else:
        python = inside(project, project/'.venv/bin/python')
        if not python.exists():
            raise ValueError('Project NumPy environment missing for CPU collection')
    return subprocess.call([str(python), '-u', str(code/'scripts/compact_grid.py'), '--project', str(project),
        '--code', str(code), '--report', str(report), '--stage', stage],
        env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--stage', choices=[STAGE, REPORT_STAGE], required=True); args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID', '').isdigit() or not (args.code/'.source_snapshot.json').is_file():
        p.error('Use a real Slurm allocation and verified code snapshot')
    try:
        if args.stage == STAGE:
            return run_task(args.project, args.code, args.report, int(os.environ['SLURM_ARRAY_TASK_ID']))
        return collect(args.project, args.code, args.report, os.environ['TRAFFIC_SOURCE_ARRAY'])
    except Exception as exc:
        report = inside(args.project, args.report); report.mkdir(parents=True, exist_ok=True)
        write(report/'compact_progress.json', {'stage': args.stage, 'phase': 'FAILED', 'error': str(exc)})
        (report/'failure_traceback.txt').write_text(traceback.format_exc())
        (report/'SUMMARY.md').write_text('# Compact grid failed\n\n'+str(exc)+'\n')
        return 2


if __name__ == '__main__':
    sys.exit(main())
