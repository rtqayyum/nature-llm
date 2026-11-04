#!/usr/bin/env python3
"""Resumable persistence/ridge foundation on the frozen public-eight contract."""
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
import sys
import time
import traceback
from ectt_common import read, write, digest, inside, completed, finish

STAGE = 'shared-local-baselines'
DEPENDENCIES = ('config/shared_baselines_v1.json', 'config/ectt_protocol_v1.json',
                'config/sources.lock.json', 'scripts/shared_baselines.py',
                'scripts/statistical_audit.py', 'scripts/ectt_common.py',
                'scripts/ectt_data.py', 'scripts/ectt_statistics.py')
PILOT_DEPENDENCIES = ('config/ectt_protocol_v1.json', 'config/ectt_execution_v1.json',
                      'config/ectt_gpu_requirements.txt', 'config/sources.lock.json',
                      'scripts/ectt_model.py', 'scripts/ectt_pilot.py',
                      'scripts/ectt_data.py', 'scripts/ectt_common.py')


def config(code):
    recipe = read(code/'config/shared_baselines_v1.json')
    if digest(code/'config/ectt_protocol_v1.json') != recipe['scientific_contract_sha256']:
        raise ValueError('Frozen science differs; amend before affected baseline runs')
    if recipe['seeds'] != [11, 22, 33] or recipe['budgets'] != ['0.0', '0.01', '0.05', '0.1']:
        raise ValueError('Unexpected seed/budget recipe')
    if recipe['selection_support_fraction'] != .7 or not 1 <= len(recipe['ridge_alphas']) <= 12 or not all(math.isfinite(a) and a > 0 for a in recipe['ridge_alphas']):
        raise ValueError('Invalid predeclared support selection/ridge search recipe')
    return recipe


def identity(code):
    return hashlib.sha256(json.dumps({p: digest(code/p) for p in DEPENDENCIES}, sort_keys=True).encode()).hexdigest()


def foundation(project, code, full=False):
    recipe = config(code); f = recipe['foundation']
    source = inside(project, project/'work/code_versions'/f['pilot_commit'])
    marker = read(source/'.source_snapshot.json')
    if marker['repository'] != 'rtqayyum/nature-llm' or marker['commit'] != f['pilot_commit']:
        raise ValueError('Pinned pilot source snapshot mismatch')
    inventory = {row['path']: row for row in marker['files']}
    hashes = {}
    for name in PILOT_DEPENDENCIES:
        data = (source/name).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if name not in inventory or inventory[name]['sha'] != blob or digest(code/name) != digest(source/name):
            raise ValueError('Changed pilot/model/scientific dependency: '+name)
        hashes[name] = digest(code/name)
    job = inside(project, project/'results/slurm_jobs'/f['pilot_job'])
    status = read(job/'job_status.json')
    if status.get('stage') != 'ectt-gpu-pilot' or status.get('execution_state') != 'COMPLETED' or status.get('shell_exit_code') != 0:
        raise ValueError('Required GPU pilot has not completed successfully')
    stores = {'ectt-preflight': (f['data_identity'], 'ectt-preflight'),
              'ectt-gpu-pilot': (f['pilot_identity'], 'ectt-gpu-pilot')}
    records = {}
    for label, (ident, stage) in stores.items():
        store = inside(project, project/'outputs/ectt/admission'/ident/stage)
        if not (store/'COMPLETE.json').is_file() or read(store/'COMPLETE.json')['identity'] != ident:
            raise ValueError('Required admission completion is missing/mismatched: '+label)
        if full and not completed(store, ident):
            raise ValueError('Admission artifacts could not be verified: '+label)
        records[label] = {'store': str(store), 'identity': ident}
    pilot = read(Path(records['ectt-gpu-pilot']['store'])/'result.json')
    if pilot.get('stage') != 'ectt-gpu-pilot' or pilot.get('execution_kind') != 'REAL_GPU_SUPPORT_SMOKE' or pilot.get('allocated_cpus') != 5 or pilot.get('visible_gpus') != 1:
        raise ValueError('GPU pilot execution scope/allocation differs')
    if pilot.get('test_evaluation_performed') is not False or pilot.get('new_method_objective_evaluated') is not False:
        raise ValueError('Unexpected evaluation in the foundation GPU pilot')
    expected_cases = {'single_origin', 'origin_permutation', 'later_companion_mutation', 'node_permutation'}
    for key in ('pretraining', 'posttraining'):
        rows = pilot.get(key, [])
        if {row['case'] for row in rows} != expected_cases or not all(row.get('passed') for row in rows):
            raise ValueError('Required GPU invariance gate has not passed')
    if not pilot.get('strict_reconstruction_pass') or not pilot.get('fit', {}).get('completed') or pilot['fit'].get('optimizer_steps') != 6:
        raise ValueError('Required six-update pilot/reconstruction has not passed')
    if pilot != read(job/'report/ectt_result.json'):
        raise ValueError('Pilot report differs from its completed artifact store')
    return {'pilot_job': f['pilot_job'], 'source_commit': f['pilot_commit'],
            'source_dependency_hashes': hashes, 'prerequisites': records,
            'full_artifact_verification': full}


def batches(bounds, size):
    for start in range(bounds[0], bounds[1], size):
        yield list(range(start, min(start+size, bounds[1])))


def design(raw, origins, scaler, fitting=False):
    import numpy as np
    from ectt_data import features
    history = np.stack([raw[:, i-12:i] for i in origins])
    x = features(history, scaler)
    # Shared feature generator is FP32 by contract; solve/forecasts are FP64.
    x = x.astype(np.float64)
    a = np.concatenate([np.ones((*x.shape[:2], 1)), x.reshape(*x.shape[:2], 36)], axis=-1)
    residual = None
    if fitting:
        y = np.stack([raw[:, i] for i in origins])
        residual = (y-scaler['minimum'])/(scaler['maximum']-scaler['minimum'])-x[:, :, -1, 0]
    return a, residual, x[:, :, -1, 0]


def moments(raw, bounds, scaler, size):
    import numpy as np
    gram, rhs, count = np.zeros((37, 37)), np.zeros(37), 0
    for origins in batches(bounds, size):
        a, residual, _ = design(raw, origins, scaler, fitting=True)
        a, residual = a.reshape(-1, 37), residual.reshape(-1)
        gram += a.T@a; rhs += a.T@residual; count += len(residual)
    if count == 0:
        raise ValueError('No within-budget support targets for ridge')
    return gram/count, rhs/count


def solve(stats, alpha):
    import numpy as np
    penalty = np.eye(37)*alpha; penalty[0, 0] = 0
    weights = np.linalg.solve(stats[0]+penalty, stats[1])
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite ridge coefficients')
    return weights


def predict(raw, bounds, model, size):
    import numpy as np
    from ectt_data import inverse
    parts = []
    for origins in batches(bounds, size):
        if model['family'] == 'persistence':
            prediction = np.asarray(raw[:, np.asarray(origins)-1].T, dtype=np.float64)
        else:
            a, _, last = design(raw, origins, model['scaler'])
            prediction = inverse(last+a@np.asarray(model['weights']), model['scaler'])
        parts.append(prediction)
    if not parts:
        raise ValueError('Empty forecast cohort')
    value = np.concatenate(parts)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite forecasts')
    return value


def scores(prediction, truth):
    import numpy as np
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or not p.size or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid common forecast/truth cohort')
    error = p-y
    denominator = float(np.abs(y).sum())
    return {'MAE': float(np.abs(error).mean()), 'RMSE': float(np.sqrt(np.square(error).mean())),
            'WMAPE_percent': 100*float(np.abs(error).sum())/denominator if denominator else None,
            'origins': len(y), 'nodes': y.shape[1], 'valid_values': int(y.size),
            'zero_truth_values_included': int(np.count_nonzero(y == 0))}


def independent_scores(prediction, truth):
    import numpy as np
    pairs = list(zip(prediction, truth))
    absolute = math.fsum(float(np.linalg.norm(a-b, ord=1)) for a, b in pairs)
    squared = math.fsum(float(np.dot(a-b, a-b)) for a, b in pairs)
    denominator = math.fsum(float(np.linalg.norm(b, ord=1)) for _, b in pairs)
    count = truth.size
    return {'MAE': absolute/count, 'RMSE': math.sqrt(squared/count),
            'WMAPE_percent': 100*absolute/denominator if denominator else None}


def fit_case(raw, profile, budget, recipe):
    import numpy as np
    from ectt_data import fit_scaler
    persistence = {'family': 'persistence'}
    if float(budget) == 0 or not profile['budgets'][budget]['eligible_for_learned_adaptation']:
        return {'state': 'ZERO_BUDGET_FALLBACK' if float(budget) == 0 else 'INFEASIBLE_BUDGET_FALLBACK',
                'models': {'persistence': persistence, 'ridge': persistence, 'support_selected_persistence_ridge': persistence},
                'ridge_fitted': False, 'supervised_rows_accessed': [], 'support_scale': None,
                'reason': 'No added labels; all outputs retain persistence'}
    b = profile['budgets'][budget]; start, end = b['support_raw']
    split = start+int(recipe['selection_support_fraction']*(end-start))
    if split <= start+12 or split >= end:
        raise ValueError('Inner support fit/validation split is empty')
    validation = [split, end]
    truth = np.asarray(raw[:, split:end].T, dtype=np.float64)
    p_error = scores(predict(raw, validation, persistence, recipe['origin_batch']), truth)['MAE']
    candidates = [{'family': 'persistence', 'validation_MAE': p_error, 'alpha': None}]
    try:
        scaler = fit_scaler(raw, [start, split])
    except ValueError as exc:
        # Constant inner fitting history cannot normalize a learned candidate;
        # report this explicitly rather than borrowing held-out extrema.
        if 'Degenerate' not in str(exc):
            raise
        return {'state': 'RIDGE_UNAVAILABLE_DEGENERATE_INNER_SUPPORT', 'ridge_fitted': False,
                'models': {name: persistence for name in ('persistence', 'ridge', 'support_selected_persistence_ridge')},
                'supervised_rows_accessed': [start, end], 'support_scale': max(p_error, 1.),
                'inner_split': split, 'candidates': candidates, 'reason': str(exc)}
    stats = moments(raw, [start+12, split], scaler, recipe['origin_batch'])
    for alpha in recipe['ridge_alphas']:
        model = {'family': 'ridge', 'scaler': scaler, 'weights': solve(stats, alpha).tolist()}
        error = scores(predict(raw, validation, model, recipe['origin_batch']), truth)['MAE']
        candidates.append({'family': 'ridge', 'alpha': alpha, 'validation_MAE': error})
    # Stable min preserves candidate order for deterministic ties.
    best_ridge = min(candidates[1:], key=lambda row: row['validation_MAE'])
    selected = min(candidates, key=lambda row: row['validation_MAE'])
    final_scaler = fit_scaler(raw, [start, end])
    weights = solve(moments(raw, b['support_target_rows'], final_scaler, recipe['origin_batch']), best_ridge['alpha'])
    ridge = {'family': 'ridge', 'alpha': best_ridge['alpha'], 'scaler': final_scaler, 'weights': weights.tolist()}
    return {'state': 'FITTED', 'ridge_fitted': True, 'inner_split': split, 'candidates': candidates,
            'selected_family': selected['family'], 'support_scale': max(selected['validation_MAE'], 1.),
            'supervised_rows_accessed': [start, end],
            'models': {'persistence': persistence, 'ridge': ridge,
                       'support_selected_persistence_ridge': ridge if selected['family'] == 'ridge' else persistence}}


def save_case(raw, profile, budget, recipe, store, ident):
    import numpy as np
    fit = fit_case(raw, profile, budget, recipe)
    bounds = profile['test_targets']; origins = np.arange(*bounds, dtype=np.int64)
    truth = np.asarray(raw[:, bounds[0]:bounds[1]].T, dtype=np.float64)
    predictions = {name: predict(raw, bounds, model, recipe['origin_batch']) for name, model in fit['models'].items()}
    metrics = {name: scores(value, truth) for name, value in predictions.items()}
    store.mkdir(parents=True, exist_ok=True)
    temporary = store/'forecasts.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, origins=origins, truth=truth, **predictions)
    temporary.replace(store/'forecasts.npz')
    # Recalculate from saved physical-unit values before marking completion.
    with np.load(store/'forecasts.npz', allow_pickle=False) as payload:
        if not np.array_equal(payload['origins'], origins) or not np.array_equal(payload['truth'], truth):
            raise ValueError('Saved cohort/truth differs')
        independent = {}
        for name in predictions:
            independent[name] = independent_scores(payload[name], payload['truth'])
            for metric, value in independent[name].items():
                expected = metrics[name][metric]
                if (expected is None) != (value is None) or (value is not None and not math.isclose(value, expected, rel_tol=1e-11, abs_tol=1e-11)):
                    raise ValueError('Independent saved-forecast metrics differ')
    write(store/'independent_metrics.json', independent)
    write(store/'fit.json', fit)
    access = {'supervised_raw_rows': fit['supervised_rows_accessed'],
              'supervised_unique_values': (fit['supervised_rows_accessed'][1]-fit['supervised_rows_accessed'][0])*raw.shape[0]
                  if fit['supervised_rows_accessed'] else 0,
              'test_targets': bounds, 'history_length': 12, 'stride': 1,
              'shared_test_history_row_union': [bounds[0]-12, bounds[1]-1],
              'verification_labels_used_for_fitting_or_selection': 0,
              'test_labels_used_for_fitting_or_selection': 0,
              'truth_access': 'test outcomes read only for fixed-model evaluation',
              'zero_budget_note': 'Histories are shared covariates; no target supervision added'}
    write(store/'access_ledger.json', access)
    result = {'budget': budget, 'state': fit['state'], 'metrics': metrics, 'ridge_fitted': fit['ridge_fitted'],
              'selected_family': fit.get('selected_family', 'persistence'),
              'deterministic_seed_references': recipe['seeds'], 'independent_seed_replications': False,
              'final_fallback_family_complete': False}
    write(store/'result.json', result); finish(store, ident)
    return result


def run(project, code, report, stop_after=None):
    import numpy as np
    from ectt_data import verify_profile
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    inside(project, code); inside(project, report)
    recipe = config(code); ident = identity(code)
    store = inside(project, project/'outputs/ectt/shared_baselines'/ident)
    store.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    state = {'stage': STAGE, 'identity': ident, 'store': str(store), 'completed_cases': 0,
             'total_cases': 32, 'benchmark_experiments_completed': 0, 'new_ridge_fits': 0,
             'training_performed': False, 'optimizer_steps': 0}
    def progress(**values):
        state.update(values, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(report/'baseline_progress.json', state)
        print('BASELINE_PROGRESS', json.dumps(state), flush=True)
    began = time.monotonic(); results = {}
    with (store/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress(phase='VERIFYING_FOUNDATION')
            evidence = foundation(project, code, full=True)
            write(report/'baseline_foundation.json', evidence)
            from statistical_audit import review
            review_store = inside(store, store/'internal_statistical_review')
            if completed(review_store, ident):
                audit = read(review_store/'internal_review.json')
            else:
                audit = review(review_store, code, progress); finish(review_store, ident)
            write(report/'internal_statistical_review.json', audit)
            contract = read(code/'config/ectt_protocol_v1.json')
            source_lock = read(code/'config/sources.lock.json')
            for slug, profile in contract['datasets'].items():
                verify_profile(profile)
                path = inside(project, project/'work/reference/snapshots'/source_lock['reference_commit']/profile['path'])
                if digest(path) != profile['container_sha256']:
                    raise ValueError('Canonical data digest changed: '+slug)
                raw = np.load(path, mmap_mode='r', allow_pickle=False)
                if list(raw.shape) != profile['shape'] or raw.dtype.kind not in 'fiu':
                    raise ValueError('Canonical shape/type changed: '+slug)
                raw = raw[..., 0]
                if not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError('Invalid canonical raw channel: '+slug)
                for budget in recipe['budgets']:
                    key = slug+'/'+budget; case = inside(store, store/'cases'/slug/budget)
                    reused = completed(case, ident)
                    result = read(case/'result.json') if reused else save_case(raw, profile, budget, recipe, case, ident)
                    results[key] = result
                    new_fit = int(not reused and result['ridge_fitted'])
                    progress(phase='BASELINES', dataset=slug, budget=budget, reused_case=reused,
                             completed_cases=len(results), benchmark_experiments_completed=len(results),
                             new_ridge_fits=state['new_ridge_fits']+new_fit,
                             training_performed=bool(state['new_ridge_fits']+new_fit))
                    write(report/'baseline_partial_results.json', results)
                    if stop_after is not None and len(results) >= stop_after:
                        return 75
            output = {'stage': STAGE, 'identity': ident, 'cases': results, 'completed_cases': len(results),
                      'benchmark_experiments_completed': len(results), 'deterministic_unique_cases': len(results),
                      'seed_references': recipe['seeds'], 'new_ridge_fits': state['new_ridge_fits'],
                      'ridge_cases_fitted': sum(row['ridge_fitted'] for row in results.values()),
                      'new_ridge_parameter_solves': state['new_ridge_fits']*(len(recipe['ridge_alphas'])+1),
                      'training_performed': state['training_performed'], 'optimizer_steps': 0,
                      'S01_independent_review': 'PENDING', 'method_evaluated': False,
                      'family_readiness': recipe['family_readiness'], 'elapsed_seconds': time.monotonic()-began,
                      'scientific_status': 'LOCAL_BASELINE_FOUNDATION_ONLY', 'all_test_origins_scored': True}
            write(store/'result.json', output); write(report/'baseline_result.json', output)
            metadata = read(code/'.source_snapshot.json') if (code/'.source_snapshot.json').exists() else {}
            import resource
            runtime = {'code_commit': metadata.get('commit', 'UNVERSIONED_LOCAL_TEST'),
                       'job_id': os.environ.get('SLURM_JOB_ID'), 'python': sys.version, 'numpy': np.__version__,
                       'allocated_cpus': os.environ.get('SLURM_CPUS_PER_TASK'), 'requested_gpus': 0,
                       'peak_host_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       'elapsed_seconds': output['elapsed_seconds'],
                       'recipe_sha256': digest(code/'config/shared_baselines_v1.json'),
                       'scientific_contract_sha256': recipe['scientific_contract_sha256']}
            write(store/'runtime.json', runtime); write(report/'baseline_runtime.json', runtime)
            rows = []
            for key, result in results.items():
                slug, budget = key.split('/')
                for method, metrics in result['metrics'].items():
                    rows.append({'dataset': slug, 'budget': budget, 'method': method, 'case_state': result['state'],
                                 'selected_family': result['selected_family'], 'seed_references': '11;22;33', **metrics})
            with (store/'baseline_results.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            shutil.copyfile(store/'baseline_results.csv', report/'baseline_results.csv')
            lines = ['# Shared local baseline foundation', '', '32 deterministic dataset-budget cases; seeds reference identical fits.',
                     '', '| Dataset | Budget | Method | MAE | RMSE | WMAPE % |', '|---|---|---|---:|---:|---:|']
            for key, row in results.items():
                slug, budget = key.split('/')
                for method, m in row['metrics'].items():
                    lines.append('| %s | %s | %s | %.6f | %.6f | %s |' %
                                 (slug, budget, method, m['MAE'], m['RMSE'], str(m['WMAPE_percent'])))
            lines += ['', 'B01 persistence is implemented; B02 ridge is implemented, compact candidate pending.',
                      'B03/B04 and the ECTT objective remain pending. S01 independent sign-off is pending.',
                      'Ordered rows only; timestamps/sensor provenance remain unverified.',
                      'Physical WMAPE includes zero truths. Completed cases resume after artifact verification.',
                      'Store: '+str(store)]
            (store/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
            shutil.copyfile(store/'SUMMARY.md', report/'SUMMARY.md')
            write(report/'baseline_store.json', {'store': str(store), 'identity': ident})
            progress(phase='COMPLETED')
            print((report/'SUMMARY.md').read_text(), flush=True)
            return 0
        except Exception as exc:
            progress(phase='FAILED', error=str(exc))
            (report/'failure_traceback.txt').write_text(traceback.format_exc())
            (report/'SUMMARY.md').write_text('# Shared baseline stage failed\n\n'+str(exc)+'\n\nPartial cases: '+str(store)+'\n')
            return 2


def dispatch(project, code, report):
    python = inside(project, project/'.venv/bin/python')
    if not python.exists():
        raise ValueError('Existing project NumPy environment missing; no shared installation allowed')
    import subprocess
    return subprocess.call([str(python), '-u', str(code/'scripts/shared_baselines.py'),
                            '--project', str(project), '--code', str(code), '--report', str(report)])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        parser.error('Use the Slurm launcher for shared baseline execution')
    if not (args.code/'.source_snapshot.json').is_file():
        parser.error('A verified immutable code snapshot is required for HPC execution')
    return run(args.project, args.code, args.report)


if __name__ == '__main__':
    sys.exit(main())
