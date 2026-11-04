#!/usr/bin/env python3
"""Development-only compact local pilot, preserving the completed CPU cases."""
import argparse
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
import shared_baselines as base

STAGE = 'shared-compact-pilot'
DEPENDENCIES = base.DEPENDENCIES+('config/compact_local_v1.json', 'scripts/compact_model.py',
    'scripts/compact_training.py', 'scripts/compact_baselines.py', 'scripts/ectt_model.py',
    'scripts/ectt_env.py', 'config/ectt_gpu_requirements.txt')


def config(code):
    recipe = read(code/'config/compact_local_v1.json')
    if digest(code/'config/ectt_protocol_v1.json') != recipe['scientific_contract_sha256']:
        raise ValueError('Compact scientific contract changed; amendment required')
    contract = read(code/'config/ectt_protocol_v1.json')
    if recipe['datasets'] != contract['development_targets'] or recipe['seed'] != 11 or recipe['budget'] != '0.05':
        raise ValueError('Compact pilot must remain development-only, seed 11, primary budget')
    if recipe['updates'] != contract['training_design']['target_updates'] or recipe['learning_rate'] != contract['training_design']['target_learning_rate']:
        raise ValueError('Compact update allowance differs from the frozen target recipe')
    if not 1 <= len(recipe['configurations']) <= contract['training_design']['development_grid_limit_per_family']:
        raise ValueError('Compact configuration allowance exceeded')
    if recipe['selection_support_fraction'] != .7 or recipe['selection_order'] != ['persistence', 'ridge', 'compact']:
        raise ValueError('Changed compact support-selection contract')
    return recipe


def identity(code):
    return hashlib.sha256(json.dumps({p: digest(code/p) for p in DEPENDENCIES}, sort_keys=True).encode()).hexdigest()


def foundation(project, code, full=False):
    recipe = config(code)
    evidence = base.foundation(project, code, full=full)
    ident = base.identity(code)
    if ident != recipe['baseline_identity']:
        raise ValueError('Completed persistence/ridge implementation changed; original fits preserved')
    job = inside(project, project/'results/slurm_jobs'/recipe['baseline_job'])
    status = read(job/'job_status.json')
    if status.get('stage') != base.STAGE or status.get('execution_state') != 'COMPLETED' or status.get('shell_exit_code') != 0:
        raise ValueError('Required shared persistence/ridge job is not completed')
    store = inside(project, project/'outputs/ectt/shared_baselines'/ident)
    result = read(store/'result.json')
    if result.get('identity') != ident or result.get('completed_cases') != 32 or result.get('ridge_cases_fitted') != 22:
        raise ValueError('Expected 32 baseline cases and 22 ridge fits')
    if result != read(job/'report/baseline_result.json'):
        raise ValueError('CPU baseline report differs from its artifact store')
    if full:
        if len(result['cases']) != 32:
            raise ValueError('Incomplete baseline case inventory')
        for key, row in result['cases'].items():
            path = inside(store, store/'cases'/key)
            if not completed(path, ident) or read(path/'result.json') != row:
                raise ValueError('Missing or inconsistent completed baseline case: '+key)
    return {'baseline_job': recipe['baseline_job'], 'baseline_identity': ident, 'store': str(store),
            'full_artifact_verification': full, 'admission': evidence}


def save_case(raw, profile, slug, recipe, prior, store, ident, device, progress):
    import numpy as np
    from compact_training import fit, predict, data_batch, recovery_probe
    from ectt_data import fit_scaler
    from ectt_model import invariance_checks
    budget = recipe['budget']; seed = recipe['seed']
    bounds = profile['budgets'][budget]
    if not bounds['eligible_for_learned_adaptation']:
        raise ValueError('Development pilot unexpectedly infeasible; no extra labels allowed')
    store.mkdir(parents=True, exist_ok=True)
    old_fit = read(prior/'fit.json')
    start, end = bounds['support_raw']; split = start+int(.7*(end-start))
    if old_fit.get('state') != 'FITTED' or old_fit['inner_split'] != split:
        raise ValueError('CPU and compact support cohorts differ')
    inner_scaler = fit_scaler(raw, [start, split]); final_scaler = fit_scaler(raw, [start, end])
    if final_scaler != old_fit['models']['ridge']['scaler']:
        raise ValueError('CPU/compact support scaler mismatch')
    inner_bounds = [start+12, split]
    import torch
    from compact_model import CompactTrafficModel
    torch.manual_seed(seed)
    admission_model = CompactTrafficModel(**recipe['configurations'][-1]).to(device)
    admission_x, _ = data_batch(raw, [inner_bounds[0], inner_bounds[0]+1], inner_scaler, device)
    write(store/'pretraining_invariance.json', invariance_checks(admission_model, admission_x, atol=1e-5, rtol=1e-4))
    del admission_model
    # This probe reads only fitting support and runs the largest architecture.
    progress(fit_key=slug+'/recovery_probe', phase='RECOVERY_PROBE')
    recovery = recovery_probe(raw, inner_bounds, inner_scaler, recipe['configurations'][-1],
        recipe, seed, store/'recovery_probe', ident, device,
        lambda **v: progress(fit_key=slug+'/probe/'+v.pop('probe_arm'), phase='RECOVERY_PROBE', **v))
    records = []
    for index, architecture in enumerate(recipe['configurations']):
        key = slug+'/inner/'+str(index)
        model, info = fit(raw, inner_bounds, inner_scaler, architecture, recipe, seed,
            store/'inner'/str(index), ident, device,
            lambda **v: progress(fit_key=key, phase='INNER_FIT', **v))
        x, _ = data_batch(raw, [inner_bounds[0], inner_bounds[0]+1], inner_scaler, device)
        gates = invariance_checks(model, x, atol=1e-5, rtol=1e-4)
        prediction = predict(model, raw, [split, end], inner_scaler, recipe['prediction_origin_batch'], device)
        error = base.scores(prediction, np.asarray(raw[:, split:end].T, dtype=np.float64))['MAE']
        row = {'index': index, 'architecture': architecture, 'validation_MAE': error,
               'fit': {k: v for k, v in info.items() if k != 'losses'}, 'invariance': gates}
        records.append(row); write(store/'compact_validation.json', records)
        del model
    best = min(records, key=lambda row: row['validation_MAE'])
    p_error = next(row['validation_MAE'] for row in old_fit['candidates'] if row['family'] == 'persistence')
    r_error = min(row['validation_MAE'] for row in old_fit['candidates'] if row['family'] == 'ridge')
    candidates = [{'family': family, 'validation_MAE': value} for family, value in
                  zip(recipe['selection_order'], (p_error, r_error, best['validation_MAE']))]
    selected = min(candidates, key=lambda row: row['validation_MAE'])
    selection = {'selected_family': selected['family'], 'support_scale': max(selected['validation_MAE'], 1.),
                 'candidates': candidates, 'best_compact_configuration': best['index'],
                 'inner_split': split, 'inner_scaler': inner_scaler, 'final_scaler': final_scaler,
                 'frozen_before_test': True, 'verification_labels_used': 0, 'test_labels_used': 0}
    write(store/'selection.json', selection)
    model, fit_info = fit(raw, bounds['support_target_rows'], final_scaler, best['architecture'], recipe, seed,
        store/'refit', ident, device, lambda **v: progress(fit_key=slug+'/refit', phase='SUPPORT_REFIT', **v))
    origins = bounds['support_target_rows']
    x, _ = data_batch(raw, [origins[0], origins[0]+1], final_scaler, device)
    gates = invariance_checks(model, x, atol=1e-5, rtol=1e-4)
    write(store/'posttraining_invariance.json', gates)
    pointer = store/'refit/latest.json'
    model_hash = digest(store/'refit'/read(pointer)['file'])
    test = profile['test_targets']
    prediction = predict(model, raw, test, final_scaler, recipe['prediction_origin_batch'], device)
    with np.load(prior/'forecasts.npz', allow_pickle=False) as old:
        truth = np.asarray(raw[:, test[0]:test[1]].T, dtype=np.float64)
        ids = np.arange(*test, dtype=np.int64)
        if not np.array_equal(old['origins'], ids) or not np.array_equal(old['truth'], truth):
            raise ValueError('Compact and CPU forecast cohorts/raw truths differ')
        predictions = {'persistence': old['persistence'].copy(), 'ridge': old['ridge'].copy(), 'compact': prediction}
    predictions['support_selected_local'] = predictions[selected['family']]
    if digest(store/'refit'/read(pointer)['file']) != model_hash:
        raise ValueError('Model checkpoint changed during fixed-model test evaluation')
    temporary = store/'forecasts.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, origins=ids, truth=truth, **predictions)
    temporary.replace(store/'forecasts.npz')
    metrics = {k: base.scores(v, truth) for k, v in predictions.items()}
    independent = {}
    with np.load(store/'forecasts.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['truth'], truth) or not np.array_equal(saved['origins'], ids):
            raise ValueError('Saved compact cohort/truth changed')
        for key in predictions:
            independent[key] = base.independent_scores(saved[key], saved['truth'])
            for name, value in independent[key].items():
                expected = metrics[key][name]
                if (value is None) != (expected is None) or (value is not None and not math.isclose(value, expected, rel_tol=1e-11, abs_tol=1e-11)):
                    raise ValueError('Independent compact saved-forecast metric mismatch')
    write(store/'independent_metrics.json', independent)
    write(store/'access_ledger.json', {'support_raw_rows': [start, end], 'unique_supervised_values': (end-start)*raw.shape[0],
          'inner_fitting_raw': [start, split], 'selection_targets': [split, end],
          'refit_target_rows': bounds['support_target_rows'], 'test_targets': test,
          'shared_test_history_row_union': [test[0]-12, test[1]-1],
          'verification_labels_used_for_fitting_or_selection': 0, 'test_labels_used_for_fitting_or_selection': 0,
          'fit_reuse': 'Ridge/persistence copied from verified 620562; repeated uses of support count once',
          'test_truth_role': 'fixed-model evaluation only; every eligible test origin retained'})
    result = {'dataset': slug, 'budget': budget, 'seed': seed, 'metrics': metrics,
              'selection': selection, 'compact_configuration': best['architecture'],
              'optimizer_steps': recipe['updates']*(len(recipe['configurations'])+1),
              'engineering_probe_optimizer_steps': 2*recipe['recovery_probe_updates'],
              'recovery_probe_passed': recovery['passed'], 'posttraining_invariance_passed': len(gates),
              'final_parameters': fit_info['parameters'], 'checkpoint_sha256': model_hash,
              'development_only': True, 'S01_independent_review': 'PENDING', 'method_evaluated': False}
    write(store/'result.json', result); finish(store, ident)
    return result


def run(project, code, report, device='cuda'):
    import numpy as np
    import torch
    from compact_training import configure
    from ectt_data import verify_profile
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    inside(project, code); inside(project, report)
    recipe = config(code); ident = identity(code)
    store = inside(project, project/'outputs/ectt/compact_local'/ident)
    store.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    state = {'stage': STAGE, 'identity': ident, 'store': str(store), 'completed_cases': 0,
             'total_cases': len(recipe['datasets']), 'new_optimizer_steps': 0, 'training_performed': False}
    new_by_fit = {}; results = {}; began = time.monotonic()
    def progress(**values):
        key = values.get('fit_key')
        if key is not None and 'new_optimizer_steps' in values:
            new_by_fit[key] = values.pop('new_optimizer_steps')
        state.update(values, new_optimizer_steps=sum(new_by_fit.values()),
                     training_performed=sum(new_by_fit.values()) > 0,
                     updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(report/'compact_progress.json', state)
        print('COMPACT_PROGRESS', json.dumps(state), flush=True)
    with (store/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress(phase='VERIFYING_FOUNDATION')
            evidence = foundation(project, code, full=True); write(report/'compact_foundation.json', evidence)
            if device == 'cuda':
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or os.environ.get('SLURM_CPUS_PER_TASK') != '5':
                    raise ValueError('Compact pilot requires exactly one visible CUDA GPU and five CPUs')
                if torch.__version__ != '2.4.1+cu118' or np.__version__ != '1.26.4':
                    raise ValueError('Compact pinned GPU environment differs')
            configure()
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            contract = read(code/'config/ectt_protocol_v1.json'); source_lock = read(code/'config/sources.lock.json')
            write(report/'compact_store.json', {'identity': ident, 'store': str(store)})
            for slug in recipe['datasets']:
                profile = contract['datasets'][slug]; verify_profile(profile)
                path = inside(project, project/'work/reference/snapshots'/source_lock['reference_commit']/profile['path'])
                if digest(path) != profile['container_sha256']:
                    raise ValueError('Canonical data digest changed: '+slug)
                raw = np.load(path, mmap_mode='r', allow_pickle=False)
                if list(raw.shape) != profile['shape'] or raw.dtype.kind not in 'fiu':
                    raise ValueError('Canonical shape/type changed: '+slug)
                raw = raw[..., 0]
                if not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError('Invalid canonical raw channel: '+slug)
                case = inside(store, store/'cases'/slug/recipe['budget']/str(recipe['seed']))
                reused = completed(case, ident)
                result = read(case/'result.json') if reused else save_case(raw, profile, slug, recipe,
                    Path(evidence['store'])/'cases'/slug/recipe['budget'], case, ident, device, progress)
                results[slug] = result
                write(report/'compact_partial_results.json', results)
                progress(phase='COMPACT_CASES', dataset=slug, completed_cases=len(results), reused_case=reused)
            output = {'stage': STAGE, 'identity': ident, 'cases': results, 'completed_cases': len(results),
                      'benchmark_experiments_completed': len(results), 'development_only': True,
                      'benchmark_optimizer_steps': sum(row['optimizer_steps'] for row in results.values()),
                      'optimizer_steps': sum(row['optimizer_steps']+row['engineering_probe_optimizer_steps'] for row in results.values()),
                      'new_optimizer_steps': state['new_optimizer_steps'], 'training_performed': state['training_performed'],
                      'scientific_status': 'COMPACT_DEVELOPMENT_PILOT_ONLY', 'method_evaluated': False,
                      'S01_independent_review': 'PENDING', 'compact_all_domain_grid_complete': False,
                      'elapsed_seconds': time.monotonic()-began}
            metadata = read(code/'.source_snapshot.json') if (code/'.source_snapshot.json').exists() else {}
            import resource
            runtime = {'code_commit': metadata.get('commit', 'UNVERSIONED_SYNTHETIC_TEST'),
                       'job_id': os.environ.get('SLURM_JOB_ID'), 'python': sys.version, 'numpy': np.__version__,
                       'torch': torch.__version__, 'cuda': torch.version.cuda, 'allocated_cpus': 5,
                       'visible_gpus': torch.cuda.device_count() if device == 'cuda' else 0,
                       'gpu': torch.cuda.get_device_name() if device == 'cuda' else 'SYNTHETIC_CPU',
                       'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                       'peak_host_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       'elapsed_seconds': output['elapsed_seconds'], 'new_optimizer_steps_including_recovery_probes': state['new_optimizer_steps'],
                       'scientific_contract_sha256': recipe['scientific_contract_sha256'],
                       'recipe_sha256': digest(code/'config/compact_local_v1.json'),
                       'qwen_full_size_interrupted_optimizer_resume_verified': False}
            write(report/'compact_result.json', output); write(report/'compact_runtime.json', runtime)
            # Completed nested cases are immutable; root files describe this invocation.
            write(store/'result.json', output); write(store/'runtime.json', runtime)
            lines = ['# Compact local development pilot', '', 'PEMS08 and Beijing; 5% budget; seed 11.', '',
                     '| Dataset | Method | MAE | RMSE | WMAPE % |', '|---|---|---:|---:|---:|']
            for slug, row in results.items():
                for method, m in row['metrics'].items():
                    lines.append('| %s | %s | %.6f | %.6f | %s |' % (slug, method, m['MAE'], m['RMSE'], m['WMAPE_percent']))
            lines += ['', 'Four fixed compact architectures selected on support only; best compact refit on all support.',
                      'Exact compact model/Adam/RNG/cursor recovery and trained-model invariance gates passed.',
                      'Original 620562 persistence/ridge stores are unchanged. No verification labels used.',
                      'Development pilot only: six confirmatory domains, other seeds/budgets and full B02 remain pending.',
                      'No ECTT objective, transfer certificate or independent S01 sign-off.',
                      'Qwen full-size interrupted optimizer recovery remains pending before long Qwen grids.',
                      'Ordered-row benchmark; timestamps and raw sensor provenance remain unverified.', 'Store: '+str(store)]
            (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n'); shutil.copyfile(report/'SUMMARY.md', store/'SUMMARY.md')
            progress(phase='COMPLETED'); print((report/'SUMMARY.md').read_text(), flush=True)
            return 0
        except Exception as exc:
            progress(phase='FAILED', error=str(exc))
            (report/'failure_traceback.txt').write_text(traceback.format_exc())
            (report/'SUMMARY.md').write_text('# Compact pilot failed\n\n'+str(exc)+'\n\nPartial artifacts: '+str(store)+'\n')
            return 2


def dispatch(project, code, report):
    foundation(project, code, full=False)
    from ectt_env import ensure
    with (project/'work/ectt_environment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/compact_baselines.py'),
                            '--project', str(project), '--code', str(code), '--report', str(report)],
                           env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID', '').isdigit() or not (args.code/'.source_snapshot.json').is_file():
        parser.error('Use the Slurm launcher and verified source snapshot')
    return run(args.project, args.code, args.report)


if __name__ == '__main__':
    sys.exit(main())
