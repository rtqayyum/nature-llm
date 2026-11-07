"""Paired development loss experiment with probe, fit and collector stages."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
import transfer_plan as baseline
import transfer_parallel as prior

LAUNCH = 'shared-transfer-loss-development'
PROBE = 'shared-transfer-loss-probe'
FIT = 'shared-transfer-loss-fit'
REPORT = 'shared-transfer-loss-report'
STAGES = (PROBE, FIT, REPORT)
GPU_STAGES = (PROBE, FIT)


def matrix(code):
    parent = baseline.plan(code)
    recipe = read(code/'config/source_loss_development_v1.json')
    contract, _ = load_contract(code)
    if (recipe['arms'] != ['clipped', 'unclipped'] or recipe['development_targets'] != contract['development_targets']
            or recipe['seeds'] != contract['seeds'] or recipe['budget'] != str(contract['partitions']['primary_budget'])
            or recipe['maximum_concurrent_gpu_tasks'] != 4):
        raise ValueError('Unexpected development loss comparison recipe')
    tasks = []
    for row in parent['source']:
        if row['fold'] in recipe['development_targets']:
            for arm in recipe['arms']:
                tasks.append({**row, 'index': len(tasks), 'baseline_source_index': row['index'],
                              'dataset': row['fold'], 'arm': arm})
    probes = [next(row for row in tasks if row['family'] == family and row['arm'] == arm
                   and row['dataset'] == 'pems08' and row['seed'] == 11)
              for family in parent['recipe']['families'] for arm in recipe['arms']]
    identity = baseline.fingerprint({'baseline': parent['identity'],
        'recipe': digest(code/'config/source_loss_development_v1.json'),
        'worker': digest(code/'scripts/source_loss_development.py'),
        'trainer': digest(code/'scripts/source_loss_train.py')})
    if len(tasks) != 24 or len(probes) != 4:
        raise ValueError('Unexpected paired development matrix')
    return {'identity': identity, 'recipe': recipe, 'baseline': parent, 'tasks': tasks, 'probes': probes}


def foundation(project, code):
    prior.foundation(project, code)
    for job_id, stage, science in (
            ('676089', prior.CPU, 'BASELINE_UNCERTAINTY_COMPLETE'),
            ('676092', prior.GPU, 'SOURCE_GRADIENT_AUDIT_COMPLETE')):
        job = inside(project, project/'results/slurm_jobs'/job_id)
        state = read(job/'job_status.json')
        result = read(job/'report/transfer_result.json')
        if (state['stage'], state['execution_state'], state['shell_exit_code']) != (stage, 'COMPLETED', 0):
            raise ValueError('Required completed analysis '+job_id+' is unavailable')
        if result['scientific_status'] != science:
            raise ValueError('Unexpected predecessor analysis status')
        if stage == prior.GPU and (result.get('diagnostic_cases_checked') != 12
                                  or result.get('source_validation_suffixes_checked') != 84
                                  or result.get('source_backbone_updates') != 0):
            raise ValueError('Source gradient audit case coverage differs')
    return matrix(code)


def store_path(project, plan):
    return inside(project, project/'outputs/ectt/source_loss_development'/plan['identity'])


def train_source(project, code, report, store, task, plan, contract, progress, probe=False):
    import torch
    from source_loss_train import fit
    from transfer_train import episode_bank, invariant_probe, load, equal_state
    from transfer_worker import gpu, local_recipes, raw_arrays, raw_paths

    recipe = plan['baseline']['recipe']
    design = {**contract['training_design'], **({'source_max_epochs': 1} if probe else {})}
    raw = raw_arrays(raw_paths(project, code, contract), task['sources'], contract)
    bank = episode_bank(task, contract, recipe, raw)
    write(store/'episode_ledger.json', {'task': task, 'episodes': bank})
    began = time.monotonic()
    model, runtime = gpu(project, code, contract, task, report)
    key = baseline.fingerprint({'experiment': plan['identity'], 'task': task, 'bank': bank,
                                'runtime': runtime, 'probe': probe})
    slug = max(task['sources'], key=lambda name: raw[name].shape[0])
    support = bank[slug]['train'][0]['support']
    gates_before = invariant_probe(model, raw[slug], support, task['family'], 'cuda')
    args = (raw, task, bank, recipe, design, local_recipes(code), 'cuda')
    cursor, new_steps = fit(model, *args, store/'source', key, progress, task['arm'])
    if probe:
        full = load(store/'source', key)
        del model
        torch.cuda.empty_cache()
        model, _ = gpu(project, code, contract, task, report)
        resumed = store/'resumed'
        probe_updates = new_steps
        if not (resumed/'latest.json').exists():
            _, updates = fit(model, *args, resumed, key, progress, task['arm'], stop_after=2)
            probe_updates += updates
        del model
        torch.cuda.empty_cache()
        model, _ = gpu(project, code, contract, task, report)
        _, updates = fit(model, *args, resumed, key, progress, task['arm'])
        probe_updates += updates
        if not equal_state(full, load(resumed, key)):
            raise ValueError('Paired loss source recovery differs')
        gates_after = invariant_probe(model, raw[slug], support, task['family'], 'cuda')
        runtime.update(elapsed_seconds=time.monotonic()-began, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        return {'task': task, 'runtime': runtime, 'scientific_status': 'PAIRED_LOSS_RECOVERY_PASSED',
                'new_optimizer_steps': probe_updates,
                'invariance_before': gates_before, 'invariance_after': gates_after}
    from transfer_train import restore_model, model_state
    restore_model(model, cursor['best_model'])
    gates_after = invariant_probe(model, raw[slug], support, task['family'], 'cuda')
    torch.save(model_state(model), store/'best.tmp')
    (store/'best.tmp').replace(store/'best.pt')
    runtime.update(elapsed_seconds=time.monotonic()-began, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    return {'task': task, 'runtime': runtime, 'epochs': cursor['epoch'], 'source_steps': cursor['step'],
            'new_optimizer_steps': new_steps, 'validation_history': cursor['history'],
            'best_validation_normalized_MAE': cursor['best'], 'best_sha256': digest(store/'best.pt'),
            'invariance_before': gates_before, 'invariance_after': gates_after}


def target_fit(project, code, report, store, task, plan, contract, source, progress):
    import numpy as np
    import torch
    from ectt_data import fit_scaler, inverse
    from shared_baselines import independent_scores
    from transfer_train import adapt, invariant_probe, restore_model
    from transfer_attribution import predict_pair
    from transfer_worker import gpu, raw_arrays, raw_paths, save_arrays
    import transfer_diagnostics as diagnostics

    budget = plan['recipe']['budget']
    profile = contract['datasets'][task['dataset']]
    support = profile['budgets'][budget]['support_raw']
    raw = raw_arrays(raw_paths(project, code, contract), [task['dataset']], contract)[task['dataset']]
    scaler = fit_scaler(raw, support)
    began = time.monotonic()
    model, runtime = gpu(project, code, contract, task, report)
    restore_model(model, torch.load(store/'best.pt', map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    key = baseline.fingerprint({'experiment': plan['identity'], 'task': task, 'source': source['best_sha256'],
                                'scaler': scaler, 'runtime': runtime})
    head, cursor = adapt(model, raw, profile['budgets'][budget]['support_target_rows'], scaler, task['seed'],
                         plan['baseline']['recipe'], contract['training_design'], 'cuda', progress, store/'target', key)
    gates = invariant_probe(model, raw, support, task['family'], 'cuda', head)
    source_z, target_z = predict_pair(model, head, raw, profile['test_targets'], scaler,
                                      plan['baseline']['recipe']['prediction_origin_batch'], 'cuda')
    predictions = inverse(target_z, scaler)
    original_task = next(row for row in plan['baseline']['target'] if row['dataset'] == task['dataset']
                         and row['family'] == task['family'] and row['seed'] == task['seed'])
    original_store = baseline.store_path(project, plan['baseline'])
    directory, artifacts = diagnostics.worker_artifacts(project, original_task, original_store, 'target', baseline.TARGET)
    original = diagnostics.forecasts(diagnostics.checked(directory, artifacts, budget+'/test_forecasts.npz'),
                                     profile['test_targets'], raw.shape[0], include_truth=True)
    if not np.array_equal(original['truth'], raw[:, slice(*profile['test_targets'])].T):
        raise ValueError('Paired target truth cohort differs')
    if digest(store/'best.pt') != source['best_sha256']:
        raise ValueError('Source checkpoint changed during target scoring')
    save_arrays(store/'test_forecasts.npz', origins=np.arange(*profile['test_targets']),
                truth=original['truth'], local=original['local'], transfer=predictions,
                source_normalized=source_z, adapted_normalized=target_z)
    span = scaler['maximum']-scaler['minimum']
    runtime.update(elapsed_seconds=time.monotonic()-began, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    return {'runtime': runtime,
            'source_clipped_fraction': float(np.mean(scaler['minimum']+source_z.astype(np.float64)*span < 0)),
            'adapted_clipped_fraction': float(np.mean(scaler['minimum']+target_z.astype(np.float64)*span < 0)),
            'metrics': {'local': independent_scores(original['local'], original['truth']),
                        'transfer': independent_scores(predictions, original['truth'])},
            'new_optimizer_steps': cursor['new_steps'], 'optimizer_steps': cursor['step'],
            'scaler': scaler, 'support_raw': support, 'test_targets': profile['test_targets'],
            'invariance': gates, 'verification_labels_used': 0}


def collection(project, code, report, plan):
    import numpy as np
    from shared_baselines import independent_scores

    root = store_path(project, plan)
    submission = read(project/'work/source_loss_submissions'/(plan['identity']+'.json'))
    jobs = submission['jobs']
    records, failures = [], {}
    for task in plan['tasks']:
        label = str(task['index'])
        try:
            key = jobs['fit']+'_'+label
            job = project/'results/slurm_jobs'/key
            status = read(job/'job_status.json')
            if status['execution_state'] != 'COMPLETED' or status['shell_exit_code'] != 0 or status['stage'] != FIT:
                raise ValueError('Fit task did not complete successfully')
            case = root/'cases'/label
            if not completed(case, plan['identity']):
                raise ValueError('Missing completed loss comparison case')
            result = read(case/'result.json')
            if result['task'] != task:
                raise ValueError('Task identity differs')
            with np.load(case/'test_forecasts.npz', allow_pickle=False) as values:
                metric = independent_scores(values['transfer'], values['truth'])
                local = independent_scores(values['local'], values['truth'])
            if metric != result['target']['metrics']['transfer'] or local != result['target']['metrics']['local']:
                raise ValueError('Independent saved metric differs')
            records.append({**{k: task[k] for k in ('family', 'dataset', 'seed', 'arm')},
                            'source_epochs': result['source']['epochs'],
                            'source_optimizer_steps': result['source']['source_steps'],
                            'source_validation_MAE': result['source']['best_validation_normalized_MAE'],
                            'test_MAE': metric['MAE'], 'local_MAE': local['MAE'],
                            'source_clipped_fraction': result['target']['source_clipped_fraction'],
                            'adapted_clipped_fraction': result['target']['adapted_clipped_fraction']})
        except Exception as exc:
            failures[label] = str(exc)
    if records:
        with (report/'source_loss_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = ['# Paired development source loss comparison', '',
             f'Verified {len(records)}/24 cases. Failed or missing {len(failures)}.', '',
             'Both arms select checkpoints using clipped source-validation MAE. '
             'The unclipped arm removes the zero floor only from the source suffix training loss.', '']
    if not failures:
        lines += ['| Development domain | Model | Local MAE | Clipped training MAE | Unclipped training MAE | Improved seeds |',
                  '|---|---|---:|---:|---:|---:|']
        for domain in plan['recipe']['development_targets']:
            for family in ('B03', 'B04'):
                rows = [r for r in records if r['dataset'] == domain and r['family'] == family]
                by = {(r['seed'], r['arm']): r for r in rows}
                wins = sum(by[seed, 'unclipped']['test_MAE'] < by[seed, 'clipped']['test_MAE']
                           for seed in plan['recipe']['seeds'])
                avg = lambda arm: mean(r['test_MAE'] for r in rows if r['arm'] == arm)
                lines.append(f"| {domain} | {family} | {mean(r['local_MAE'] for r in rows):.3f} | "
                             f"{avg('clipped'):.3f} | {avg('unclipped'):.3f} | {wins}/3 |")
    else:
        lines += [f'- Task {idx}: {reason}' for idx, reason in failures.items()]
    lines += ['', 'Development experiment after exposed negative baseline results. '
              'Confirmatory method evaluation and independent S01 sign-off remain pending.', '']
    (report/'SUMMARY.md').write_text('\n'.join(lines))
    return {'scientific_status': 'SOURCE_LOSS_DEVELOPMENT_COMPLETE' if not failures else 'SOURCE_LOSS_DEVELOPMENT_INCOMPLETE',
            'diagnostic_cases_checked': len(records), 'failed': failures}


def run(project, code, report, stage):
    plan = foundation(project, code)
    contract, _ = load_contract(code)
    root = store_path(project, plan)
    progress_state = {'stage': stage}
    def progress(**values):
        progress_state.update(values)
        write(report/'transfer_progress.json', progress_state)
        print('LOSS_PROGRESS', progress_state, flush=True)
    if stage == REPORT:
        result = collection(project, code, report, plan)
    else:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        task = (plan['probes'] if stage == PROBE else plan['tasks'])[index]
        case = root/('probes' if stage == PROBE else 'cases')/str(index)
        case.mkdir(parents=True, exist_ok=True)
        with (case/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if completed(case, plan['identity']):
                result = {**read(case/'result.json'), 'new_optimizer_steps': 0}
            elif stage == PROBE:
                result = train_source(project, code, report, case, task, plan, contract, progress, probe=True)
                write(case/'result.json', result)
                finish(case, plan['identity'])
            else:
                for i in range(len(plan['probes'])):
                    if not completed(root/'probes'/str(i), plan['identity']):
                        raise ValueError('Loss trainer recovery probe is missing')
                source_file = case/'source_result.json'
                if source_file.exists():
                    source = dict(read(source_file), new_optimizer_steps=0)
                    if digest(case/'best.pt') != source['best_sha256'] or source['task'] != task:
                        raise ValueError('Saved source completion differs')
                else:
                    source = train_source(project, code, report, case, task, plan, contract, progress)
                    write(source_file, source)
                target = target_fit(project, code, report, case, task, plan, contract, source, progress)
                result = {'task': task, 'source': source, 'target': target,
                          'new_optimizer_steps': source['new_optimizer_steps']+target['new_optimizer_steps'],
                          'scientific_status': 'SOURCE_LOSS_DEVELOPMENT_CASE_COMPLETE'}
                write(case/'result.json', result)
                finish(case, plan['identity'])
        if result['task'] != task:
            raise ValueError('Completed task does not match the requested loss condition')
        (report/'SUMMARY.md').write_text(f'# Development loss task\n\n{task}\n\nCompleted.\n')
    write(report/'transfer_result.json', {**result, 'identity': plan['identity'], 'store': str(root),
          'method_evaluated': False, 'S01_independent_review': 'PENDING'})
    progress(phase='FAILED' if result.get('failed') else 'COMPLETED')
    return 2 if result.get('failed') else 0


def dispatch(project, code, report, stage):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/source_loss_development.py'),
                            str(project), str(code), str(report), stage],
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    parser.add_argument('stage', choices=STAGES)
    args = parser.parse_args()
    try:
        if not os.environ.get('SLURM_JOB_ID', '').isdigit():
            raise ValueError('Use the Slurm launcher')
        sys.exit(run(args.project.resolve(), inside(args.project, args.code), inside(args.project, args.report), args.stage))
    except Exception as exc:
        traceback.print_exc()
        (args.report/'failure_traceback.txt').write_text(traceback.format_exc())
        (args.report/'SUMMARY.md').write_text(f'# Development loss experiment failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
