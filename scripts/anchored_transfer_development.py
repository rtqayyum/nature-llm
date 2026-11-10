"""Matched persistence-anchor development experiment and independent review inputs."""
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
import source_loss_development as predecessor

LAUNCH = 'shared-transfer-anchor-development'
REVIEW = 'shared-transfer-anchor-review'
PROBE = 'shared-transfer-anchor-probe'
FIT = 'shared-transfer-anchor-fit'
REPORT = 'shared-transfer-anchor-report'
STAGES = (REVIEW, PROBE, FIT, REPORT)
GPU_STAGES = (PROBE, FIT)
JOURNAL = 'anchor_submissions'
JOB_PREFIX = 'ectt-anchor'


def submission_steps(plan):
    return [('review', REVIEW, JOB_PREFIX+'-review', None),
            ('probe', PROBE, JOB_PREFIX+'-probe', plan['probes']),
            ('fit', FIT, JOB_PREFIX+'-fit', plan['tasks']),
            ('report', REPORT, JOB_PREFIX+'-report', None)]


def matrix(code):
    parent = predecessor.matrix(code)
    recipe = read(code/'config/anchored_transfer_development_v1.json')
    if (recipe['arms'] != ['direct_zero', 'persistence_residual']
            or recipe['development_targets'] != parent['recipe']['development_targets']
            or recipe['seeds'] != parent['recipe']['seeds'] or recipe['budget'] != parent['recipe']['budget']
            or recipe['maximum_concurrent_gpu_tasks'] != 4 or recipe['source_loss'] != 'unclipped'
            or recipe['predecessor_report_job'] != '1116448'):
        raise ValueError('Unexpected anchored development matrix')
    tasks = []
    for row in parent['tasks']:
        if row['arm'] == 'unclipped':
            for arm in recipe['arms']:
                tasks.append({**row, 'index': len(tasks), 'predecessor_index': row['index'], 'arm': arm})
    probes = [next(row for row in tasks if row['family'] == family and row['arm'] == arm
                   and row['dataset'] == 'pems08' and row['seed'] == 11)
              for family in ('B03', 'B04') for arm in recipe['arms']]
    paths = ['config/anchored_transfer_development_v1.json',
             'scripts/anchored_transfer_development.py', 'scripts/anchored_transfer_model.py',
             'scripts/statistical_audit.py', 'scripts/ectt_statistics.py',
             'scripts/frozen_verifier_replay.py', 'research/S01_REVIEW_CHECKLIST.md']
    identity = baseline.fingerprint({'predecessor': parent['identity'],
                                      'implementation': {p: digest(code/p) for p in paths}})
    if len(tasks) != 24 or len(probes) != 4:
        raise ValueError('Unexpected anchor development task counts')
    return {'identity': identity, 'recipe': recipe, 'baseline': parent['baseline'],
            'predecessor': parent, 'tasks': tasks, 'probes': probes}


def foundation(project, code):
    previous = predecessor.foundation(project, code)
    plan = matrix(code)
    job = project/'results/slurm_jobs'/plan['recipe']['predecessor_report_job']
    state, result, receipt = (read(job/name) for name in
                             ('job_status.json', 'report/transfer_result.json', 'submission.json'))
    pinned = project/'work/code_versions/ae9e6bad1e509cb8029be677c0c06570fd0ece87'
    if ((state['stage'], state['execution_state'], state['shell_exit_code']) !=
            (predecessor.REPORT, 'COMPLETED', 0) or Path(receipt['code']).resolve() != pinned.resolve()
            or result.get('scientific_status') != 'SOURCE_LOSS_DEVELOPMENT_COMPLETE'
            or result.get('diagnostic_cases_checked') != 24 or result.get('failed')
            or result.get('identity') != previous['identity']
            or Path(result['store']).resolve() != predecessor.store_path(project, previous)):
        raise ValueError('Verified complete predecessor 1116448 is required')
    return plan


def store_path(project, plan):
    return inside(project, project/'outputs/ectt/anchor_development'/plan['identity'])


def build(project, code, contract, task, report):
    from anchored_transfer_model import AnchoredTransfer
    from transfer_worker import gpu
    model, runtime = gpu(project, code, contract, task, report)
    model = AnchoredTransfer(model, task['arm'])
    runtime.update(condition=task['arm'], output_head_initialization='zero',
                   residual_last_value_feature=True,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, runtime


def train_source(project, code, report, store, task, plan, contract, progress, probe=False):
    import torch
    from source_loss_train import fit
    from transfer_train import adapt, episode_bank, forecast, invariant_probe, load, equal_state
    from transfer_worker import local_recipes, raw_arrays, raw_paths

    recipe = plan['baseline']['recipe']
    design = {**contract['training_design'], **({'source_max_epochs': 1} if probe else {})}
    raw = raw_arrays(raw_paths(project, code, contract), task['sources'], contract)
    bank = episode_bank(task, contract, recipe, raw)
    write(store/'episode_ledger.json', {'task': task, 'episodes': bank})
    began = time.monotonic()
    model, runtime = build(project, code, contract, task, report)
    key = baseline.fingerprint({'experiment': plan['identity'], 'task': task, 'bank': bank,
                                'runtime': runtime, 'probe': probe})
    slug = max(task['sources'], key=lambda name: raw[name].shape[0])
    support = bank[slug]['train'][0]['support']
    from anchored_transfer_model import initial_check
    from compact_training import data_batch
    from ectt_data import fit_scaler
    inputs, _ = data_batch(raw[slug], [support[0]+12, support[0]+13], fit_scaler(raw[slug], support), 'cuda')
    initial = initial_check(model, inputs)
    gates_before = invariant_probe(model, raw[slug], support, task['family'], 'cuda')
    args = (raw, task, bank, recipe, design, local_recipes(code), 'cuda')
    cursor, new_steps = fit(model, *args, store/'source', key, progress, 'unclipped')
    if probe:
        full = load(store/'source', key)
        del model
        torch.cuda.empty_cache()
        model, _ = build(project, code, contract, task, report)
        resumed = store/'resumed'
        probe_updates = new_steps
        if not (resumed/'latest.json').exists():
            _, updates = fit(model, *args, resumed, key, progress, 'unclipped', stop_after=2)
            probe_updates += updates
        del model
        torch.cuda.empty_cache()
        model, _ = build(project, code, contract, task, report)
        _, updates = fit(model, *args, resumed, key, progress, 'unclipped')
        probe_updates += updates
        if not equal_state(full, load(resumed, key)):
            raise ValueError('Anchored source recovery differs')
        head_training = {**design, 'target_updates': recipe['recovery_probe_steps']}
        head_args = (model, raw[slug], [support[0]+12, support[1]], fit_scaler(raw[slug], support),
                     task['seed'], recipe, head_training, 'cuda', progress)
        head, head_full = adapt(*head_args, checkpoint=store/'head-full', identity=key)
        prediction = forecast(model, raw[slug], bank[slug]['train'][0]['suffix'],
                              fit_scaler(raw[slug], support), recipe, 'cuda', head)
        head_steps = head_full['new_steps']
        if not (store/'head-resumed/latest.json').exists():
            _, partial = adapt(*head_args, checkpoint=store/'head-resumed', identity=key, stop_after=2)
            head_steps += partial['new_steps']
        head, head_resumed = adapt(*head_args, checkpoint=store/'head-resumed', identity=key)
        head_steps += head_resumed['new_steps']
        replay = forecast(model, raw[slug], bank[slug]['train'][0]['suffix'],
                          fit_scaler(raw[slug], support), recipe, 'cuda', head)
        if (not equal_state(load(store/'head-full', key), load(store/'head-resumed', key))
                or not (prediction == replay).all()):
            raise ValueError('Anchored residual model/Adam/RNG/forecast recovery differs')
        gates_after = invariant_probe(model, raw[slug], support, task['family'], 'cuda', head)
        runtime.update(elapsed_seconds=time.monotonic()-began, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        return {'task': task, 'runtime': runtime, 'scientific_status': 'ANCHOR_RECOVERY_PASSED', 'initial_forecast_check': initial,
                'new_optimizer_steps': probe_updates+head_steps, 'target_recovery_steps': head_training['target_updates'],
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
            'initial_forecast_check': initial,
            'temporary_support_head_updates_in_completed_trajectory': cursor['epoch']*2*len(task['sources'])*design['target_updates'],
            'invariance_before': gates_before, 'invariance_after': gates_after}


def target_fit(project, code, report, store, task, plan, contract, source, progress):
    import numpy as np
    import torch
    from ectt_data import fit_scaler, inverse
    from shared_baselines import independent_scores
    from transfer_train import adapt, invariant_probe, restore_model
    from transfer_attribution import predict_pair
    from transfer_worker import raw_arrays, raw_paths, save_arrays
    import transfer_diagnostics as diagnostics

    budget = plan['recipe']['budget']
    profile = contract['datasets'][task['dataset']]
    support = profile['budgets'][budget]['support_raw']
    raw = raw_arrays(raw_paths(project, code, contract), [task['dataset']], contract)[task['dataset']]
    scaler = fit_scaler(raw, support)
    began = time.monotonic()
    model, runtime = build(project, code, contract, task, report)
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
    persistence = np.asarray(raw[:, slice(profile['test_targets'][0]-1, profile['test_targets'][1]-1)].T, dtype=np.float64)
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
                source_normalized=source_z, adapted_normalized=target_z, persistence=persistence)
    span = scaler['maximum']-scaler['minimum']
    runtime.update(elapsed_seconds=time.monotonic()-began, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    return {'runtime': runtime,
            'source_clipped_fraction': float(np.mean(scaler['minimum']+source_z.astype(np.float64)*span < 0)),
            'adapted_clipped_fraction': float(np.mean(scaler['minimum']+target_z.astype(np.float64)*span < 0)),
            'metrics': {'persistence': independent_scores(persistence, original['truth']),
                        'local': independent_scores(original['local'], original['truth']),
                        'transfer': independent_scores(predictions, original['truth'])},
            'new_optimizer_steps': cursor['new_steps'], 'optimizer_steps': cursor['step'],
            'scaler': scaler, 'support_raw': support, 'test_targets': profile['test_targets'],
            'invariance': gates, 'verification_labels_used': 0}


def review_packet(project, code, report, plan, progress):
    import shutil
    import numpy as np
    from statistical_audit import review
    from frozen_verifier_replay import replay

    review_result = review(report/'internal_checks', code, progress)
    origins = np.arange(80)
    local = np.full((80, 2), 10.)
    transfer = np.full((80, 2), 2.)
    reads = []
    def matured(origin):
        reads.append(origin)
        return np.full(2, 2.)
    accepted = replay(origins, local, transfer, 1., matured)
    if (not accepted['accepted'] or len(reads) >= len(origins)
            or reads != list(range(accepted['verification_origins_read']))):
        raise ValueError('Synthetic consumer failed chronological stopping')
    reads.clear()
    rejected = replay(origins, local, local, 1., matured)
    if not rejected['censored'] or rejected['first_certificate'] != 81 or len(reads) != 80:
        raise ValueError('Synthetic consumer failed nonacceptance accounting')
    files = ['research/S01_REVIEW_CHECKLIST.md', 'research/ECTT_V2_DESIGN.md',
             'research/RELATED_WORK.md', 'config/ectt_protocol_v1.json',
             'scripts/ectt_statistics.py', 'scripts/statistical_audit.py',
             'scripts/frozen_verifier_replay.py', 'tests/test_transfer_parallel.py']
    packet = report/'S01_packet'
    hashes = {}
    for name in files:
        destination = packet/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code/name, destination)
        hashes[name] = digest(destination)
    write(packet/'review_disposition_template.json', {
        'status': 'PENDING_INDEPENDENT_REVIEW', 'reviewer': None, 'date': None,
        'reviewed_hashes': hashes, 'findings': [], 'disposition': None,
        'independent_review_complete': False})
    write(packet/'manifest.json', {'code_files': hashes, 'experiment_identity': plan['identity'],
                                   'traffic_labels_read': 0, 'independent_review_complete': False})
    (report/'SUMMARY.md').write_text(
        '# S01 review preparation\n\nInternal synthetic checks passed. '
        'Derivation, implementation and review template are in S01_packet/.\n'
        'Independent mathematical and deployment-filtration review remain pending. '
        'No traffic verification labels were read.\n')
    return {'scientific_status': 'S01_PACKET_PREPARED_SIGNOFF_PENDING',
            'internal_checks': review_result, 'synthetic_consumer_cases': 2,
            'verification_labels_used': 0, 'new_optimizer_steps': 0}


def collection(project, code, report, plan):
    import numpy as np
    from shared_baselines import independent_scores
    root = store_path(project, plan)
    submission = read(project/'work'/JOURNAL/(plan['identity']+'.json'))
    jobs = submission['jobs']
    records, failures = [], {}
    packet = project/'results/slurm_jobs'/jobs['review']/'report'
    try:
        status = read(packet.parent/'job_status.json')
        result = read(packet/'transfer_result.json')
        if (status['execution_state'] != 'COMPLETED' or status['shell_exit_code'] != 0
                or status['stage'] != REVIEW or result['identity'] != plan['identity']
                or result['scientific_status'] != 'S01_PACKET_PREPARED_SIGNOFF_PENDING'):
            raise ValueError('Review preparation job incomplete')
        manifest = read(packet/'S01_packet/manifest.json')
        for name, sha in manifest['code_files'].items():
            if digest(inside(packet/'S01_packet', packet/'S01_packet'/name)) != sha:
                raise ValueError('Review packet hash differs')
    except Exception as exc:
        failures['review'] = str(exc)
    for task in plan['tasks']:
        label = str(task['index'])
        try:
            job = project/'results/slurm_jobs'/(jobs['fit']+'_'+label)
            status = read(job/'job_status.json')
            if (status['execution_state'], status['shell_exit_code'], status['stage']) != ('COMPLETED', 0, FIT):
                raise ValueError('Anchor task did not complete')
            case = root/'cases'/label
            if not completed(case, plan['identity']):
                raise ValueError('Anchor artifacts are incomplete')
            result = read(case/'result.json')
            if result['task'] != task:
                raise ValueError('Anchor task identity differs')
            with np.load(case/'test_forecasts.npz', allow_pickle=False) as values:
                for name in ('transfer', 'local', 'persistence'):
                    if independent_scores(values[name], values['truth']) != result['target']['metrics'][name]:
                        raise ValueError('Saved independent metrics differ')
                reference = predecessor.store_path(project, plan['predecessor'])/'cases'/str(task['predecessor_index'])
                if not completed(reference, plan['predecessor']['identity']):
                    raise ValueError('Predecessor reference incomplete')
                old = read(reference/'result.json')
                old_task = old['task']
                if (old_task['arm'] != 'unclipped' or any(old_task[k] != task[k] for k in ('family', 'dataset', 'seed'))):
                    raise ValueError('Predecessor reference task differs')
                with np.load(reference/'test_forecasts.npz', allow_pickle=False) as archived:
                    if any(not np.array_equal(values[k], archived[k]) for k in ('truth', 'origins', 'local')):
                        raise ValueError('Predecessor paired cohort differs')
            scores = result['target']['metrics']
            records.append({**{k: task[k] for k in ('family', 'dataset', 'seed', 'arm')},
                'source_epochs': result['source']['epochs'],
                'source_optimizer_steps': result['source']['source_steps'],
                'source_validation_local_error_scaled_MAE': result['source']['best_validation_normalized_MAE'],
                'temporary_source_support_head_updates': result['source']['temporary_support_head_updates_in_completed_trajectory'],
                'target_optimizer_steps': result['target']['optimizer_steps'],
                'test_MAE': scores['transfer']['MAE'], 'local_MAE': scores['local']['MAE'],
                'persistence_MAE': scores['persistence']['MAE'],
                'archived_unclipped_MAE': old['target']['metrics']['transfer']['MAE'],
                'target_source_only_clipped_fraction': result['target']['source_clipped_fraction'],
                'target_adapted_clipped_fraction': result['target']['adapted_clipped_fraction']})
        except Exception as exc:
            failures[label] = str(exc)
    if records:
        with (report/'anchor_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = ['# Persistence anchor development comparison', '',
             f'Verified {len(records)}/24 cases. Failed or missing {len(failures)}.', '',
             'Both new conditions use a zero-initialized output head and the same residual features. '
             'Only the persistence skip connection differs. Source training loss is unclipped.', '']
    if not failures:
        lines += ['| Domain | Model | Selected local MAE | Persistence MAE | Archived unclipped MAE | New direct MAE | Anchored MAE | Improved seeds |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for domain in plan['recipe']['development_targets']:
            for family in ('B03', 'B04'):
                rows = [r for r in records if r['dataset'] == domain and r['family'] == family]
                by = {(r['seed'], r['arm']): r for r in rows}
                avg = lambda key: mean(r[key] for r in rows)
                score = lambda arm: mean(r['test_MAE'] for r in rows if r['arm'] == arm)
                wins = sum(by[seed, 'persistence_residual']['test_MAE'] < by[seed, 'direct_zero']['test_MAE']
                           for seed in plan['recipe']['seeds'])
                lines.append(f"| {domain} | {family} | {avg('local_MAE'):.3f} | {avg('persistence_MAE'):.3f} | "
                             f"{avg('archived_unclipped_MAE'):.3f} | {score('direct_zero'):.3f} | "
                             f"{score('persistence_residual'):.3f} | {wins}/3 |")
    else:
        lines += [f'- {key}: {value}' for key, value in failures.items()]
    lines += ['', 'Archived scores are references, not the fresh matched control. '
              'Development-only results after exposed failures. No verifier decisions or ECTT objective. '
              'Independent S01 sign-off remains pending.', '', 'S01 review packet: '+str(packet/'S01_packet'), '']
    (report/'SUMMARY.md').write_text('\n'.join(lines))
    return {'scientific_status': 'ANCHOR_DEVELOPMENT_COMPLETE' if not failures else 'ANCHOR_DEVELOPMENT_INCOMPLETE',
            'diagnostic_cases_checked': len(records), 'failed': failures, 'review_packet': str(packet/'S01_packet')}


def run(project, code, report, stage):
    plan = foundation(project, code)
    contract, _ = load_contract(code)
    root = store_path(project, plan)
    progress_state = {'stage': stage}
    def progress(**values):
        progress_state.update(values)
        write(report/'transfer_progress.json', progress_state)
        print('ANCHOR_PROGRESS', progress_state, flush=True)
    if stage == REVIEW:
        result = review_packet(project, code, report, plan, progress)
    elif stage == REPORT:
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
                        raise ValueError('Anchor recovery probe is missing')
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
                          'scientific_status': 'ANCHOR_DEVELOPMENT_CASE_COMPLETE'}
                write(case/'result.json', result)
                finish(case, plan['identity'])
        if result['task'] != task:
            raise ValueError('Completed task does not match the requested anchor condition')
        (report/'SUMMARY.md').write_text(f'# Anchor development task\n\n{task}\n\nCompleted.\n')
    write(report/'transfer_result.json', {**result, 'identity': plan['identity'], 'store': str(root),
          'method_evaluated': False, 'S01_independent_review': 'PENDING'})
    progress(phase='FAILED' if result.get('failed') else 'COMPLETED')
    return 2 if result.get('failed') else 0


def dispatch(project, code, report, stage):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/anchored_transfer_development.py'),
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
        (args.report/'SUMMARY.md').write_text(f'# Anchor development experiment failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
