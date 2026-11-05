"""Slurm workers for data admission, transfer baselines and collection."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
import transfer_plan as plan


def dispatch(project, code, report, stage):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'OPENBLAS_NUM_THREADS': str(os.environ.get('SLURM_CPUS_PER_TASK', 5))}
    return subprocess.call([str(python), '-u', __file__, str(project), str(code), str(report), stage], env=env)


def raw_paths(project, code, contract):
    lock = read(code/'config/sources.lock.json')
    snapshot = project/'work/reference/snapshots'/lock['reference_commit']
    return {slug: inside(project, snapshot/profile['path']) for slug, profile in contract['datasets'].items()}


def preflight(project, code, directory, contract, matrix):
    from ectt_data import dataset_audit
    from qwen_recovery import cached_weights
    evidence, collection = plan.foundation(project, code, full=True)
    lock = read(code/'config/sources.lock.json')
    datasets = {slug: dataset_audit(project, code, directory, slug, profile, lock)
                for slug, profile in contract['datasets'].items()}
    hashes = [value['statistics']['canonical_raw_sha256'] for value in datasets.values()]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate canonical domain arrays')
    _, qwen = cached_weights(project, contract)
    from acquire import verify
    base = inside(project, project/'work/models/gpt2'/lock['gpt2']['revision'])
    if not all(verify(base/row['path'], row) for row in lock['gpt2']['files']):
        raise ValueError('Cached GPT-2 weights failed verification')
    cases = collection['records']
    if len(cases) != 96 or collection['failed_cases'] or any(r['state'] != 'VERIFIED' for r in cases.values()):
        raise ValueError('Local foundation case records are incomplete')
    return {'scientific_status': 'TRANSFER_DATA_READY', 'datasets': list(datasets),
            'qwen_inventory_sha256': qwen, 'local_collector': evidence['collector_job'],
            'matrix_identity': matrix['identity']}


def raw_arrays(paths, slugs, contract):
    import numpy as np
    result = {}
    for slug in slugs:
        if digest(paths[slug]) != contract['datasets'][slug]['container_sha256']:
            raise ValueError('Canonical array hash changed: '+slug)
        value = np.load(paths[slug], mmap_mode='r', allow_pickle=False)
        if list(value.shape) != contract['datasets'][slug]['shape']:
            raise ValueError('Canonical array shape changed: '+slug)
        result[slug] = value[..., 0]
    return result


def gpu(project, code, contract, task, report):
    import torch
    from compact_training import configure
    from transfer_model import create
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expected one allocated CUDA GPU')
    if not torch.cuda.is_bf16_supported() or str(torch.__version__) != '2.4.1+cu118':
        raise ValueError('Pinned CUDA/BF16 environment required')
    configure()
    torch.manual_seed(task['seed'])
    model = create(task['family'], project, code, contract).cuda()
    if any(p.requires_grad and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('All trainable parameters must be FP32')
    runtime = {'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
               'cpus': int(os.environ.get('SLURM_CPUS_PER_TASK', '0')), 'torch': str(torch.__version__),
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'environment_sha256': digest(report/'ectt_environment_freeze.txt'),
               'active_parameters': sum(p.numel() for p in model.parameters()),
               'trainable_parameters': sum(p.numel() for p in model.parameters())}
    if runtime['cpus'] != 5:
        raise ValueError('Expected five allocated CPUs')
    return model, runtime


def require(directory, matrix):
    if not completed(directory, matrix['identity']):
        raise ValueError('Required stage artifact is missing: '+str(directory))
    return read(directory/'result.json')


def local_recipes(code):
    return read(code/'config/shared_baselines_v1.json'), read(code/'config/compact_local_v1.json')


def probe(project, code, report, directory, task, contract, matrix, paths, progress):
    import torch
    from ectt_data import fit_scaler
    from transfer_train import (adapt, episode_bank, equal_state, forecast,
                                invariant_probe, load, source_fit)
    r = matrix['recipe']
    training = {**contract['training_design'], 'source_max_epochs': 1}
    family = task['family']
    raw = raw_arrays(paths, task['sources'], contract)
    bank = episode_bank(task, contract, r, raw)
    slug = max(task['sources'], key=lambda name: raw[name].shape[0])
    support = bank[slug]['train'][0]['support']
    pre_support = fit_scaler(raw[slug], support)
    source_key = plan.fingerprint({'task': task, 'identity': matrix['identity'], 'probe': 'source'})
    head_key = plan.fingerprint({'source': source_key, 'probe': 'adaptation'})
    model, runtime = gpu(project, code, contract, task, report)
    # Include runtime fingerprints in checkpoints. Cross-architecture restarts
    # require separate admission and cannot silently continue a trajectory.
    source_key = plan.fingerprint({'key': source_key, 'runtime': runtime})
    head_key = plan.fingerprint({'key': head_key, 'runtime': runtime})
    pre = invariant_probe(model, raw[slug], support, family, 'cuda')
    start = time.monotonic()
    recipes = local_recipes(code)
    source_fit(model, raw, task, bank, r, training, recipes, 'cuda',
               directory/'source-full', source_key, progress)
    timing = directory/'source_probe_time.json'
    if not timing.exists():
        write(timing, {'seconds': time.monotonic()-start})
    full_seconds = read(timing)['seconds']
    expected_source = load(directory/'source-full', source_key)
    del model
    torch.cuda.empty_cache()
    model, _ = gpu(project, code, contract, task, report)
    source_resume = directory/'source-resumed'
    if not (source_resume/'latest.json').exists():
        source_fit(model, raw, task, bank, r, training, recipes, 'cuda', source_resume,
                   source_key, progress, stop_after=r['recovery_probe_interrupt_step'])
    del model
    torch.cuda.empty_cache()
    model, _ = gpu(project, code, contract, task, report)
    source_fit(model, raw, task, bank, r, training, recipes, 'cuda', source_resume, source_key, progress)
    if not equal_state(expected_source, load(source_resume, source_key)):
        raise ValueError('Source episode optimizer/RNG/cursor recovery differs')
    del expected_source

    bounds = [support[0]+12, support[1]]
    suffix = bank[slug]['train'][0]['suffix']
    head_training = {**training, 'target_updates': r['recovery_probe_steps']}
    args = (model, raw[slug], bounds, pre_support, task['seed'], r, head_training, 'cuda', progress)
    head, _ = adapt(*args, checkpoint=directory/'head-full', identity=head_key)
    expected = forecast(model, raw[slug], suffix, pre_support, r, 'cuda', head)
    resumed = directory/'head-resumed'
    if not (resumed/'latest.json').exists():
        adapt(*args, checkpoint=resumed, identity=head_key, stop_after=r['recovery_probe_interrupt_step'])
    head, _ = adapt(*args, checkpoint=resumed, identity=head_key)
    if not equal_state(load(directory/'head-full', head_key), load(resumed, head_key)):
        raise ValueError('Residual optimizer/RNG/cursor recovery differs')
    actual = forecast(model, raw[slug], suffix, pre_support, r, 'cuda', head)
    if not (expected == actual).all():
        raise ValueError('Recovered residual forecasts differ')
    post = invariant_probe(model, raw[slug], support, family, 'cuda', head)
    estimated = math.ceil(full_seconds*contract['training_design']['source_max_epochs']*1.5)
    limit = int(os.environ['TRAFFIC_SOURCE_TIME_SECONDS'])
    if estimated > limit:
        raise ValueError(f'Projected source walltime {estimated}s exceeds queued limit {limit}s')
    runtime.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   source_probe_epoch_seconds=full_seconds, conservative_source_seconds=estimated)
    return {'scientific_status': 'TRANSFER_RECOVERY_PASSED', 'source_cursor_recovery': True,
            'target_cursor_recovery': True, 'probe_training': training, 'source_domains': task['sources'],
            'invariance_before': pre, 'invariance_after': post, 'runtime': runtime, 'profiled_domain': slug}


def fit_source(project, code, report, directory, task, contract, matrix, paths, progress):
    import torch
    from transfer_train import episode_bank, invariant_probe, restore_model, source_fit
    recipe = matrix['recipe']
    raw = raw_arrays(paths, task['sources'], contract)
    bank = episode_bank(task, contract, recipe, raw)
    write(directory/'episode_ledger.json', {'task': task, 'episodes': bank,
          'source_scaler_policy': 'Pseudo-target support rows only, within each source training range',
          'verification_targets_used': 0, 'held_out_domains_read': [], 'nodes': {k: v.shape[0] for k, v in raw.items()}})
    model, runtime = gpu(project, code, contract, task, report)
    started = time.monotonic()
    slug = max(task['sources'], key=lambda name: raw[name].shape[0])
    probe_rows = bank[slug]['train'][0]['support']
    gates_before = invariant_probe(model, raw[slug], probe_rows, task['family'], 'cuda')
    identity = plan.fingerprint({'matrix': matrix['identity'], 'task': task, 'bank': bank, 'runtime': runtime})
    cursor, new_steps = source_fit(model, raw, task, bank, recipe, contract['training_design'],
                                   local_recipes(code), 'cuda', directory/'fit', identity, progress)
    if cursor['best_model'] is None or not cursor['history']:
        raise ValueError('No source validation-selected checkpoint')
    restore_model(model, cursor['best_model'])
    gates_after = invariant_probe(model, raw[slug], probe_rows, task['family'], 'cuda')
    torch.save(cursor['best_model'], directory/'best.tmp')
    (directory/'best.tmp').replace(directory/'best.pt')
    runtime.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   elapsed_this_job_seconds=time.monotonic()-started)
    return {'scientific_status': 'SOURCE_FIT_COMPLETE', 'task': task, 'runtime': runtime,
            'optimizer_steps': cursor['step'], 'new_optimizer_steps': new_steps,
            'epochs': cursor['epoch'], 'validation_history': cursor['history'],
            'training_loss_history': cursor['losses'], 'objective_gamma': 0,
            'best_validation_normalized_MAE': cursor['best'], 'best_sha256': digest(directory/'best.pt'),
            'invariance_before': gates_before, 'invariance_after': gates_after}


def local_case(project, collection, task, budget):
    row = collection['records'][f"{task['dataset']}/{budget}/{task['seed']}"]
    directory = inside(project, Path(row['artifact_store']))
    if row['state'] != 'VERIFIED' or not completed(directory, row['artifact_identity']):
        raise ValueError('Saved local baseline case failed verification')
    return row, directory


def local_verification(project, code, raw, row, directory, bounds):
    import shared_baselines as local
    family = row['selected_family']
    from compact_baselines import foundation
    from compact_model import CompactTrafficModel
    from compact_training import load_checkpoint, predict
    from shared_baselines import predict as predict_local
    kernel = directory/'kernel' if row['kind'] == 'GPU_FIT' else directory
    selection = read(kernel/'selection.json')
    if selection['selected_family'] != family or not selection['frozen_before_test']:
        raise ValueError('Saved local selection differs')
    if family == 'compact':
        result = read(kernel/'result.json')
        info = read(kernel/'refit/fit.json')
        payload = load_checkpoint(kernel/'refit', info['identity'])
        model = CompactTrafficModel(**result['compact_configuration']).cuda()
        model.load_state_dict(payload['model'], strict=True)
        return selection, predict(model, raw, bounds, selection['final_scaler'], 8, 'cuda')
    if family == 'ridge':
        evidence = foundation(project, code)
        prior = inside(project, Path(evidence['store'])/'cases'/row['dataset']/row['budget'])
        model = read(prior/'fit.json')['models']['ridge']
        if model['scaler'] != selection['final_scaler']:
            raise ValueError('Saved ridge scaler differs')
    elif family == 'persistence':
        model = {'family': 'persistence'}
    else:
        raise ValueError('Unknown saved local family')
    return selection, predict_local(raw, bounds, model, local.config(code)['origin_batch'])


def save_arrays(path, **arrays):
    import numpy as np
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def fit_target(project, code, report, directory, task, contract, matrix, paths, root, progress):
    import numpy as np
    import torch
    from ectt_data import fit_scaler
    from shared_baselines import scores
    from transfer_train import adapt, forecast, invariant_probe, restore_model
    recipe = matrix['recipe']
    _, collection = plan.foundation(project, code)
    source = root/'source'/str(task['source_index'])
    src = require(source, matrix)
    slug = task['dataset']
    if (src['task'] != matrix['source'][task['source_index']] or slug not in src['task']['excluded']
            or slug in src['task']['sources'] or src['task']['seed'] != task['seed']
            or src['task']['family'] != task['family']):
        raise ValueError('Source checkpoint does not exclude the full target domain')
    p = contract['datasets'][slug]
    raw = raw_arrays(paths, [slug], contract)[slug]
    model, runtime = gpu(project, code, contract, task, report)
    restore_model(model, torch.load(source/'best.pt', map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    records, new_steps = {}, 0
    started = time.monotonic()
    for budget in task['budgets']:
        case = directory/budget
        case.mkdir(parents=True, exist_ok=True)
        if completed(case, matrix['identity']):
            records[budget] = read(case/'result.json')
            continue
        bounds = p['budgets'][budget]
        row, local_dir = local_case(project, collection, task, budget)
        origins = np.arange(*p['test_targets'], dtype=np.int64)
        local_file = inside(local_dir, local_dir/row['forecast_file'])
        eligible = bounds['eligible_for_learned_adaptation']
        steps = 0
        manifest = {'source_checkpoint_sha256': src['best_sha256'],
                    'local_completion_sha256': digest(local_dir/'COMPLETE.json'),
                    'local_forecast_sha256': digest(local_file), 'test_targets': p['test_targets'],
                    'verification_labels_used': 0, 'test_labels_used_for_fitting': 0,
                    'support_raw': bounds['support_raw'] if eligible else None,
                    'unique_support_values': (bounds['support_raw'][1]-bounds['support_raw'][0])*raw.shape[0] if eligible else 0,
                    'local_family': row['selected_family'], 'fallback': not eligible,
                    'test_history_role': 'rolling observed histories, with frozen parameters'}
        if eligible:
            scaler = fit_scaler(raw, bounds['support_raw'])
            key = plan.fingerprint({'task': task, 'budget': budget, 'scaler': scaler, 'runtime': runtime,
                                    'source': src['best_sha256'], 'matrix': matrix['identity']})
            residual, cursor = adapt(model, raw, bounds['support_target_rows'], scaler, task['seed'],
                                     recipe, contract['training_design'], 'cuda',
                                     lambda **v: progress(budget=budget, **v), case/'fit', key)
            steps = cursor['step']
            new_steps += cursor['new_steps']
            verification = bounds['verification_target_rows']
            selection, local_values = local_verification(project, code, raw, row, local_dir, verification)
            if selection['final_scaler'] != scaler:
                raise ValueError('Transfer support scaler differs from the local foundation')
            manifest.update(scaler=scaler, support_scale=selection['support_scale'],
                            verification_targets=verification,
                            verification_history_union=[verification[0]-12, verification[1]-1],
                            target_residual_sha256=read(case/'fit/latest.json')['sha256'],
                            invariance=invariant_probe(model, raw, bounds['support_raw'], task['family'], 'cuda', residual))
            write(case/'frozen_candidates.json', manifest)
            # No verification truth is collected here. The ordered candidate
            # histories may include earlier observed verification rows.
            value = forecast(model, raw, verification, scaler, recipe, 'cuda', residual)
            save_arrays(case/'verification_candidates.npz', origins=np.arange(*verification),
                        local=local_values, transfer=value)
            progress(phase='FROZEN_TEST_FORECASTS', budget=budget, optimizer_step=steps)
            prediction = forecast(model, raw, p['test_targets'], scaler, recipe, 'cuda', residual)
        else:
            write(case/'frozen_candidates.json', manifest)
        truth = np.asarray(raw[:, slice(*p['test_targets'])].T, dtype=np.float64)
        with np.load(local_file, allow_pickle=False) as saved:
            if not np.array_equal(origins, saved['origins']) or not np.array_equal(truth, saved['truth']):
                raise ValueError('Local transfer evaluation cohort differs')
            fallback = saved[row['forecast_columns']['support_selected_local']]
        if not eligible:
            prediction = fallback
        save_arrays(case/'test_forecasts.npz', origins=origins, truth=truth, local=fallback, transfer=prediction)
        record = {'budget': budget, 'eligible': eligible, 'optimizer_steps': steps,
                  'local_family': row['selected_family'], 'metrics': {
                      'local': scores(fallback, truth), 'transfer': scores(prediction, truth)}}
        write(case/'result.json', record)
        finish(case, matrix['identity'])
        records[budget] = record
    runtime.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   elapsed_this_job_seconds=time.monotonic()-started)
    return {'scientific_status': 'UNCONDITIONAL_TRANSFER_BASELINE_COMPLETE', 'task': task,
            'records': records, 'runtime': runtime, 'new_optimizer_steps': new_steps,
            'optimizer_steps': sum(r['optimizer_steps'] for r in records.values())}


def collect(project, code, report, root, matrix, contract, progress):
    import numpy as np
    from shared_baselines import independent_scores
    ids = json.loads(os.environ['TRAFFIC_TRANSFER_DEPENDENCIES'])
    records, failures, summary = [], {}, {}
    details, missing_receipts = [], {}
    _, collection = plan.foundation(project, code)

    def receipt(stage, job):
        status_path = job/'job_status.json'
        status = read(status_path) if status_path.exists() else {}
        if status and status.get('stage') != stage:
            raise ValueError('Worker receipt stage differs')
        if status.get('execution_state') != 'COMPLETED' or status.get('shell_exit_code') != 0:
            progress_path = job/'report/transfer_progress.json'
            error = read(progress_path).get('error') if progress_path.exists() else None
            if error:
                raise ValueError(error)
            if not status:
                return None
            raise ValueError(f"Worker status {status['execution_state']} (exit {status['shell_exit_code']})")
        result = read(job/'report/transfer_result.json')
        if result['identity'] != matrix['identity']:
            raise ValueError('Worker receipt identity differs')
        return result

    for stage, label, tasks in ((plan.PREFLIGHT, 'data', [None]), (plan.PROBE, 'probe', matrix['probe']),
                                (plan.SOURCE, 'source', matrix['source']), (plan.TARGET, 'target', matrix['target'])):
        job_id = ids[label]
        if not job_id.isdigit():
            raise ValueError('Invalid parent Slurm ID')
        count = 0
        for idx, task in enumerate(tasks):
            key = label if task is None else f'{label}/{idx}'
            task_id = job_id if task is None else f'{job_id}_{idx}'
            try:
                directory = root/key
                job = receipt(stage, project/'results/slurm_jobs'/task_id)
                if job is None:
                    failures[key] = f'No worker status for {task_id}'
                    missing_receipts[label] = missing_receipts.get(label, 0)+1
                    continue
                require(directory, matrix)
                if job['store'] != str(directory) or job['artifact_sha256'] != digest(directory/'COMPLETE.json'):
                    raise ValueError('Worker receipt differs from the completed artifact')
                if label == 'target':
                    scored = []
                    for budget in task['budgets']:
                        row, local = local_case(project, collection, task, budget)
                        case = directory/budget
                        saved = require(case, matrix)
                        with np.load(case/'test_forecasts.npz', allow_pickle=False) as fc, np.load(
                                inside(local, local/row['forecast_file']), allow_pickle=False) as old:
                            for col in ('origins', 'truth'):
                                if not np.array_equal(fc[col], old[col]):
                                    raise ValueError('Paired origin/truth mismatch')
                            if not np.array_equal(fc['local'], old[row['forecast_columns']['support_selected_local']]):
                                raise ValueError('Paired local forecast mismatch')
                            metrics = {name: independent_scores(fc[name], fc['truth']) for name in ('transfer', 'local')}
                            for name, scores in metrics.items():
                                for metric, value in scores.items():
                                    reference = saved['metrics'][name][metric]
                                    if (value is None) != (reference is None) or (value is not None and not math.isclose(
                                            value, reference, abs_tol=1e-10, rel_tol=1e-10)):
                                        raise ValueError('Independent saved forecast metric mismatch')
                        manifest = read(case/'frozen_candidates.json')
                        if manifest['verification_labels_used'] or manifest['test_labels_used_for_fitting']:
                            raise ValueError('Candidate fit/label timing failed')
                        scored.append({'family': task['family'], 'dataset': task['dataset'], 'seed': task['seed'],
                                       'budget': budget, 'eligible': saved['eligible'],
                                       'local_family': saved['local_family'], 'optimizer_steps': saved['optimizer_steps'],
                                       **{f'{kind}_{metric}': val for kind, values in metrics.items() for metric, val in values.items()}})
                    records.extend(scored)
                count += 1
            except Exception as exc:
                failures[key] = str(exc)
                details.append(f'- {key} (job `{task_id}`) — {exc}')
        summary[label] = {'verified': count, 'expected': len(tasks)}
        progress(phase='COLLECTING', verified=summary, failed=len(failures))
    with (report/'transfer_metrics.csv').open('w', newline='') as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    expected = sum(len(t['budgets']) for t in matrix['target'])
    lines = ['# Shared transfer baseline collection', '',
             f'Verified {len(records)}/{expected} dataset-budget-seed records. '
             f'Verified {summary["source"]["verified"]}/{summary["source"]["expected"]} source fits.', '',
             'B03 uses two active, fully trainable GPT-2 blocks. B04 uses frozen Qwen with trainable LoRA and numerical modules.',
             'Each learned target fits a fresh residual head on support only. Zero or infeasible budgets use persistence.',
             'These are unconditional transfer scores. Frozen candidate files are saved for separately reviewed verification.',
             'S01 independent review and the ECTT objective remain pending.', '']
    if not failures:
        lines += ['## Primary 5% budget', '',
                  'Mean test MAE across the three paired seeds in each domain.', '',
                  '| Panel | Dataset | Local | B03 LLM-UTP | B04 Qwen |',
                  '|---|---|---:|---:|---:|']
        for slug in contract['datasets']:
            panel = 'Development' if slug in contract['development_targets'] else 'Confirmatory'
            rows = [r for r in records if r['dataset'] == slug and r['budget'] == '0.05']
            means = [sum(r[col] for r in rows if r['family'] == family)/len(contract['seeds'])
                     for family, col in (('B03', 'local_MAE'), ('B03', 'transfer_MAE'), ('B04', 'transfer_MAE'))]
            lines.append(f'| {panel} | {slug} | {means[0]:.6f} | {means[1]:.6f} | {means[2]:.6f} |')
    else:
        lines += ['## Failed or missing tasks', ''] + details
        lines += [f'- {label} (job `{ids[label]}`) — {count} tasks with no saved worker status. Check Slurm accounting.'
                  for label, count in missing_receipts.items()]
    lines += ['', 'Source arrays and jobs: '+', '.join(f'{k} {v}' for k, v in ids.items()),
              'The full per-seed scores are in transfer_metrics.csv.', '']
    (report/'SUMMARY.md').write_text('\n'.join(lines))
    return {'scientific_status': 'UNCONDITIONAL_TRANSFER_BASELINES_COMPLETE' if not failures else 'TRANSFER_INCOMPLETE',
            'verified': summary, 'benchmark_experiments_completed': len(records),
            'failed': failures, 'expected_records': expected, 'source_jobs': ids}


def run(project, code, report, stage):
    project = project.resolve()
    code, report = inside(project, code), inside(project, report)
    matrix = plan.plan(code)
    contract, _ = load_contract(code)
    recipe = matrix['recipe']
    root = plan.store_path(project, matrix)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    stage_key = {plan.PREFLIGHT: 'data', plan.PROBE: 'probe', plan.SOURCE: 'source',
                 plan.TARGET: 'target', plan.REPORT: 'report'}[stage]
    task = matrix[stage_key][index] if stage in plan.GPU_STAGES else None
    directory = inside(root, root/stage_key/str(index) if task is not None else root/stage_key)
    if stage == plan.REPORT:
        directory = report
    directory.mkdir(parents=True, exist_ok=True)
    current = {'stage': stage, 'identity': matrix['identity'], 'task': task, 'store': str(directory)}

    def progress(**fields):
        current.update(fields)
        write(report/'transfer_progress.json', current)
        print('TRANSFER_PROGRESS', json.dumps(current), flush=True)

    progress(phase='PREPARING')
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reused = stage != plan.REPORT and completed(directory, matrix['identity'])
        if reused:
            result = read(directory/'result.json')
        elif stage == plan.PREFLIGHT:
            result = preflight(project, code, directory, contract, matrix)
        elif stage == plan.REPORT:
            result = collect(project, code, report, root, matrix, contract, progress)
        else:
            # Check the data receipt without rehashing all canonical arrays in
            # each GPU task. Each source worker opens only its eligible domains.
            admission = read(root/'data/COMPLETE.json')
            if admission['identity'] != matrix['identity'] or admission['artifacts']['result.json'] != digest(root/'data/result.json'):
                raise ValueError('Invalid data admission receipt')
            paths = raw_paths(project, code, contract)
            if stage == plan.PROBE:
                result = probe(project, code, report, directory, task, contract, matrix, paths, progress)
            else:
                probe_idx = recipe['families'].index(task['family'])
                gate = root/'probe'/str(probe_idx)
                marker = read(gate/'COMPLETE.json')
                result_path = gate/'result.json'
                if (marker['identity'] != matrix['identity'] or marker['artifacts']['result.json'] != digest(result_path)
                        or read(result_path)['scientific_status'] != 'TRANSFER_RECOVERY_PASSED'):
                    raise ValueError('Required episode/adaptation recovery probe did not pass')
                if stage == plan.SOURCE:
                    result = fit_source(project, code, report, directory, task, contract, matrix, paths, progress)
                else:
                    result = fit_target(project, code, report, directory, task, contract, matrix, paths, root, progress)
        if not reused and stage != plan.REPORT:
            write(directory/'result.json', result)
            finish(directory, matrix['identity'])
        output = {**result, 'identity': matrix['identity'], 'stage': stage, 'store': str(directory),
                  'reused': reused, 'S01_independent_review': 'PENDING', 'method_evaluated': False}
        if stage != plan.REPORT:
            output['artifact_sha256'] = digest(directory/'COMPLETE.json')
            (report/'SUMMARY.md').write_text(f"# {stage}\n\n{result['scientific_status']}\n\nArtifacts: {directory}\n")
        if reused:
            output['new_optimizer_steps'] = 0
        write(report/'transfer_result.json', output)
        failed = bool(result.get('failed'))
        progress(phase='FAILED' if failed else 'COMPLETED')
        return 2 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    parser.add_argument('stage', choices=plan.STAGES)
    args = parser.parse_args()
    try:
        sys.exit(run(args.project, args.code, args.report, args.stage))
    except Exception as exc:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        (args.report/'failure_traceback.txt').write_text(error)
        (args.report/'SUMMARY.md').write_text(f'# Shared transfer stage failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'stage': args.stage, 'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
