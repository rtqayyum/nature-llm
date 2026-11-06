"""Independent CPU and GPU analyses of the frozen transfer baseline."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
import transfer_attribution as attribution
import transfer_diagnostics as diagnostic
import transfer_plan as plan

CPU = 'shared-transfer-uncertainty'
GPU = 'shared-transfer-gradient-audit'
STAGES = (CPU, GPU)
REPLAY_JOB = '623019'
REPLAY_COMMIT = '29c7819190875086053b8f42721123eb138d7b93'
BOOTSTRAP_SEED = 20261106


def foundation(project, code):
    matrix, inventory = attribution.foundation(project, code)
    job = inside(project, project/'results/slurm_jobs'/REPLAY_JOB)
    status = read(job/'job_status.json')
    result = read(job/'report/transfer_result.json')
    submission = read(job/'submission.json')
    if (status['job_id'], status['stage'], status['execution_state'], status['shell_exit_code']) != (
            REPLAY_JOB, attribution.STAGE, 'COMPLETED', 0):
        raise ValueError('Completed development checkpoint replay is required')
    if (Path(submission['code']).resolve() != inside(project, project/'work/code_versions'/REPLAY_COMMIT)
            or result['scientific_status'] != 'DEVELOPMENT_BEFORE_AFTER_COMPLETE'
            or result['diagnostic_job'] != attribution.DIAGNOSTIC_JOB
            or result['diagnostic_cases_checked'] != 12 or result['new_optimizer_steps'] != 0):
        raise ValueError('Development replay receipt does not match the pinned run')
    return matrix, inventory


def block_means(values, length, replicates, rng):
    """Common circular moving blocks for all columns of one domain."""
    import numpy as np

    n = len(values)
    if values.ndim != 2 or n < 1 or length < 1 or replicates < 1 or not np.isfinite(values).all():
        raise ValueError('Invalid row-block bootstrap inputs')
    length = min(length, n)
    full, remainder = divmod(n, length)
    extended = np.concatenate((values, values[:length-1]))
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(extended, axis=0)))
    starts = rng.integers(0, n, size=(replicates, full+bool(remainder)))
    head = starts[:, :full]
    sums = (prefix[head+length]-prefix[head]).sum(axis=1)
    if remainder:
        tail = starts[:, full]
        sums += prefix[tail+remainder]-prefix[tail]
    return sums/n


def interval(values):
    import numpy as np

    return np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None


def ratios(values):
    import numpy as np

    return np.divide(values[..., 1:], values[..., :1], out=np.full_like(values[..., 1:], np.nan),
                     where=values[..., :1] > 0)


def bootstrap(panel, lengths, replicates):
    import numpy as np

    means = {domain: series.mean(axis=0) for domain, series in panel.items()}
    point = np.asarray([ratios(value) for value in means.values()]).mean(axis=0)
    out = {}
    for length in lengths:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws, domain_results = [], {}
        for domain, series in panel.items():
            sampled = block_means(series, length, replicates, rng)
            relative = ratios(sampled)
            draws.append(relative)
            value = means[domain]
            domain_results[domain] = {'origins': len(series), 'full_blocks': len(series)//length,
                'unstable_descriptive': len(series)//length < 8,
                'arms': {family: {'local_MAE': float(value[0]), 'transfer_MAE': float(value[i+1]),
                                  'excess_MAE': float(value[i+1]-value[0]),
                                  'excess_MAE_95_interval': interval(sampled[:, i+1]-sampled[:, 0]),
                                  'MAE_ratio_95_interval': interval(relative[:, i])}
                         for i, family in enumerate(('B03', 'B04'))}}
        macro = np.asarray(draws).mean(axis=0)
        out[str(length)] = {'domains': domain_results, 'macro': {
            family: {'MAE_ratio': float(point[i]) if np.isfinite(point[i]) else None,
                     'MAE_ratio_95_interval': interval(macro[:, i])}
            for i, family in enumerate(('B03', 'B04'))}}
    return out


def uncertainty(project, code, report, matrix, contract):
    import numpy as np

    store = plan.store_path(project, matrix)
    budget = str(contract['partitions']['primary_budget'])
    domains = contract['confirmatory_targets']
    index = {(t['dataset'], t['seed'], t['family']): t for t in matrix['target']}
    panel = {}
    for slug in domains:
        profile = contract['datasets'][slug]
        if not profile['budgets'][budget]['eligible_for_learned_adaptation']:
            raise ValueError('Primary confirmatory case is ineligible')
        seeds, reference_truth = [], None
        for seed in contract['seeds']:
            paired = []
            for family in ('B03', 'B04'):
                task = index[slug, seed, family]
                directory, artifacts = diagnostic.worker_artifacts(project, task, store, 'target', plan.TARGET)
                path = diagnostic.checked(directory, artifacts, budget+'/test_forecasts.npz')
                paired.append(diagnostic.forecasts(path, profile['test_targets'], profile['shape'][0], include_truth=True))
            a, b = paired
            truth = a['truth']
            if not np.array_equal(truth, b['truth']) or not np.array_equal(a['local'], b['local']):
                raise ValueError('Matched family forecasts differ in local baseline or ground truth')
            if reference_truth is not None and not np.array_equal(reference_truth, truth):
                raise ValueError('Seed-specific test truths differ')
            reference_truth = truth
            predictions = (a['local'], a['transfer'], b['transfer'])
            seeds.append(np.column_stack([np.abs(p-truth).mean(axis=1) for p in predictions]))
        panel[slug] = np.mean(seeds, axis=0)
        write(report/'transfer_progress.json', {'phase': 'READ_TEST_FORECASTS', 'domains_checked': len(panel)})
    statistics = contract['statistics']
    lengths = [statistics['block_steps'], *statistics['sensitivity_blocks']]
    draws = statistics['paired_bootstrap_replicates']
    results = bootstrap(panel, lengths, draws)
    write(report/'baseline_uncertainty.json', {'budget': budget, 'replicates': draws, 'bootstrap_seed': BOOTSTRAP_SEED,
          'method': 'paired circular moving row blocks; fixed seed mean per origin; equal domain weighting',
          'scope': 'descriptive conditional-on-panel baseline uncertainty', 'results': results})
    lines = ['# Paired uncertainty for the unconditional transfer baselines', '',
             '5% budget. Six confirmatory domains. Each interval uses 2,000 paired row-block resamples.',
             'The three training seeds are averaged within each row and domain.', '',
             '| Rows per block | Model | Macro MAE / local | Descriptive 95% interval |',
             '|---:|---|---:|---:|']
    for length, result in results.items():
        for family, metrics in result['macro'].items():
            band = metrics['MAE_ratio_95_interval']
            point = metrics['MAE_ratio']
            text = f'{point:.3f}' if point is not None else 'undefined'
            lines.append(f'| {length} | {family} | {text} | '
                         +(f'{band[0]:.3f} to {band[1]:.3f} |' if band else 'undefined |'))
    lines += ['', 'Domain-level MAE differences and short-series flags are in baseline_uncertainty.json.',
              'These are descriptive comparisons on archived test forecasts. S01 and ECTT evaluation remain pending.', '']
    (report/'SUMMARY.md').write_text('\n'.join(lines))
    return {'scientific_status': 'BASELINE_UNCERTAINTY_COMPLETE', 'diagnostic_cases_checked': len(domains)*len(contract['seeds'])*2}


def gradient_summary(physical, truth, scale):
    import numpy as np

    if (physical.shape != truth.shape or physical.size == 0 or not np.isfinite(physical).all()
            or not np.isfinite(truth).all() or np.any(truth < 0)
            or not math.isfinite(scale) or scale < 1):
        raise ValueError('Invalid source suffix predictions, targets or scale')
    negative = physical < 0
    missed = negative & (truth > 0)
    return {'suffix_values': int(truth.size), 'negative_pre_floor_values': int(negative.sum()),
            'positive_truth_with_blocked_output_gradient': int(missed.sum()),
            'all_pre_floor_negative': bool(negative.all()),
            'normalized_MAE_with_floor': float(np.abs(np.maximum(0, physical)-truth).mean()/scale),
            'normalized_MAE_without_floor': float(np.abs(physical-truth).mean()/scale)}


def source_audit_case(project, code, report, task, matrix, contract, paths):
    import numpy as np
    import torch
    from ectt_data import fit_scaler
    from transfer_train import adapt, restore_model
    from transfer_worker import gpu, raw_arrays

    store = plan.store_path(project, matrix)
    directory, artifacts = diagnostic.worker_artifacts(project, task, store, 'source', plan.SOURCE)
    ledger = read(diagnostic.checked(directory, artifacts, 'episode_ledger.json'))
    result = read(diagnostic.checked(directory, artifacts, 'result.json'))
    history = diagnostic.source_history(result)
    selected = next(row for row in result['validation_history'] if row['epoch'] == history['source_selected_epoch'])
    if ledger['task'] != task or result['task'] != task or history['source_best_sha256'] != artifacts['best.pt']:
        raise ValueError('Source task or checkpoint provenance differs')
    if set(ledger['episodes']) != set(task['sources']) or set(task['sources']) & set(task['excluded']):
        raise ValueError('Source episode ledger differs from the held-out fold')
    raw = raw_arrays(paths, task['sources'], contract)
    model, runtime = gpu(project, code, contract, task, report)
    checkpoint = diagnostic.checked(directory, artifacts, 'best.pt')
    restore_model(model, torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    results = []
    design = contract['training_design']
    recipe = matrix['recipe']
    for slug in task['sources']:
        val = ledger['episodes'][slug]['validation']
        values = raw[slug]
        a, b = val['support']
        start, end = val['suffix']
        train = contract['datasets'][slug]['source_train_raw']
        validation = contract['datasets'][slug]['source_validation_targets']
        if not train[0] <= a < b <= train[1] <= validation[0] <= start < end <= validation[1]:
            raise ValueError('Source support or suffix falls outside its frozen bounds')
        scaler = fit_scaler(values, [a, b])
        scale = read(diagnostic.checked(directory, artifacts, f'fit/scales/{slug}/{a}.json'))['scale']
        residual, fit = adapt(model, values, [a+12, b], scaler, task['seed'], recipe,
                              design, 'cuda', lambda **_: None)
        _, corrected = attribution.predict_pair(model, residual, values, [start, end], scaler,
                                                recipe['origin_microbatch'], 'cuda')
        # Match the original FP32 affine transform, retaining the pre-floor value.
        physical = scaler['minimum']+corrected*(scaler['maximum']-scaler['minimum'])
        metrics = gradient_summary(physical.astype(np.float64), values[:, start:end].T, scale)
        archived = selected['validation_by_source'][slug]
        replay_error = abs(metrics['normalized_MAE_with_floor']-archived)
        if not math.isclose(metrics['normalized_MAE_with_floor'], archived, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f'Source validation replay differs for {slug}: delta {replay_error:.6g}')
        results.append({'source_domain': slug, 'support': [a, b], 'validation_suffix': [start, end],
                        'local_scale': scale, 'archived_validation_MAE': archived,
                        'replay_absolute_delta': replay_error, 'diagnostic_head_updates': fit['new_steps'], **metrics})
    return {'task': task, 'checkpoint_sha256': artifacts['best.pt'], 'runtime': runtime, 'episodes': results}


def gradient_audit(project, code, report, matrix, contract, inventory):
    from transfer_worker import raw_paths

    tasks = [matrix['source'][t['source_index']] for t in matrix['target']
             if t['dataset'] in contract['development_targets']]
    if len(tasks) != 12 or len({t['index'] for t in tasks}) != 12:
        raise ValueError('Unexpected development source checkpoint set')
    paths = raw_paths(project, code, contract)
    identity = plan.fingerprint({'matrix': matrix['identity'], 'inventory': digest(inventory),
                                 'script': digest(Path(__file__)), 'stage': GPU})
    store = inside(project, project/'outputs/ectt/shared_transfer_gradient_audit'/identity)
    results, new_updates = [], 0
    for task in tasks:
        case = inside(store, store/str(task['index']))
        case.mkdir(parents=True, exist_ok=True)
        write(report/'transfer_progress.json', {'phase': 'REPLAY_SOURCE_VALIDATION', 'checkpoints_checked': len(results),
                                                 'expected_checkpoints': len(tasks), 'current_task': task})
        with (case/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if completed(case, identity):
                record = read(case/'result.json')
            else:
                record = source_audit_case(project, code, report, task, matrix, contract, paths)
                write(case/'result.json', record)
                finish(case, identity)
                new_updates += sum(row['diagnostic_head_updates'] for row in record['episodes'])
        if record['task'] != task:
            raise ValueError('Saved audit case belongs to a different checkpoint')
        results.append(record)
    write(report/'source_gradient_audit.json', {'checkpoint_cases': results, 'matrix_identity': matrix['identity'],
          'source_backbone_updates': 0, 'new_diagnostic_head_updates': new_updates})
    lines = ['# Source validation audit of the zero floor', '',
             'Replayed 12 archived source checkpoints for the PEMS08 and Beijing development folds.',
             'Each row counts source-validation suffix values after replaying the original temporary support head fit.',
             'A negative pre-floor forecast has zero derivative through the floor. The last column counts such cases with positive truth.', '',
             '| Held-out development fold | Model | Source suffixes | Entirely below zero | Negative before floor | Positive truth with blocked output gradient |',
             '|---|---|---:|---:|---:|---:|']
    for fold in contract['development_targets']:
        for family in ('B03', 'B04'):
            rows = [r for item in results if item['task']['fold'] == fold and item['task']['family'] == family for r in item['episodes']]
            n = sum(r['suffix_values'] for r in rows)
            below = sum(r['negative_pre_floor_values'] for r in rows)/n
            blocked = sum(r['positive_truth_with_blocked_output_gradient'] for r in rows)/n
            lines.append(f"| {fold} | {family} | {len(rows)} | {sum(r['all_pre_floor_negative'] for r in rows)} | {below:.1%} | {blocked:.1%} |")
    lines += ['', 'These selected-checkpoint replays measure a potential failure mechanism on source validation suffixes.',
              'They do not establish the training history or the performance of an alternative loss.',
              f'Temporary source-support head updates in this allocation: {new_updates}. Source backbone updates: 0.',
              'S01 independent review and verifier scoring remain pending.', '']
    (report/'SUMMARY.md').write_text('\n'.join(lines))
    return {'scientific_status': 'SOURCE_GRADIENT_AUDIT_COMPLETE', 'diagnostic_cases_checked': len(results),
            'source_backbone_updates': 0, 'new_optimizer_steps': new_updates, 'training_performed': new_updates > 0,
            'source_validation_suffixes_checked': sum(len(item['episodes']) for item in results)}


def run(project, code, report, stage):
    matrix, inventory = foundation(project, code)
    contract, _ = load_contract(code)
    result = (uncertainty(project, code, report, matrix, contract) if stage == CPU
              else gradient_audit(project, code, report, matrix, contract, inventory))
    write(report/'transfer_result.json', {**result, 'S01_independent_review': 'PENDING',
          'verification_labels_used_for_scoring': 0, 'method_evaluated': False})
    write(report/'transfer_progress.json', {'phase': 'COMPLETE', 'cases_checked': result['diagnostic_cases_checked']})


def dispatch(project, code, report, stage):
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    if stage == CPU:
        python = inside(project, project/'.venv-ectt-gpu/bin/python')
        env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1')
    else:
        from ectt_env import ensure
        with (project/'work/acquisition.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/transfer_parallel.py'),
                            str(project), str(code), str(report), stage], env=env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    parser.add_argument('stage', choices=STAGES)
    args = parser.parse_args()
    try:
        if not os.environ.get('SLURM_JOB_ID', '').isdigit():
            raise ValueError('Run these analyses inside Slurm')
        run(args.project.resolve(), inside(args.project, args.code), inside(args.project, args.report), args.stage)
    except Exception as exc:
        traceback.print_exc()
        (args.report/'failure_traceback.txt').write_text(traceback.format_exc())
        (args.report/'SUMMARY.md').write_text(f'# Transfer analysis failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
