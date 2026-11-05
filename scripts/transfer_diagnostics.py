"""Read-only diagnostics and S01 input inventory for the completed transfer grid."""
import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import traceback

from ectt_common import digest, inside, load_contract, read, write
import transfer_plan as plan

STAGE = 'shared-transfer-diagnose'
COLLECTION = '621882'
JOBS = {'data': '621878', 'probe': '621879', 'source': '621880', 'target': '621881'}
PRODUCER = 'fdc8c3bdb34a2aa8d0769ba87b02813f4c898cdd'
MATRIX = '9791dd1998b3140780ada24a6779d335a9ab60a623671161084e94caf9bfa348'
METRICS = ('MAE', 'RMSE', 'WMAPE_percent')


def receipt(project, job_id, stage):
    job = inside(project, project/'results/slurm_jobs'/job_id)
    status = read(job/'job_status.json')
    result = read(job/'report/transfer_result.json')
    if (status['job_id'], status['stage'], status['execution_state'], status['shell_exit_code']) != (
            job_id, stage, 'COMPLETED', 0):
        raise ValueError('Unsuccessful or mismatched transfer job '+job_id)
    if result['identity'] != MATRIX or result['stage'] != stage:
        raise ValueError('Unexpected artifact identity for '+job_id)
    return job, result


def foundation(project, code):
    matrix = plan.plan(code)
    if matrix['identity'] != MATRIX:
        raise ValueError('Diagnostic code does not match the completed transfer recipe')
    job, result = receipt(project, COLLECTION, plan.REPORT)
    submission = read(job/'submission.json')
    producer = inside(project, project/'work/code_versions'/PRODUCER)
    if (submission['project'] != str(project) or Path(submission['code']).resolve() != producer
            or result['source_jobs'] != JOBS or result['benchmark_experiments_completed'] != 192
            or result['scientific_status'] != 'UNCONDITIONAL_TRANSFER_BASELINES_COMPLETE'
            or result['failed'] or result['verified']['source'] != {'verified': 42, 'expected': 42}):
        raise ValueError('Required completed collection does not match the admitted grid')
    return matrix, job/'report/transfer_metrics.csv'


def worker_artifacts(project, task, store, phase, stage):
    job_id = f"{JOBS[phase]}_{task['index']}"
    directory = inside(store, store/phase/str(task['index']))
    _, result = receipt(project, job_id, stage)
    marker = directory/'COMPLETE.json'
    if result['task'] != task or Path(result['store']) != directory or result['artifact_sha256'] != digest(marker):
        raise ValueError('Worker receipt differs from completion marker for '+job_id)
    manifest = read(marker)
    if manifest['identity'] != MATRIX or manifest['state'] != 'COMPLETED':
        raise ValueError('Invalid worker completion marker for '+job_id)
    return directory, manifest['artifacts']


def checked(directory, artifacts, name):
    path = inside(directory, directory/name)
    if digest(path) != artifacts[name]:
        raise ValueError('Changed artifact '+str(path))
    return path


def source_history(result):
    history = result['validation_history']
    values = [row['validation_normalized_MAE'] for row in history]
    if (not values or len(values) != result['epochs'] or result['objective_gamma'] != 0
            or any(not math.isfinite(value) or value < 0 for value in values)
            or min(values) != result['best_validation_normalized_MAE']):
        raise ValueError('Invalid source validation history')
    return {'source_best_sha256': result['best_sha256'], 'source_epochs': len(values),
            'source_updates': result['optimizer_steps'],
            'source_selected_epoch': history[values.index(min(values))]['epoch'],
            'source_initial_validation': values[0], 'source_best_validation': min(values),
            'source_final_validation': values[-1]}


def adaptation_history(directory, artifacts, budget, manifest, expected):
    import torch

    fit = budget+'/fit/'
    pointer = read(checked(directory, artifacts, fit+'latest.json'))
    checkpoint = checked(directory, artifacts, fit+pointer['file'])
    if pointer['sha256'] != manifest['target_residual_sha256'] or pointer['sha256'] != artifacts[fit+pointer['file']]:
        raise ValueError('Adaptation checkpoint differs from frozen candidate')
    cursor = torch.load(checkpoint, map_location='cpu', weights_only=True)['cursor']
    losses = cursor['losses']
    if (cursor['step'] != expected or pointer['step'] != expected or len(losses) != expected
            or any(not math.isfinite(loss) or loss < 0 for loss in losses)):
        raise ValueError('Invalid saved adaptation trace')
    return {'adaptation_initial20_normalized_MAE': mean(losses[:20]),
            'adaptation_final20_normalized_MAE': mean(losses[-20:]),
            'adaptation_steps': expected}


def forecasts(path, bounds, nodes, include_truth=False):
    import numpy as np

    columns = ['local', 'transfer'] + (['truth'] if include_truth else [])
    origins = np.arange(*bounds, dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'origins', *columns} or not np.array_equal(archive['origins'], origins):
            raise ValueError('Forecast shard has unexpected columns or origins '+str(path))
        values = {col: archive[col].astype(np.float64, copy=False) for col in columns}
    if any(a.shape != (len(origins), nodes) or not np.isfinite(a).all() or np.any(a < 0)
           for a in values.values()):
        raise ValueError('Invalid forecast values '+str(path))
    return values


def error_profile(forecast, truth, scaler):
    import numpy as np

    error = forecast-truth
    per_origin = np.abs(error).mean(axis=1)
    top = math.ceil(.01*len(per_origin))
    total = per_origin.sum()
    profile = {'signed_bias': float(error.mean()), 'MAE_p50_by_origin': float(np.median(per_origin)),
               'MAE_p95_by_origin': float(np.quantile(per_origin, .95)),
               'MAE_worst_origin': float(per_origin.max()),
               'error_share_worst_1pct_origins': float(np.partition(per_origin, -top)[-top:].sum()/total) if total else 0.,
               'zero_forecast_fraction': float(np.mean(forecast == 0))}
    if scaler is not None:
        profile.update(forecast_above_support_max_fraction=float(np.mean(forecast > scaler['maximum'])),
                       truth_above_support_max_fraction=float(np.mean(truth > scaler['maximum'])))
    return profile, per_origin


def reconcile_metrics(row, values):
    from shared_baselines import independent_scores

    metrics = {}
    for model in ('local', 'transfer'):
        scores = independent_scores(values[model], values['truth'])
        for metric in METRICS:
            value = scores[metric]
            key = model+'_'+metric
            saved = float(row[key]) if row[key] else None
            if (value is None) != (saved is None) or (value is not None and not math.isclose(
                    value, saved, rel_tol=1e-10, abs_tol=1e-10)):
                raise ValueError('Recomputed metric differs from the original collector for '+key)
            metrics[key] = value
    return metrics


def diagnose_case(directory, artifacts, task, budget, row, raw, profile, src, updates):
    import numpy as np
    from ectt_data import fit_scaler

    bounds = profile['budgets'][budget]
    eligible = bounds['eligible_for_learned_adaptation']
    manifest = read(checked(directory, artifacts, budget+'/frozen_candidates.json'))
    if (row['eligible'] != str(eligible) or int(row['optimizer_steps']) != (updates if eligible else 0)
            or row['local_family'] != manifest['local_family'] or manifest['fallback'] != (not eligible)
            or manifest['test_targets'] != profile['test_targets']
            or manifest['source_checkpoint_sha256'] != src['source_best_sha256']
            or manifest['verification_labels_used'] != 0 or manifest['test_labels_used_for_fitting'] != 0):
        raise ValueError('Saved case metadata disagrees with the frozen recipe')
    scaler = None
    trace, verification_input = {}, None
    if eligible:
        support = bounds['support_raw']
        verification = bounds['verification_target_rows']
        scaler = fit_scaler(raw, support)
        if (manifest['scaler'] != scaler or manifest['support_raw'] != support
                or manifest['unique_support_values'] != (support[1]-support[0])*raw.shape[0]
                or manifest['verification_targets'] != verification
                or manifest['verification_history_union'] != [verification[0]-12, verification[1]-1]
                or not math.isfinite(manifest['support_scale']) or manifest['support_scale'] < 1):
            raise ValueError('Support scale or verification ledger differs')
        trace = adaptation_history(directory, artifacts, budget, manifest, updates)
        name = budget+'/verification_candidates.npz'
        path = checked(directory, artifacts, name)
        candidates = forecasts(path, verification, raw.shape[0])
        verification_input = {'task': task, 'budget': budget, 'path': str(path), 'sha256': artifacts[name],
                              'manifest_sha256': artifacts[budget+'/frozen_candidates.json'],
                              'support_rows': support, 'verification_target_rows': verification,
                              'verification_history_union': manifest['verification_history_union'],
                              'support_scale': manifest['support_scale'],
                              'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
                              'residual_checkpoint_sha256': manifest['target_residual_sha256'],
                              'local_forecast_sha256': manifest['local_forecast_sha256'],
                              'mean_candidate_disagreement': float(np.abs(candidates['transfer']-candidates['local']).mean())}
    elif manifest['support_raw'] is not None or manifest['unique_support_values'] != 0:
        raise ValueError('Fallback record claims support access')

    test = forecasts(checked(directory, artifacts, budget+'/test_forecasts.npz'),
                     profile['test_targets'], raw.shape[0], include_truth=True)
    if not np.array_equal(test['truth'], raw[:, slice(*profile['test_targets'])].T):
        raise ValueError('Saved test truth differs from the pinned raw cohort')
    if not eligible and (manifest['local_family'] != 'persistence' or not np.array_equal(test['local'], test['transfer'])):
        raise ValueError('Fallback forecasts differ from persistence reference')
    metrics = reconcile_metrics(row, test)
    errors, diagnostics = {}, {}
    for model in ('local', 'transfer'):
        diagnostic, errors[model] = error_profile(test[model], test['truth'], scaler)
        diagnostics.update({model+'_'+key: value for key, value in diagnostic.items()})
    excess = errors['transfer']-errors['local']
    return ({**{key: task[key] for key in ('family', 'dataset', 'seed')}, 'budget': budget,
             'eligible': eligible, 'local_family': manifest['local_family'],
             'origins': len(excess), 'nodes': raw.shape[0], **metrics, **diagnostics,
             'MAE_ratio': metrics['transfer_MAE']/metrics['local_MAE'] if metrics['local_MAE'] else None,
             'fraction_origins_transfer_worse': float(np.mean(excess > 0)),
             'fraction_origins_transfer_better': float(np.mean(excess < 0)),
             **(scaler or {}), **src, **trace}, verification_input)


def report_summary(rows, contract):
    development = set(contract['development_targets'])
    primary = str(contract['partitions']['primary_budget'])
    learned = [row for row in rows if row['eligible']]
    counts = Counter(row['family'] for row in learned)
    losses = sum(row['transfer_MAE'] > row['local_MAE'] for row in learned)
    lines = ['# Shared transfer diagnostic', '',
             f'Reconciled {len(rows)} saved records from collection {COLLECTION}. '
             f'{len(learned)} learned transfers ({dict(counts)}) and {len(rows)-len(learned)} persistence fallbacks.',
             f'Learned runs with test MAE above local: {losses}/{len(learned)}.', '',
             '## Development domains at the primary budget', '',
             'Rows summarize the three seeds. Bias is forecast minus truth. '
             'Tail share is the fraction of total transfer absolute error from the worst 1% of forecast origins.', '',
             '| Dataset | Model | Median MAE / local | Median signed bias | Median origins worse | Median tail share |',
             '|---|---|---:|---:|---:|---:|']
    for slug in contract['development_targets']:
        for family in ('B03', 'B04'):
            group = [r for r in learned if r['dataset'] == slug and r['family'] == family and r['budget'] == primary]
            ratio = [r['MAE_ratio'] for r in group if r['MAE_ratio'] is not None]
            middle = f'{median(ratio):.2f}' if len(ratio) == len(group) and ratio else 'undefined'
            lines.append(f"| {slug} | {family} | {middle} | {median(r['transfer_signed_bias'] for r in group):.3f} | "
                         f"{median(r['fraction_origins_transfer_worse'] for r in group):.1%} | "
                         f"{median(r['transfer_error_share_worst_1pct_origins'] for r in group):.1%} |")
    dev = [r for r in learned if r['dataset'] in development]
    improved = sum(r['adaptation_final20_normalized_MAE'] < r['adaptation_initial20_normalized_MAE'] for r in dev)
    lines += ['', f'Development runs with lower mean sampled adaptation loss in the last 20 steps: {improved}/{len(dev)}.',
              'Sampled adaptation losses describe optimization on support. They do not establish generalization.',
              'Support extrema were recalculated from the pinned raw support rows. Saved forecasts are in physical units. '
              'The physical-unit shards omit normalized pre-inverse predictions. Zero counts combine any clipped and direct zero predictions.',
              '', '## Largest observed relative errors', '',
              '| Dataset | Model | Budget | Seed | MAE / local | Origins worse | Tail share |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in sorted((r for r in learned if r['MAE_ratio'] is not None), key=lambda r: r['MAE_ratio'], reverse=True)[:5]:
        lines.append(f"| {row['dataset']} | {row['family']} | {float(row['budget']):.0%} | {row['seed']} | "
                     f"{row['MAE_ratio']:.2f} | {row['fraction_origins_transfer_worse']:.1%} | "
                     f"{row['transfer_error_share_worst_1pct_origins']:.1%} |")
    lines += ['', 'Confirmatory diagnostics are descriptive. Development decisions use PEMS08 and Beijing only.',
              'Detailed case diagnostics are in transfer_diagnostics.csv. The frozen verification input inventory is in S01_INPUTS.json.',
              'S01 independent sign-off, verification scoring and ECTT training are pending. This report uses the frozen test forecasts.', '']
    return '\n'.join(lines)


def run(project, code, report):
    from transfer_worker import raw_arrays, raw_paths

    matrix, csv_path = foundation(project, code)
    contract, _ = load_contract(code)
    store = plan.store_path(project, matrix)
    with csv_path.open(newline='') as stream:
        original = list(csv.DictReader(stream))
    by_case = {(r['family'], r['dataset'], int(r['seed']), r['budget']): r for r in original}
    if len(original) != len(by_case) or len(original) != 192:
        raise ValueError('Incomplete or duplicated original metric rows')
    sources = {}
    for task in matrix['source']:
        directory, artifacts = worker_artifacts(project, task, store, 'source', plan.SOURCE)
        result = read(checked(directory, artifacts, 'result.json'))
        src = source_history(result)
        if result['task'] != task or src['source_best_sha256'] != artifacts['best.pt']:
            raise ValueError('Source checkpoint receipt differs')
        sources[task['index']] = src
    rows, inputs = [], []
    paths = raw_paths(project, code, contract)
    raw = raw_arrays(paths, contract['datasets'], contract)
    for task in matrix['target']:
        directory, artifacts = worker_artifacts(project, task, store, 'target', plan.TARGET)
        src = sources[task['source_index']]
        slug = task['dataset']
        for budget in task['budgets']:
            key = task['family'], slug, task['seed'], budget
            row, input_record = diagnose_case(directory, artifacts, task, budget, by_case.pop(key), raw[slug],
                                               contract['datasets'][slug], src,
                                               contract['training_design']['target_updates'])
            rows.append(row)
            if input_record is not None:
                inputs.append(input_record)
        write(report/'transfer_progress.json', {'phase': 'SCORING', 'cases_checked': len(rows), 'last_task': task})
    if by_case or len(rows) != 192 or len(inputs) != 132:
        raise ValueError('Unexpected diagnostic case coverage')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = report/'transfer_diagnostics.csv'
    temp = output.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)
    write(report/'S01_INPUTS.json', {'independent_review': 'PENDING', 'collection_job': COLLECTION,
          'producer_commit': PRODUCER, 'matrix_identity': MATRIX, 'verification_labels_used_for_scoring': 0,
          'protocol_sha256': digest(code/'config/ectt_protocol_v1.json'),
          'statistics_code_sha256': digest(code/'scripts/ectt_statistics.py'),
          'baseline_metrics_sha256': digest(csv_path), 'cases': inputs})
    (report/'SUMMARY.md').write_text(report_summary(rows, contract))
    write(report/'transfer_result.json', {'scientific_status': 'TRANSFER_DIAGNOSTIC_COMPLETE',
          'diagnostic_cases_checked': len(rows), 'frozen_candidate_pairs_checked': len(inputs),
          'verification_labels_used_for_scoring': 0, 'S01_independent_review': 'PENDING',
          'new_optimizer_steps': 0, 'training_performed': False, 'method_evaluated': False})
    write(report/'transfer_progress.json', {'phase': 'COMPLETE', 'cases_checked': len(rows)})


def dispatch(project, code, report):
    python = inside(project, project/'.venv-ectt-gpu/bin/python')
    return subprocess.call([str(python), '-u', str(code/'scripts/transfer_diagnostics.py'),
                            str(project), str(code), str(report)],
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
                                'TRANSFORMERS_OFFLINE': '1', 'OPENBLAS_NUM_THREADS': '1'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        if not os.environ.get('SLURM_JOB_ID', '').isdigit():
            raise ValueError('Run this diagnostic inside a Slurm job')
        run(args.project.resolve(), inside(args.project, args.code), inside(args.project, args.report))
    except Exception as exc:
        traceback.print_exc()
        (args.report/'SUMMARY.md').write_text(f'# Shared transfer diagnostic failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
