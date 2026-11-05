"""Replay saved checkpoints on development test origins before and after adaptation."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
import transfer_diagnostics as diagnostic
import transfer_plan as plan

STAGE = 'shared-transfer-attribution'
DIAGNOSTIC_JOB = '622973'
DIAGNOSTIC_COMMIT = '9043cfc39f973b69efefd5a79a3c26deb8ab68b0'
TOLERANCE = 1e-3


def foundation(project, code):
    matrix, _ = diagnostic.foundation(project, code)
    job = inside(project, project/'results/slurm_jobs'/DIAGNOSTIC_JOB)
    status = read(job/'job_status.json')
    result = read(job/'report/transfer_result.json')
    inventory = job/'report/S01_INPUTS.json'
    if (status.get('job_id'), status.get('stage'), status.get('execution_state'),
            status.get('shell_exit_code')) != (DIAGNOSTIC_JOB, diagnostic.STAGE, 'COMPLETED', 0):
        raise ValueError('Completed diagnostic job '+DIAGNOSTIC_JOB+' is required')
    submission = read(job/'submission.json')
    expected_code = inside(project, project/'work/code_versions'/DIAGNOSTIC_COMMIT)
    if Path(submission['code']).resolve() != expected_code or result.get('scientific_status') != 'TRANSFER_DIAGNOSTIC_COMPLETE':
        raise ValueError('Diagnostic receipt differs from the pinned producer')
    if (result.get('diagnostic_cases_checked'), result.get('frozen_candidate_pairs_checked'),
            result.get('verification_labels_used_for_scoring')) != (192, 132, 0):
        raise ValueError('Incomplete or incompatible diagnostic receipt')
    reviewed = read(inventory)
    if (reviewed['matrix_identity'] != matrix['identity'] or reviewed['producer_commit'] != diagnostic.PRODUCER
            or reviewed['protocol_sha256'] != digest(code/'config/ectt_protocol_v1.json')):
        raise ValueError('Diagnostic inventory differs from the archived transfer protocol')
    return matrix, inventory


def predict_pair(model, residual, raw, bounds, scaler, batch_size, device):
    import numpy as np
    import torch
    from compact_training import data_batch
    from shared_baselines import batches

    before, after = [], []
    with torch.no_grad():
        for ids in batches(bounds, batch_size):
            features, _ = data_batch(raw, ids, scaler, device)
            embedding = model.encode(features)
            source = model.head(embedding)
            before.append(source.squeeze(-1).cpu().numpy())
            after.append((source+residual(embedding)).squeeze(-1).cpu().numpy())
    return np.concatenate(before), np.concatenate(after)


def compare(before_z, after_z, saved, scaler):
    import numpy as np
    from ectt_data import inverse
    from shared_baselines import independent_scores

    truth = saved['truth']
    if any(z.shape != truth.shape or not np.isfinite(z).all() for z in (before_z, after_z)):
        raise ValueError('Invalid paired normalized forecasts')
    before, replay = inverse(before_z, scaler), inverse(after_z, scaler)
    archived = saved['transfer']
    span = scaler['maximum']-scaler['minimum']
    replay_delta = float(np.max(np.abs(replay-archived)))
    if not np.allclose((replay-scaler['minimum'])/span, (archived-scaler['minimum'])/span,
                       atol=TOLERANCE, rtol=TOLERANCE):
        raise ValueError(f'Adapted checkpoint replay differs from the archive (max physical delta {replay_delta:.6g})')

    metrics = {}
    for label, predictions in [('local', saved['local']), ('before', before), ('after', archived)]:
        metrics.update({label+'_'+key: value for key, value in independent_scores(predictions, truth).items()})
        metrics[label+'_signed_bias'] = float((predictions-truth).mean())
    before_abs = np.abs(before-truth).mean(axis=1)
    after_abs = np.abs(archived-truth).mean(axis=1)
    metrics.update(after_minus_before_MAE=metrics['after_MAE']-metrics['before_MAE'],
                   after_minus_before_signed_bias=metrics['after_signed_bias']-metrics['before_signed_bias'],
                   fraction_origins_adaptation_improves=float(np.mean(after_abs < before_abs)),
                   fraction_origins_adaptation_worsens=float(np.mean(after_abs > before_abs)),
                   before_clipped_fraction=float(np.mean(scaler['minimum']+before_z.astype(np.float64)*span < 0)),
                   after_replay_clipped_fraction=float(np.mean(scaler['minimum']+after_z.astype(np.float64)*span < 0)),
                   replay_max_physical_delta=replay_delta,
                   replay_max_normalized_delta=replay_delta/span)
    return metrics


def score_case(project, code, report, task, matrix, contract, raw, budget, output):
    import numpy as np
    import torch
    from ectt_data import fit_scaler
    from transfer_model import residual_head
    from transfer_train import restore_model
    from transfer_worker import gpu, save_arrays

    store = plan.store_path(project, matrix)
    source_task = matrix['source'][task['source_index']]
    source, source_artifacts = diagnostic.worker_artifacts(project, source_task, store, 'source', plan.SOURCE)
    source_checkpoint = diagnostic.checked(source, source_artifacts, 'best.pt')
    target, artifacts = diagnostic.worker_artifacts(project, task, store, 'target', plan.TARGET)
    case = budget+'/'
    manifest = read(diagnostic.checked(target, artifacts, case+'frozen_candidates.json'))
    profile = contract['datasets'][task['dataset']]
    support = profile['budgets'][budget]['support_raw']
    test = profile['test_targets']
    scaler = fit_scaler(raw, support)
    if (manifest['fallback'] or manifest['support_raw'] != support or manifest['scaler'] != scaler
            or manifest['test_targets'] != test or manifest['source_checkpoint_sha256'] != source_artifacts['best.pt']
            or manifest['verification_labels_used'] or manifest['test_labels_used_for_fitting']):
        raise ValueError('Frozen candidate does not match the source checkpoint or development cohort')
    name = case+'test_forecasts.npz'
    archive_path = diagnostic.checked(target, artifacts, name)
    saved = diagnostic.forecasts(archive_path, test, raw.shape[0], include_truth=True)
    if not np.array_equal(saved['truth'], raw[:, slice(*test)].T):
        raise ValueError('Archived development truth differs from pinned data')
    fit = case+'fit/'
    pointer = read(diagnostic.checked(target, artifacts, fit+'latest.json'))
    checkpoint = diagnostic.checked(target, artifacts, fit+pointer['file'])
    if (pointer['sha256'] != artifacts[fit+pointer['file']]
            or pointer['sha256'] != manifest['target_residual_sha256']
            or pointer['step'] != contract['training_design']['target_updates']):
        raise ValueError('Residual checkpoint does not match the frozen candidate')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['cursor']['step'] != pointer['step']:
        raise ValueError('Residual checkpoint update count differs')
    model, runtime = gpu(project, code, contract, task, report)
    restore_model(model, torch.load(source_checkpoint, map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    residual = residual_head(state['model']['0.weight'].shape[1], 'cuda')
    restore_model(residual, state['model'])
    residual.eval().requires_grad_(False)
    before_z, after_z = predict_pair(model, residual, raw, test, scaler,
                                     matrix['recipe']['prediction_origin_batch'], 'cuda')
    metrics = compare(before_z, after_z, saved, scaler)
    save_arrays(output/'paired_normalized_forecasts.npz', origins=np.arange(*test), before=before_z, after_replay=after_z)
    return {'task': task, 'budget': budget, 'test_targets': test, 'origins': len(saved['truth']),
            'support_scaler': scaler, 'runtime': runtime, 'metrics': metrics,
            'source_checkpoint_sha256': source_artifacts['best.pt'], 'target_residual_sha256': pointer['sha256'],
            'original_forecasts_sha256': artifacts[name], 'original_forecasts': str(archive_path)}


def summarize(report, rows, budget):
    lines = ['# Development forecast comparison before and after target adaptation', '',
             f'Completed {len(rows)} paired cases at the {float(budget):.0%} target-label budget. '
             'Forecast origins and support scaling match the saved transfer run.', '',
             'The source-only column uses the same target support scaler with the residual head disabled. '
             'The after column uses the archived adapted forecast, checked against checkpoint replay.', '',
             '| Dataset | Model | Local MAE | Source-only MAE | Adapted MAE | Change in MAE | Source-only bias | Adapted bias |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in ('pems08', 'bj'):
        for family in ('B03', 'B04'):
            group = [row for row in rows if (row['dataset'], row['family']) == (dataset, family)]
            keys = ('local_MAE', 'before_MAE', 'after_MAE', 'after_minus_before_MAE', 'before_signed_bias', 'after_signed_bias')
            values = ' | '.join(f'{median(row[key] for row in group):.3f}' for key in keys)
            lines.append(f'| {dataset} | {family} | {values} |')
    improved = sum(row['after_minus_before_MAE'] < 0 for row in rows)
    worsened = sum(row['after_minus_before_MAE'] > 0 for row in rows)
    lines += ['', 'Table entries are medians over three seeds. Change is the median of paired differences.',
              f'Adaptation lowered test MAE in {improved}/{len(rows)} cases and increased it in {worsened}/{len(rows)} cases.',
              f"Maximum replay discrepancy across cases: {max(row['replay_max_normalized_delta'] for row in rows):.6g} in support-normalized units.",
              'Detailed seed results and clipping fractions are in transfer_before_after.csv.',
              'Scope is development test inference from existing checkpoints. S01 independent review and verifier scoring remain pending.', '']
    csv_path = report/'transfer_before_after.csv'
    temp = csv_path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)
    (report/'SUMMARY.md').write_text('\n'.join(lines))


def run(project, code, report):
    from transfer_worker import raw_arrays, raw_paths

    matrix, inventory = foundation(project, code)
    contract, _ = load_contract(code)
    budget = str(contract['partitions']['primary_budget'])
    domains = contract['development_targets']
    tasks = [task for task in matrix['target'] if task['dataset'] in domains]
    if len(tasks) != 12 or any(not contract['datasets'][task['dataset']]['budgets'][budget]['eligible_for_learned_adaptation']
                               for task in tasks):
        raise ValueError('Unexpected primary development comparison matrix')
    identity = plan.fingerprint({'matrix': matrix['identity'], 'inventory': digest(inventory),
                                 'script': digest(Path(__file__)), 'budget': budget})
    store = inside(project, project/'outputs/ectt/shared_transfer_attribution'/identity)
    raw = raw_arrays(raw_paths(project, code, contract), domains, contract)
    rows = []
    for task in tasks:
        case = inside(store, store/str(task['index']))
        case.mkdir(parents=True, exist_ok=True)
        write(report/'transfer_progress.json', {'phase': 'FORECAST_REPLAY', 'cases_checked': len(rows),
                                                'expected_cases': len(tasks), 'current_task': task})
        with (case/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if completed(case, identity):
                result = read(case/'result.json')
            else:
                result = score_case(project, code, report, task, matrix, contract, raw[task['dataset']], budget, case)
                write(case/'result.json', result)
                finish(case, identity)
        if result['task'] != task or result['budget'] != budget:
            raise ValueError('Reused case belongs to a different comparison')
        rows.append({**{key: task[key] for key in ('family', 'dataset', 'seed')},
                     'budget': budget, 'origins': result['origins'], **result['metrics']})
    summarize(report, rows, budget)
    write(report/'transfer_result.json', {'scientific_status': 'DEVELOPMENT_BEFORE_AFTER_COMPLETE',
          'diagnostic_cases_checked': len(rows), 'store': str(store), 'diagnostic_job': DIAGNOSTIC_JOB,
          'new_optimizer_steps': 0, 'training_performed': False, 'method_evaluated': False,
          'verification_labels_used_for_scoring': 0, 'S01_independent_review': 'PENDING'})
    write(report/'transfer_progress.json', {'phase': 'COMPLETE', 'cases_checked': len(rows)})


def dispatch(project, code, report):
    from ectt_env import ensure

    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/transfer_attribution.py'),
                            str(project), str(code), str(report)],
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        if not os.environ.get('SLURM_JOB_ID', '').isdigit():
            raise ValueError('Run checkpoint inference inside a Slurm job')
        run(args.project.resolve(), inside(args.project, args.code), inside(args.project, args.report))
    except Exception as exc:
        traceback.print_exc()
        (args.report/'failure_traceback.txt').write_text(traceback.format_exc())
        (args.report/'SUMMARY.md').write_text(f'# Development forecast comparison failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
