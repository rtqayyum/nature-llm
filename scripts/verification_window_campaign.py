"""Batched development replay on predeclared chronological windows."""
import argparse
import csv
import fcntl
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
from transfer_plan import fingerprint
import verification_pilot as pilot
from verification_pilot_candidate_audit import admission


LAUNCH = 'shared-transfer-ectt-window-campaign'
PROBE = 'shared-transfer-ectt-window-probe'
FIT = 'shared-transfer-ectt-window-fit'
REPORT = 'shared-transfer-ectt-window-report'
STAGES = (PROBE, FIT, REPORT)
GPU_STAGES = (FIT,)
JOURNAL = 'verification_window_submissions'
RULES = ('locked_two_sided', 'fixed_budget_mean', 'one_sided_mixture')
WINDOWS_PER_DOMAIN = 4


def window_tasks(recipe, contract):
    tasks = []
    for domain in recipe['development_targets']:
        profile = contract['datasets'][domain]
        budget = profile['budgets'][recipe['budget']]
        support_len = budget['support_raw'][1]-budget['support_raw'][0]
        verify_len = budget['verification_target_rows'][1]-budget['verification_target_rows'][0]
        # Use the start of the largest original target support budget. Every
        # new window precedes all original target support/verification/test.
        cutoff = min(item['support_raw'][0] for item in profile['budgets'].values())
        size = support_len+2*verify_len
        start = cutoff-WINDOWS_PER_DOMAIN*size
        if start < 0 or support_len < 24 or verify_len < contract['partitions']['minimum_verification_origins']:
            raise ValueError(f'Insufficient contiguous rows for the locked windows in {domain}')
        for window in range(WINDOWS_PER_DOMAIN):
            a = start+window*size
            b, c, d = a+support_len, a+support_len+verify_len, a+size
            for seed in recipe['seeds']:
                tasks.append({'index': len(tasks), 'family': recipe['family'], 'dataset': domain,
                              'seed': seed, 'arm': 'all', 'window': window+1,
                              'support': [a, b], 'verification': [b, c], 'test': [c, d]})
    return tasks


def foundation(project, code):
    previous = admission(project, code)
    contract, _ = load_contract(code)
    recipe = previous['recipe']
    tasks = window_tasks(recipe, contract)
    if len(tasks) != 24 or len(recipe['arms']) != 3:
        raise ValueError('Unexpected development window or objective coverage')
    identities = {'original_pilot': previous['identity'], 'rules': list(RULES), 'tasks': tasks,
                  'implementation_sha256': digest(code/'scripts/verification_window_campaign.py')}
    return {'identity': fingerprint(identities), 'previous': previous, 'recipe': recipe,
            'tasks': tasks, 'probes': []}


def submission_steps(plan):
    return [('probe', PROBE, 'ectt-window-preflight', None),
            ('fit', FIT, 'ectt-window-fit', plan['tasks']),
            ('report', REPORT, 'ectt-window-report', None)]


def store_path(project, plan):
    return inside(project, project/'outputs/ectt/verification_windows'/plan['identity'])


def one_sided_log_mixture(benefit_sum, variance, rho):
    if benefit_sum <= 0:
        return 0.0  # The true mixture is at most one, so it cannot cross.
    z = benefit_sum/math.sqrt(variance+rho)
    return 0.5*(z*z-math.log1p(variance/rho))+math.log1p(math.erf(z/math.sqrt(2)))


def alternative(rule, origins, local, transfer, scale, matured, *, rho, delta, eta, minimum):
    import numpy as np

    running_benefit = running_variance = 0.
    trace = []
    accepted = False
    envelope = np.abs(transfer-local).mean(axis=1)/scale
    for i, origin in enumerate(origins):
        y = np.asarray(matured(int(origin)), dtype=np.float64)
        diff = float((np.abs(transfer[i]-y).mean()-np.abs(local[i]-y).mean())/scale)
        bound = float(envelope[i])
        if not math.isfinite(diff) or not math.isfinite(bound) or abs(diff) > bound+1e-12:
            raise ValueError('Invalid paired verification difference or bound')
        running_benefit -= diff+eta
        running_variance += bound*bound
        n = i+1
        if rule == 'one_sided_mixture':
            evidence = one_sided_log_mixture(running_benefit, running_variance, rho)
            if not math.isfinite(evidence):
                raise ValueError('Nonfinite one-sided verification statistic')
            accepted = n >= minimum and evidence > -math.log(delta)
        elif rule == 'fixed_budget_mean':
            evidence = running_benefit/n
            accepted = n == len(origins) and n >= minimum and evidence > 0
        else:
            raise ValueError('Unknown development verification rule')
        trace.append({'origin': int(origin), 'difference': diff, 'bound': bound,
                      'statistic': evidence, 'accepted': accepted})
        if accepted:
            break
    return {'accepted': accepted, 'trace': trace, 'verification_origins_read': len(trace)}


def decide_all(raw, bounds, local, transfer, scale, contract):
    import numpy as np
    from frozen_verifier_replay import replay

    origins = np.arange(*bounds)
    for value in (local, transfer):
        if value.shape != (len(origins), raw.shape[0]) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('Invalid frozen verification forecasts')
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('Invalid frozen support scale')
    settings = {'rho': contract['verifier']['rho'], 'delta': contract['verifier']['delta_per_stream'],
                'eta': contract['training_design']['mechanism_coefficients']['epsilon'],
                'minimum': contract['partitions']['minimum_verification_origins']}
    decisions = {}
    for rule in RULES:
        next_origin = bounds[0]

        def matured(origin):
            nonlocal next_origin
            if origin != next_origin or origin >= bounds[1]:
                raise ValueError('Verification labels must arrive in chronological order')
            next_origin += 1
            truth = np.asarray(raw[:, origin], dtype=np.float64)
            if not np.isfinite(truth).all() or np.any(truth < 0):
                raise ValueError('Invalid matured traffic truth')
            return truth

        args = (origins, local, transfer, scale, matured)
        result = replay(*args, **settings) if rule == RULES[0] else alternative(rule, *args, **settings)
        read_rows = next_origin-bounds[0]
        if result['verification_origins_read'] != read_rows or (not result['accepted'] and read_rows != len(origins)):
            raise ValueError('Verification label ledger mismatch')
        decisions[rule] = result
    return decisions


def source_path(project, plan, task, arm):
    previous = plan['previous']
    source_task, = [row for row in previous['tasks'] if row['dataset'] == task['dataset']
                    and row['seed'] == task['seed'] and row['arm'] == arm]
    if task['dataset'] in source_task['sources']:
        raise ValueError('Target domain was part of the source fit')
    case = pilot.store_path(project, previous)/'cases'/str(source_task['index'])
    completion = read(case/'COMPLETE.json')
    source = read(case/'result.json')
    checksum = source['source']['best_sha256']
    if (completion['identity'] != previous['identity'] or source['task'] != source_task
            or source['scientific_status'] != 'ECTT_PILOT_FROZEN_CASE_COMPLETE'
            or completion['artifacts']['best.pt'] != checksum
            or completion['artifacts']['result.json'] != digest(case/'result.json')):
        raise ValueError('Archived source model lineage differs')
    return source_task, inside(case, case/'best.pt'), checksum


def preflight(project, code, report, plan, contract):
    import numpy as np
    from ectt_data import fit_scaler
    from transfer_worker import raw_arrays, raw_paths

    raw = raw_arrays(raw_paths(project, code, contract), plan['recipe']['development_targets'], contract)
    checked = set()
    for task in plan['tasks']:
        domain, seed = task['dataset'], task['seed']
        a, b = task['support']
        c = a+int(.7*(b-a))
        if c <= a+12:
            raise ValueError('Not enough history for support-only local selection')
        fit_scaler(raw[domain], [a, c])
        for arm in plan['recipe']['arms']:
            if (domain, seed, arm) not in checked:
                _, weights, checksum = source_path(project, plan, task, arm)
                if digest(weights) != checksum:
                    raise ValueError('Archived source checkpoint checksum differs')
                checked.add((domain, seed, arm))
    truth = np.full((2, 90), 2.)
    local = np.full((80, 2), 10.)
    decisions = decide_all(truth, [10, 90], local, np.full((80, 2), 2.), 1., contract)
    if not all(value['accepted'] for value in decisions.values()):
        raise ValueError('Strong synthetic improvement was not selected')
    if decisions[RULES[1]]['verification_origins_read'] != 80 or any(
            decisions[rule]['verification_origins_read'] >= 80 for rule in (RULES[0], RULES[2])):
        raise ValueError('Synthetic stopping rules differ')
    if any(value['accepted'] for value in decide_all(truth, [10, 90], local, local, 1., contract).values()):
        raise ValueError('Identical candidates incorrectly selected transfer')
    write(store_path(project, plan)/'preflight.json', {'identity': plan['identity'], 'tasks': plan['tasks'],
                                                      'verified_source_checkpoints': len(checked)})
    (report/'SUMMARY.md').write_text('# ECTT window preflight\n\nValidated 18 source checkpoints and 24 window/seed tasks.\n')
    return {'scientific_status': 'ECTT_WINDOW_PREFLIGHT_COMPLETE', 'diagnostic_cases_checked': len(checked)}


def fit_window(project, code, report, plan, contract, task, case, progress):
    import numpy as np
    import torch
    from compact_training import configure
    from ectt_data import fit_scaler
    from shared_baselines import independent_scores
    from transfer_train import adapt, forecast, invariant_probe, restore_model
    from transfer_worker import local_recipes, raw_arrays, raw_paths, save_arrays
    from verification_training import local_reference

    preflight_record = read(store_path(project, plan)/'preflight.json')
    if preflight_record['identity'] != plan['identity'] or preflight_record['tasks'] != plan['tasks']:
        raise ValueError('Window preflight differs from the submitted matrix')
    domain = task['dataset']
    raw = raw_arrays(raw_paths(project, code, contract), [domain], contract)[domain]
    raw = raw[:, :task['test'][1]]
    support, prefix, test = task['support'], task['verification'], task['test']
    a, b = support
    configure()
    scaler = fit_scaler(raw, support)
    device = 'cuda'
    all_origins = [prefix[0], test[1]]
    prefix_count = prefix[1]-prefix[0]
    scale, local = local_reference(raw, {'support': support, 'verification': all_origins},
                                   task['seed'], local_recipes(code), device, case/'local',
                                   plan['identity'], True)
    local_hash = digest(case/'local/selection.json')
    decisions, predictions, lineage = {}, {}, {}
    total_updates = 0
    recipe = plan['previous']['baseline']['recipe']
    for arm in plan['recipe']['arms']:
        source_task, weights, checksum = source_path(project, plan, task, arm)
        if digest(weights) != checksum:
            raise ValueError('Archived source checkpoint checksum differs')
        model, runtime = pilot.model_for(project, code, contract, source_task, report)
        restore_model(model, torch.load(weights, map_location='cpu', weights_only=True))
        model.eval().requires_grad_(False)
        key = fingerprint({'campaign': plan['identity'], 'task': task, 'arm': arm, 'scaler': scaler,
                           'source': checksum, 'runtime': runtime})
        arm_dir = case/arm
        head, cursor = adapt(model, raw, [a+12, b], scaler, task['seed'], recipe,
                             contract['training_design'], device,
                             lambda **values: progress(arm=arm, **values), arm_dir/'target', key)
        total_updates += cursor['new_steps']
        gates = invariant_probe(model, raw, support, task['family'], device, head)
        values = forecast(model, raw, all_origins, scaler, recipe, device, head)
        del head, model
        torch.cuda.empty_cache()
        if values.shape != local.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid transfer forecast cohort')
        save_arrays(arm_dir/'frozen.npz', origins=np.arange(*all_origins), local=local, transfer=values)
        predictions[arm] = values
        lineage[arm] = {'source_sha256': checksum, 'forecasts_sha256': digest(arm_dir/'frozen.npz'),
                        'target_head_sha256': read(arm_dir/'target/latest.json')['sha256'], 'invariance': gates,
                        'target_optimizer_steps': cursor['step']}
        progress(phase='CANDIDATE_FROZEN', arm=arm)
    write(case/'frozen_manifest.json', {'task': task, 'scaler': scaler, 'support_scale': scale,
                                      'local_selection_sha256': local_hash, 'arms': lineage})
    for arm, values in predictions.items():
        decisions[arm] = decide_all(raw, prefix, local[:prefix_count], values[:prefix_count], scale, contract)
    decision_file = case/'decisions.json'
    record = {'identity': plan['identity'], 'task': task, 'decisions': decisions}
    if decision_file.exists() and read(decision_file) != record:
        raise ValueError('Previously frozen window decisions differ')
    write(decision_file, record)
    decision_hash = digest(decision_file)
    truth = np.asarray(raw[:, slice(*test)].T, dtype=np.float64)
    if truth.shape != local[prefix_count:].shape or not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError('Invalid window test truth cohort')
    local_mae = independent_scores(local[prefix_count:], truth)['MAE']
    local_errors = np.abs(local[prefix_count:]-truth).mean(axis=1)
    eta = contract['training_design']['mechanism_coefficients']['epsilon']
    rows = []
    for arm, values in predictions.items():
        transfer_mae = independent_scores(values[prefix_count:], truth)['MAE']
        for rule, decision in decisions[arm].items():
            accepted = decision['accepted']
            deployed = values[prefix_count:] if accepted else local[prefix_count:]
            metrics = independent_scores(deployed, truth)
            excess = (np.abs(deployed-truth).mean(axis=1)-local_errors)/scale
            count = decision['verification_origins_read']
            rows.append({'dataset': domain, 'window': task['window'], 'seed': task['seed'], 'objective': arm,
                         'rule': rule, 'accepted': accepted, 'useful_acceptance': accepted and transfer_mae < local_mae,
                         'harmful_acceptance': accepted and transfer_mae > local_mae,
                         'verification_rows_read': count, 'verification_rows_available': prefix_count,
                         'verification_labels_used': count*raw.shape[0], 'support_labels_used': (b-a)*raw.shape[0],
                         'material_harm_fraction': float(np.mean(excess > eta)),
                         'local_mae': local_mae, 'always_transfer_mae': transfer_mae,
                         'deployed_mae': metrics['MAE'], 'deployed_rmse': metrics['RMSE'],
                         'deployed_wmape_percent': metrics['WMAPE_percent']})
    if digest(decision_file) != decision_hash:
        raise ValueError('Frozen decisions changed during test scoring')
    result = {'task': task, 'rows': rows, 'new_optimizer_steps': total_updates,
              'scientific_status': 'ECTT_WINDOW_CASE_COMPLETE'}
    write(case/'result.json', result)
    finish(case, plan['identity'])
    return result


def collect(project, report, plan):
    root = store_path(project, plan)
    jobs = read(project/'work'/JOURNAL/(plan['identity']+'.json'))['jobs']
    rows, missing = [], []
    for task in plan['tasks']:
        case = root/'cases'/str(task['index'])
        try:
            pilot.job_report(project, f"{jobs['fit']}_{task['index']}", FIT)
            if not completed(case, plan['identity']):
                raise ValueError('Case incomplete')
            result = read(case/'result.json')
            keys = {(row['objective'], row['rule']) for row in result['rows']}
            if result['task'] != task or len(result['rows']) != 9 or keys != {
                    (arm, rule) for arm in plan['recipe']['arms'] for rule in RULES}:
                raise ValueError('Unmatched window or verifier outcomes')
            rows.extend(result['rows'])
        except (OSError, ValueError, KeyError) as exc:
            missing.append({'task': task, 'reason': str(exc)})
    if rows:
        with (report/'window_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# ECTT chronological development replay', '',
             f'Collected {len(rows)}/216 policy replays from {len(plan["tasks"])-len(missing)}/24 window/seed tasks.',
             'Development-only comparison. Independent S01 review is pending.', '']
    if missing:
        lines.append('Some paired runs are missing. See transfer_result.json for task-level errors.')
    else:
        lines += ['| Dataset | Objective | Rule | Local MAE | Deployed MAE | Useful / harmful / accepted | Mean verification rows |',
                  '|---|---|---|---:|---:|---:|---:|']
        for domain in plan['recipe']['development_targets']:
            for arm in plan['recipe']['arms']:
                for rule in RULES:
                    group = [r for r in rows if (r['dataset'], r['objective'], r['rule']) == (domain, arm, rule)]
                    counts = ' / '.join(str(sum(r[key] for r in group)) for key in
                                        ('useful_acceptance', 'harmful_acceptance', 'accepted'))
                    lines.append(f"| {domain} | {arm} | {rule} | {mean(r['local_mae'] for r in group):.3f} | "
                                 f"{mean(r['deployed_mae'] for r in group):.3f} | {counts} | "
                                 f"{mean(r['verification_rows_read'] for r in group):.1f} |")
        lines += ['', 'Each group contains four windows crossed with three paired seeds. Counts are descriptive.',
                  'Full paired results and actual label counts are in window_metrics.csv.']
    (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return {'scientific_status': 'ECTT_WINDOW_CAMPAIGN_INCOMPLETE' if missing else 'ECTT_WINDOW_CAMPAIGN_COMPLETE',
            'diagnostic_cases_checked': len(rows), 'missing_tasks': missing}


def run(project, code, report, stage):
    plan = foundation(project, code)
    contract, _ = load_contract(code)

    def progress(**fields):
        write(report/'transfer_progress.json', fields)
        print('ECTT_WINDOW_PROGRESS', fields, flush=True)

    if stage == PROBE:
        result = preflight(project, code, report, plan, contract)
    elif stage == REPORT:
        result = collect(project, report, plan)
    else:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if not 0 <= index < len(plan['tasks']):
            raise ValueError('Unexpected Slurm array index')
        task = plan['tasks'][index]
        case = store_path(project, plan)/'cases'/str(task['index'])
        case.mkdir(parents=True, exist_ok=True)
        with (case/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if completed(case, plan['identity']):
                result = {**read(case/'result.json'), 'new_optimizer_steps': 0}
            else:
                result = fit_window(project, code, report, plan, contract, task, case, progress)
        (report/'SUMMARY.md').write_text(f"# ECTT window task\n\n{task['dataset']} window {task['window']}, seed {task['seed']}. Nine paired replays completed.\n")
    write(report/'transfer_result.json', {**result, 'identity': plan['identity'], 'method_evaluated': False,
          'S01_independent_review': 'PENDING', 'evaluation_scope': 'EXPLORATORY_DEVELOPMENT'})
    progress(phase='FAILED' if result.get('missing_tasks') else 'COMPLETED')
    return 2 if result.get('missing_tasks') else 0


def dispatch(project, code, report, stage):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/verification_window_campaign.py'),
                            str(project), str(code), str(report), stage],
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    parser.add_argument('stage', choices=STAGES)
    args = parser.parse_args()
    project = args.project.resolve()
    report = inside(project, args.report)
    try:
        sys.exit(run(project, inside(project, args.code), report, args.stage))
    except Exception as exc:
        traceback.print_exc()
        (report/'SUMMARY.md').write_text(f'# ECTT window stage failed\n\n{exc}\n')
        write(report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
