"""Prepare the matched ECTT pilot on development targets and freeze forecasts."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import anchored_transfer_development as anchor
from ectt_common import completed, digest, finish, inside, load_contract, read, write
import transfer_plan as baseline


LAUNCH = 'shared-transfer-ectt-development'
REVIEW = 'shared-transfer-ectt-review'
PROBE = 'shared-transfer-ectt-probe'
FIT = 'shared-transfer-ectt-fit'
REPORT = 'shared-transfer-ectt-report'
SCORE = 'shared-transfer-ectt-score'
STAGES = (REVIEW, PROBE, FIT, REPORT, SCORE)
GPU_STAGES = (PROBE, FIT)
JOURNAL = 'verification_pilot_submissions'
CONFIG = 'config/verification_training_pilot_v1.json'
REVIEW_FILES = (
    CONFIG, 'config/ectt_protocol_v1.json', 'research/S01_REVIEW_CHECKLIST.md',
    'research/ECTT_V2_DESIGN.md', 'research/RELATED_WORK.md', 'config/sources.lock.json',
    'config/shared_transfer_v1.json', 'config/shared_baselines_v1.json', 'config/compact_local_v1.json',
    'research/VERIFICATION_TRAINING_PILOT_V1.md', 'scripts/verification_training.py',
    'scripts/verification_pilot.py', 'scripts/verification_pilot_score.py',
    'scripts/frozen_verifier_replay.py', 'scripts/ectt_statistics.py', 'scripts/statistical_audit.py',
    'scripts/transfer_worker.py', 'scripts/transfer_train.py', 'scripts/ectt_data.py',
    'scripts/compact_training.py', 'scripts/shared_baselines.py', 'scripts/anchored_transfer_model.py',
    'scripts/transfer_model.py', 'scripts/compact_model.py', 'scripts/ectt_common.py',
    'scripts/anchored_transfer_development.py', 'scripts/transfer_plan.py', 'scripts/transfer_parallel.py',
)


def reviewed_hashes(code):
    return {name: digest(code/name) for name in REVIEW_FILES}


def matrix(code):
    previous = anchor.matrix(code)
    recipe = read(code/CONFIG)
    if (recipe['family'], recipe['anchor'], recipe['arms']) != (
            'B03', 'persistence_residual', ['suffix_mae', 'prefix_mae', 'verification_margin']):
        raise ValueError('Unexpected pilot family or objective registry')
    for key in ('development_targets', 'seeds', 'budget', 'maximum_concurrent_gpu_tasks'):
        if recipe[key] != previous['recipe'][key]:
            raise ValueError('Pilot differs from the matched anchor contract: '+key)
    tasks = []
    for row in previous['tasks']:
        if row['family'] == recipe['family'] and row['arm'] == recipe['anchor']:
            for arm in recipe['arms']:
                tasks.append({**row, 'index': len(tasks), 'arm': arm})
    probes = [row for row in tasks if row['dataset'] == 'pems08' and row['seed'] == 11]
    hashes = reviewed_hashes(code)
    ident = baseline.fingerprint({'anchor': previous['identity'], 'files': hashes})
    if len(tasks) != 18 or len(probes) != 3 or any(row['dataset'] in row['sources'] for row in tasks):
        raise ValueError('Pilot task coverage or target exclusion differs')
    return {'identity': ident, 'recipe': recipe, 'baseline': previous['baseline'],
            'previous': previous, 'tasks': tasks, 'probes': probes, 'reviewed_hashes': hashes}


def foundation(project, code):
    plan = matrix(code)
    recipe = plan['recipe']
    job = inside(project, project/'results/slurm_jobs'/recipe['predecessor_report_job'])
    status, submission, result = (read(job/name) for name in
                                  ('job_status.json', 'submission.json', 'report/transfer_result.json'))
    pinned = inside(project, project/'work/code_versions'/recipe['predecessor_commit'])
    if (status['stage'], status['execution_state'], status['shell_exit_code']) != (anchor.REPORT, 'COMPLETED', 0):
        raise ValueError('Completed anchor collector 2773172 is required')
    if (Path(submission['code']).resolve() != pinned
            or result.get('identity') != plan['previous']['identity']
            or result.get('scientific_status') != 'ANCHOR_DEVELOPMENT_COMPLETE'
            or result.get('diagnostic_cases_checked') != 24 or result.get('failed')):
        raise ValueError('Anchor collection does not match the pinned development experiment')
    return plan


def submission_steps(plan):
    return [('review', REVIEW, 'ectt-method-review', None),
            ('probe', PROBE, 'ectt-method-probe', plan['probes']),
            ('fit', FIT, 'ectt-method-fit', plan['tasks']),
            ('report', REPORT, 'ectt-method-ready', None)]


def store_path(project, plan):
    return inside(project, project/'outputs/ectt/verification_pilot'/plan['identity'])


def approval_path(project, plan):
    return inside(project, project/'work/ectt_s01_reviews'/(plan['identity']+'.json'))


def review_packet(project, code, report, plan, progress):
    result = anchor.review_packet(project, code, report, plan, progress)
    packet = report/'S01_packet'
    hashes = plan['reviewed_hashes']
    for name, expected in hashes.items():
        destination = packet/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code/name, destination)
        if digest(destination) != expected:
            raise ValueError('Review packet file differs from pinned source')
    write(packet/'manifest.json', {'pilot_identity': plan['identity'], 'code_files': hashes,
                                  'traffic_labels_read': 0, 'independent_review_complete': False})
    write(packet/'review_disposition_template.json', {
        'status': 'PENDING_INDEPENDENT_REVIEW', 'reviewer': None, 'date': None,
        'pilot_identity': plan['identity'], 'reviewed_hashes': hashes, 'findings': [],
        'disposition': None, 'independent_review_complete': False})
    with (report/'SUMMARY.md').open('a') as stream:
        stream.write('\nThe packet includes the pilot scorer and its exact code hashes.\n'
                     'An independent reviewer records the disposition in the supplied template.\n'
                     f'Approved record path: {approval_path(project, plan)}\n')
    return result


def model_for(project, code, contract, task, report):
    return anchor.build(project, code, contract, {**task, 'arm': 'persistence_residual'}, report)


def train_source(project, code, report, case, task, plan, contract, progress, probe):
    import torch
    from anchored_transfer_model import initial_check
    from compact_training import data_batch
    from ectt_data import fit_scaler
    from transfer_train import adapt, episode_bank, equal_state, forecast, invariant_probe, load, restore_model
    from transfer_worker import local_recipes, raw_arrays, raw_paths
    from verification_training import fit

    began = time.monotonic()
    recipe = plan['baseline']['recipe']
    training = dict(contract['training_design'])
    if probe:
        training['source_max_epochs'] = 1
    raw = raw_arrays(raw_paths(project, code, contract), task['sources'], contract)
    bank = episode_bank(task, contract, recipe, raw)
    for domain, episodes in bank.items():
        bounds = contract['datasets'][domain]['source_train_raw']
        if any(not bounds[0] <= episode['support'][0] < episode['suffix'][1] <= bounds[1]
               for episode in episodes['train']):
            raise ValueError('Source objective labels fall outside source training rows')
    write(case/'episode_ledger.json', {'task': task, 'episodes': bank, 'held_out_domains_read': []})
    model, runtime = model_for(project, code, contract, task, report)
    ident = baseline.fingerprint({'pilot': plan['identity'], 'task': task, 'runtime': runtime,
                                  'episodes': bank, 'probe': probe})
    domain = max(raw, key=lambda name: raw[name].shape[0])
    episode = bank[domain]['train'][0]
    support = episode['support']
    scaler = fit_scaler(raw[domain], support)
    x, _ = data_batch(raw[domain], [support[0]+12, support[0]+13], scaler, 'cuda')
    initial_check(model, x)
    before = invariant_probe(model, raw[domain], support, task['family'], 'cuda')
    args = (raw, task, bank, recipe, training, contract['verifier'], local_recipes(code), 'cuda')
    cursor, steps = fit(model, *args, case/'source', ident, progress)
    if probe:
        expected = load(case/'source', ident)
        del model
        torch.cuda.empty_cache()
        model, _ = model_for(project, code, contract, task, report)
        if not (case/'source-resumed/latest.json').exists():
            _, updates = fit(model, *args, case/'source-resumed', ident, progress, stop_after=2)
            steps += updates
        del model
        torch.cuda.empty_cache()
        model, _ = model_for(project, code, contract, task, report)
        _, updates = fit(model, *args, case/'source-resumed', ident, progress)
        steps += updates
        if not equal_state(expected, load(case/'source-resumed', ident)):
            raise ValueError('Source model, Adam, RNG or cursor differs after recovery')
        del expected, cursor
        head_training = {**training, 'target_updates': recipe['recovery_probe_steps']}
        head_args = (model, raw[domain], [support[0]+12, support[1]], scaler, task['seed'],
                     recipe, head_training, 'cuda', progress)
        head, info = adapt(*head_args, checkpoint=case/'head', identity=ident)
        steps += info['new_steps']
        expected_prediction = forecast(model, raw[domain], episode['suffix'], scaler, recipe, 'cuda', head)
        if not (case/'head-resumed/latest.json').exists():
            _, info = adapt(*head_args, checkpoint=case/'head-resumed', identity=ident, stop_after=2)
            steps += info['new_steps']
        head, info = adapt(*head_args, checkpoint=case/'head-resumed', identity=ident)
        steps += info['new_steps']
        prediction = forecast(model, raw[domain], episode['suffix'], scaler, recipe, 'cuda', head)
        if (not equal_state(load(case/'head', ident), load(case/'head-resumed', ident))
                or not (prediction == expected_prediction).all()):
            raise ValueError('Residual model, Adam, RNG or forecasts differ after recovery')
        return {'task': task, 'scientific_status': 'ECTT_PILOT_RECOVERY_PASSED',
                'new_optimizer_steps': steps, 'invariance_before': before,
                'invariance_after': invariant_probe(model, raw[domain], support, task['family'], 'cuda', head)}
    restore_model(model, cursor['best_model'])
    after = invariant_probe(model, raw[domain], support, task['family'], 'cuda')
    torch.save(cursor['best_model'], case/'best.tmp')
    (case/'best.tmp').replace(case/'best.pt')
    runtime.update(elapsed_seconds=time.monotonic()-began,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    return {'task': task, 'new_optimizer_steps': steps, 'epochs': cursor['epoch'],
            'source_optimizer_steps': cursor['step'], 'source_validation_scaled_MAE': cursor['best'],
            'validation_history': cursor['history'], 'source_objective_trajectory': cursor['losses'],
            'best_sha256': digest(case/'best.pt'), 'runtime': runtime,
            'invariance_before': before, 'invariance_after': after}


def freeze_target(project, code, report, case, task, plan, contract, source, progress):
    import numpy as np
    import torch
    from ectt_data import fit_scaler
    from transfer_train import adapt, forecast, invariant_probe, restore_model
    from transfer_worker import local_case, local_verification, raw_arrays, raw_paths, save_arrays

    began = time.monotonic()
    slug = task['dataset']
    budget = plan['recipe']['budget']
    profile = contract['datasets'][slug]
    ledger = profile['budgets'][budget]
    if not ledger['eligible_for_learned_adaptation'] or slug in task['sources']:
        raise ValueError('Invalid target support or exclusion')
    raw = raw_arrays(raw_paths(project, code, contract), [slug], contract)[slug]
    support = ledger['support_raw']
    scaler = fit_scaler(raw, support)
    model, runtime = model_for(project, code, contract, task, report)
    if digest(case/'best.pt') != source['best_sha256']:
        raise ValueError('Source checkpoint changed')
    restore_model(model, torch.load(case/'best.pt', map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    ident = baseline.fingerprint({'pilot': plan['identity'], 'task': task, 'runtime': runtime,
                                  'scaler': scaler, 'source': source['best_sha256']})
    recipe = plan['baseline']['recipe']
    head, cursor = adapt(model, raw, ledger['support_target_rows'], scaler, task['seed'], recipe,
                         contract['training_design'], 'cuda', progress, case/'target', ident)
    gates = invariant_probe(model, raw, support, task['family'], 'cuda', head)
    _, collection = baseline.foundation(project, code)
    row, directory = local_case(project, collection, task, budget)
    verification = ledger['verification_target_rows']
    test = profile['test_targets']
    selection, local_prefix = local_verification(project, code, raw, row, directory, verification)
    if selection['final_scaler'] != scaler or row['selected_family'] != selection['selected_family']:
        raise ValueError('Local comparator differs from the frozen support selection')
    local_file = inside(directory, directory/row['forecast_file'])
    with np.load(local_file, allow_pickle=False) as saved:
        origins = np.arange(*test)
        if not np.array_equal(saved['origins'], origins):
            raise ValueError('Local and transferred test origins differ')
        local_test = saved[row['forecast_columns']['support_selected_local']]
    residual = read(case/'target/latest.json')
    manifest = {'task': task, 'budget': budget, 'target_container_sha256': profile['container_sha256'],
                'scaler': scaler, 'support_raw': support, 'support_targets': ledger['support_target_rows'],
                'unique_support_values': (support[1]-support[0])*raw.shape[0],
                'support_scale': selection['support_scale'], 'local_family': row['selected_family'],
                'local_completion_sha256': digest(directory/'COMPLETE.json'),
                'source_checkpoint_sha256': source['best_sha256'], 'target_residual_sha256': residual['sha256'],
                'verification_targets': verification, 'test_targets': test,
                'verification_history_union': [verification[0]-12, verification[1]-1],
                'test_history_union': [test[0]-12, test[1]-1],
                'verification_labels_used_for_scoring': 0, 'test_labels_used_for_scoring': 0,
                'invariance': gates}
    write(case/'frozen_candidates.json', manifest)
    save_arrays(case/'verification_candidates.npz', origins=np.arange(*verification), local=local_prefix,
                transfer=forecast(model, raw, verification, scaler, recipe, 'cuda', head))
    save_arrays(case/'test_candidates.npz', origins=origins, local=local_test,
                transfer=forecast(model, raw, test, scaler, recipe, 'cuda', head))
    if (digest(case/'best.pt') != source['best_sha256']
            or digest(inside(case/'target', case/'target'/residual['file'])) != residual['sha256']):
        raise ValueError('Frozen weights changed during candidate prediction')
    runtime.update(elapsed_seconds=time.monotonic()-began,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    return {'optimizer_steps': cursor['step'], 'new_optimizer_steps': cursor['new_steps'], 'runtime': runtime}


def jobs(project, plan):
    state = read(project/'work'/JOURNAL/(plan['identity']+'.json'))
    if state['identity'] != plan['identity'] or set(state['jobs']) != {'review', 'probe', 'fit', 'report'}:
        raise ValueError('Incomplete pilot submission receipt')
    if any(not str(value).isdigit() for value in state['jobs'].values()):
        raise ValueError('Invalid pilot job identifier')
    return state['jobs']


def job_report(project, job, stage):
    directory = inside(project, project/'results/slurm_jobs'/job)
    status = read(directory/'job_status.json')
    if (status['stage'], status['execution_state'], status['shell_exit_code']) != (stage, 'COMPLETED', 0):
        raise ValueError('Required completed job missing: '+job)
    return directory/'report'


def collect(project, report, plan):
    scheduled = jobs(project, plan)
    root = store_path(project, plan)
    failures = {}
    try:
        review = job_report(project, scheduled['review'], REVIEW)
        manifest = read(review/'S01_packet/manifest.json')
        if manifest['pilot_identity'] != plan['identity'] or manifest['code_files'] != plan['reviewed_hashes']:
            raise ValueError('Review manifest differs from the pilot')
        for name, expected in plan['reviewed_hashes'].items():
            if digest(inside(review/'S01_packet', review/'S01_packet'/name)) != expected:
                raise ValueError('Review packet content differs: '+name)
    except (OSError, ValueError, KeyError) as exc:
        failures['review'] = str(exc)
    rows = []
    for task in plan['tasks']:
        index = str(task['index'])
        try:
            job_report(project, scheduled['fit']+'_'+index, FIT)
            case = root/'cases'/index
            if not completed(case, plan['identity']) or read(case/'result.json')['task'] != task:
                raise ValueError('Missing or incompatible frozen case')
            rows.append(index)
        except (OSError, ValueError, KeyError) as exc:
            failures[index] = str(exc)
    lines = ['# ECTT development pilot candidate preparation', '',
             f'Verified {len(rows)}/18 frozen cases.',
             'Anchored GPT-2 on PEMS08 and Beijing with the three matched source objectives.',
             'Verification and test forecasts were saved without scoring their labels.', '',
             f"S01 packet: {project/'results/slurm_jobs'/scheduled['review']/'report/S01_packet'}",
             f'Independent review disposition path: {approval_path(project, plan)}', '',
             'The CPU scoring stage uses the identical frozen verifier for all conditions.']
    lines.extend(f'{key}: {value}' for key, value in failures.items())
    (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return {'scientific_status': 'ECTT_PILOT_CANDIDATES_READY' if not failures else 'ECTT_PILOT_INCOMPLETE',
            'diagnostic_cases_checked': len(rows), 'failed': failures}


def run(project, code, report, stage):
    plan = foundation(project, code)
    contract, _ = load_contract(code)
    root = store_path(project, plan)
    state = {'stage': stage}

    def progress(**fields):
        state.update(fields)
        write(report/'transfer_progress.json', state)
        print('ECTT_PILOT_PROGRESS', state, flush=True)

    if stage == REVIEW:
        result = review_packet(project, code, report, plan, progress)
    elif stage == REPORT:
        result = collect(project, report, plan)
    elif stage == SCORE:
        from verification_pilot_score import score
        result = score(project, code, report, plan, contract, progress)
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
                result = train_source(project, code, report, case, task, plan, contract, progress, True)
                write(case/'result.json', result)
                finish(case, plan['identity'])
            else:
                for i in range(len(plan['probes'])):
                    probe = root/'probes'/str(i)
                    if (not completed(probe, plan['identity'])
                            or read(probe/'result.json')['scientific_status'] != 'ECTT_PILOT_RECOVERY_PASSED'):
                        raise ValueError('Matched objective recovery probe is missing')
                pointer = case/'source_result.json'
                if pointer.exists():
                    source = {**read(pointer), 'new_optimizer_steps': 0}
                    if source['task'] != task or digest(case/'best.pt') != source['best_sha256']:
                        raise ValueError('Saved source fit is incompatible')
                else:
                    source = train_source(project, code, report, case, task, plan, contract, progress, False)
                    write(pointer, source)
                target = freeze_target(project, code, report, case, task, plan, contract, source, progress)
                result = {'task': task, 'source': source, 'target': target,
                          'new_optimizer_steps': source['new_optimizer_steps']+target['new_optimizer_steps'],
                          'scientific_status': 'ECTT_PILOT_FROZEN_CASE_COMPLETE'}
                write(case/'result.json', result)
                finish(case, plan['identity'])
        if result['task'] != task:
            raise ValueError('Saved case task differs')
        (report/'SUMMARY.md').write_text(f'# ECTT pilot task\n\n{task}\n\nCompleted.\n')
    write(report/'transfer_result.json', {'method_evaluated': False, 'S01_independent_review': 'PENDING',
          **result, 'identity': plan['identity'], 'store': str(root)})
    progress(phase='FAILED' if result.get('failed') else 'COMPLETED')
    return 2 if result.get('failed') else 0


def dispatch(project, code, report, stage):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/verification_pilot.py'),
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
        (args.report/'SUMMARY.md').write_text(f'# ECTT pilot stage failed\n\n{exc}\n')
        write(args.report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
