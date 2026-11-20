"""Freeze matched GPT-2 candidates on the six primary-budget held-out targets."""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
import transfer_plan as baseline
import verification_pilot as pilot
import verification_window_campaign as windows

LAUNCH = 'shared-transfer-ectt-heldout-prepare'
PROBE = 'shared-transfer-ectt-heldout-preflight'
FIT = 'shared-transfer-ectt-heldout-fit'
REPORT = 'shared-transfer-ectt-heldout-report'
STAGES = (PROBE, FIT, REPORT)
GPU_STAGES = (FIT,)
JOURNAL = 'heldout_candidate_submissions'
PROTOCOL = 'research/HELDOUT_GPT2_V1.md'
TWO_THIRDS = 'fixed_two_thirds_mean'


def two_thirds_rows(n):
    if n < 1:
        raise ValueError('Expected a positive verification length')
    return (2*n+2)//3


def policy_decision(rule, origins, local, transfer, scale, matured, settings):
    """Use the same callback for matured labels. Only the fixed look is new."""
    if rule == TWO_THIRDS:
        n = two_thirds_rows(len(origins))
        return windows.alternative('fixed_budget_mean', origins[:n], local[:n], transfer[:n],
                                   scale, matured, **settings)
    if rule == 'locked_two_sided':
        from frozen_verifier_replay import replay
        return replay(origins, local, transfer, scale, matured, **settings)
    if rule not in windows.RULES:
        raise ValueError('Unknown held-out decision rule')
    return windows.alternative(rule, origins, local, transfer, scale, matured, **settings)


def matrix(code):
    contract, _ = load_contract(code)
    parent = pilot.matrix(code)
    recipe = parent['recipe']
    targets = contract['confirmatory_targets']
    tasks = []
    for row in parent['baseline']['source']:
        domains = [slug for slug in targets if slug in row['excluded']]
        if not domains or row['family'] != recipe['family']:
            continue
        for arm in recipe['arms']:
            tasks.append({**row, 'index': len(tasks), 'dataset': domains[0], 'targets': domains,
                          'baseline_source_index': row['index'], 'arm': arm})
    expected = {(d, s, a) for d in targets for s in recipe['seeds'] for a in recipe['arms']}
    keys = [(d, t['seed'], t['arm']) for t in tasks for d in t['targets']]
    if set(keys) != expected or len(keys) != 54 or len(tasks) != 45:
        raise ValueError('Incomplete or duplicate held-out design')
    for task in tasks:
        if set(task['excluded']) & set(task['sources']) or set(task['targets']) != set(task['excluded']):
            raise ValueError('Source fold violates held-out exclusions')
    for domain in targets:
        profile = contract['datasets'][domain]
        ledger = profile['budgets'][recipe['budget']]
        start, stop = ledger['verification_target_rows']
        if (not ledger['eligible_for_learned_adaptation'] or
                not ledger['support_raw'][1] <= start < stop <= profile['test_targets'][0] or
                two_thirds_rows(stop-start) < contract['partitions']['minimum_verification_origins']):
            raise ValueError('Ineligible fixed-budget target '+domain)
    files = {**parent['reviewed_hashes']}
    for name in (PROTOCOL, 'scripts/heldout_candidates.py', 'scripts/verification_window_campaign.py',
                 'research/MASTER_PROTOCOL_V1.md', 'config/ectt_execution_v1.json'):
        files[name] = digest(code/name)
    identity = baseline.fingerprint({'pilot': parent['identity'], 'files': files, 'tasks': tasks})
    return {'identity': identity, 'reviewed_hashes': files, 'recipe': recipe, 'tasks': tasks,
            'probes': [], 'baseline': parent['baseline'], 'pilot_identity': parent['identity']}


def foundation(project, code):
    plan = matrix(code)
    previous = windows.foundation(project, code)
    report = pilot.job_report(project, '3100357', windows.REPORT)
    result = read(report/'transfer_result.json')
    if (result.get('identity') != previous['identity'] or
            result.get('scientific_status') != 'ECTT_WINDOW_CAMPAIGN_COMPLETE' or
            result.get('diagnostic_cases_checked') != 216 or result.get('missing_tasks') != []):
        raise ValueError('Complete development window collection 3100357 is required')
    return plan


def submission_steps(plan):
    return [('probe', PROBE, 'ectt-hold-pre', None), ('fit', FIT, 'ectt-hold-fit', plan['tasks']),
            ('report', REPORT, 'ectt-hold-report', None)]


def preflight(project, code, report, plan, contract):
    packet = report/'S01_packet'
    pilot_plan = pilot.matrix(code)
    pilot_jobs = pilot.jobs(project, pilot_plan)
    for index, task in enumerate(pilot_plan['probes']):
        pilot.job_report(project, pilot_jobs['probe']+'_'+str(index), pilot.PROBE)
        case = pilot.store_path(project, pilot_plan)/'probes'/str(index)
        probe_result = read(case/'result.json')
        if (not completed(case, pilot_plan['identity']) or probe_result['task'] != task or
                probe_result['scientific_status'] != 'ECTT_PILOT_RECOVERY_PASSED'):
            raise ValueError('Prior recovery probe differs from the pinned objective')
    for name, expected in plan['reviewed_hashes'].items():
        destination = packet/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code/name, destination)
        if digest(destination) != expected:
            raise ValueError('Review input changed: '+name)
    ledgers = {}
    for domain in contract['confirmatory_targets']:
        profile = contract['datasets'][domain]
        ledger = profile['budgets'][plan['recipe']['budget']]
        bounds = ledger['verification_target_rows']
        ledgers[domain] = {**ledger, 'test_targets': profile['test_targets'],
                           'two_thirds_rows': two_thirds_rows(bounds[1]-bounds[0])}
    write(packet/'manifest.json', {'identity': plan['identity'], 'files': plan['reviewed_hashes'],
          'tasks': plan['tasks'], 'ledgers': ledgers, 'rules': [*windows.RULES, TWO_THIRDS],
          'development_collection': '3100357', 'independent_review_complete': False,
          'scope': 'candidate preparation and prospective policy specification'})
    (report/'SUMMARY.md').write_text('# Held-out preflight\n\n45 source fits and 54 target pairs registered.\n'
                                     'S01 packet prepared. Independent review is pending.\n')
    return {'scientific_status': 'HELDOUT_PREFLIGHT_COMPLETE'}


def fit_group(project, code, report, root, task, plan, contract, progress):
    group = root/'groups'/str(task['index'])
    group.mkdir(parents=True, exist_ok=True)
    with (group/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = group/'source_result.json'
        steps = 0
        if pointer.exists():
            source = read(pointer)
            if source['task'] != task or digest(group/'best.pt') != source['best_sha256']:
                raise ValueError('Saved source checkpoint differs')
        else:
            source = pilot.train_source(project, code, report, group, task, plan, contract, progress, False)
            write(pointer, source)
            steps = source['new_optimizer_steps']
        for domain in task['targets']:
            target = {**task, 'dataset': domain}
            case = group/domain
            case.mkdir(exist_ok=True)
            if completed(case, plan['identity']):
                if read(case/'result.json')['task'] != target:
                    raise ValueError('Completed target task differs')
                continue
            if not (case/'best.pt').exists():
                os.link(group/'best.pt', case/'best.pt')
            outcome = pilot.freeze_target(project, code, report, case, target, plan, contract, source, progress)
            write(case/'result.json', {'task': target, 'target': outcome, 'source_sha256': source['best_sha256']})
            finish(case, plan['identity'])
            steps += outcome['new_optimizer_steps']
    (report/'SUMMARY.md').write_text(f"# Frozen held-out forecasts\n\n{task['fold']} / {task['arm']} / seed {task['seed']}\n")
    return {'scientific_status': 'HELDOUT_GROUP_FROZEN', 'new_optimizer_steps': steps,
            'frozen_cases': len(task['targets'])}


def collect(project, report, root, plan, contract):
    from verification_pilot_score import candidate_arrays
    import numpy as np

    jobs = read(project/'work'/JOURNAL/(plan['identity']+'.json'))['jobs']
    failures, inventory, references = [], [], {}
    try:
        packet = pilot.job_report(project, jobs['probe'], PROBE)/'S01_packet'
        manifest = read(packet/'manifest.json')
        if (manifest['identity'] != plan['identity'] or manifest['files'] != plan['reviewed_hashes']
                or manifest['tasks'] != plan['tasks']
                or any(digest(packet/name) != sha for name, sha in plan['reviewed_hashes'].items())):
            raise ValueError('Preflight manifest changed')
    except (OSError, ValueError, KeyError) as exc:
        failures.append({'stage': PROBE, 'error': str(exc)})
    for task in plan['tasks']:
        index = str(task['index'])
        group = root/'groups'/index
        for domain in task['targets']:
            try:
                pilot.job_report(project, jobs['fit']+'_'+index, FIT)
                case = group/domain
                if not completed(case, plan['identity']):
                    raise ValueError('Missing candidate artifacts')
                meta = read(case/'frozen_candidates.json')
                expected = {**task, 'dataset': domain}
                source = read(group/'source_result.json')
                result = read(case/'result.json')
                profile = contract['datasets'][domain]
                ledger = profile['budgets'][plan['recipe']['budget']]
                hashes = read(case/'COMPLETE.json')['artifacts']
                residual = read(case/'target/latest.json')
                if (meta['task'] != expected or result['task'] != expected or source['task'] != task
                        or meta['source_checkpoint_sha256'] != source['best_sha256']
                        or result['source_sha256'] != hashes['best.pt'] or hashes['best.pt'] != source['best_sha256']
                        or residual['sha256'] != meta['target_residual_sha256']
                        or hashes['target/'+residual['file']] != residual['sha256']
                        or meta['support_targets'] != ledger['support_target_rows']
                        or meta['target_container_sha256'] != profile['container_sha256']
                        or result['target']['optimizer_steps'] != contract['training_design']['target_updates']
                        or meta['support_raw'] != ledger['support_raw'] or meta['budget'] != plan['recipe']['budget']
                        or meta['verification_labels_used_for_scoring'] != 0 or meta['test_labels_used_for_scoring'] != 0):
                    raise ValueError('Invalid candidate lineage')
                key = (domain, task['seed'])
                comparator = [meta['local_family'], meta['scaler'], meta['support_scale'], meta['local_completion_sha256']]
                for prefix, bounds in [('verification', ledger['verification_target_rows']), ('test', profile['test_targets'])]:
                    if meta[prefix+'_targets'] != bounds:
                        raise ValueError('Invalid forecast origins')
                    arrays = candidate_arrays(case/(prefix+'_candidates.npz'), bounds, profile['shape'][0])
                    comparator.append(hashlib.sha256(np.ascontiguousarray(arrays['local']).tobytes()).hexdigest())
                if references.setdefault(key, comparator) != comparator:
                    raise ValueError('Local comparator differs across objectives')
                inventory.append({'task': expected, 'case': str(case), 'manifest_sha256': digest(case/'COMPLETE.json')})
            except (OSError, ValueError, KeyError) as exc:
                failures.append({'group': index, 'domain': domain, 'error': str(exc)})
    write(report/'candidate_inventory.json', {'identity': plan['identity'], 'records': inventory, 'failures': failures})
    lines = ['# Held-out GPT-2 candidate preparation', '', f'Verified {len(inventory)}/54 frozen target pairs.',
             'Verification and test scoring await independent S01 review.',
             'Review packet '+str(project/'results/slurm_jobs'/jobs['probe']/'report/S01_packet')]
    lines.extend(str(row) for row in failures)
    (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return {'scientific_status': 'HELDOUT_CANDIDATES_INCOMPLETE' if failures else 'HELDOUT_CANDIDATES_READY',
            'diagnostic_cases_checked': len(inventory), 'failed': failures}


def run(project, code, report, stage):
    plan = foundation(project, code)
    contract, _ = load_contract(code)
    root = inside(project, project/'outputs/ectt/heldout_gpt2'/plan['identity'])

    def progress(**fields):
        write(report/'transfer_progress.json', fields)

    if stage == PROBE:
        result = preflight(project, code, report, plan, contract)
    elif stage == REPORT:
        result = collect(project, report, root, plan, contract)
    elif stage == FIT:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if not 0 <= index < len(plan['tasks']):
            raise ValueError('Unexpected held-out array index')
        result = fit_group(project, code, report, root, plan['tasks'][index], plan, contract, progress)
    else:
        raise ValueError('Unknown held-out stage')
    write(report/'transfer_result.json', {**result, 'identity': plan['identity'], 'method_evaluated': False,
          'verification_labels_scored': 0, 'test_labels_scored': 0, 'S01_independent_review': 'PENDING'})
    progress(phase='FAILED' if result.get('failed') else 'COMPLETED')
    return 2 if result.get('failed') else 0


def dispatch(project, code, report, stage):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/heldout_candidates.py'),
                            str(project), str(code), str(report), stage],
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


if __name__ == '__main__':
    project, code, report = map(Path, sys.argv[1:4])
    try:
        sys.exit(run(project, inside(project, code), inside(project, report), sys.argv[4]))
    except Exception as exc:
        traceback.print_exc()
        (report/'SUMMARY.md').write_text(f'# Held-out preparation failed\n\n{exc}\n')
        write(report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
