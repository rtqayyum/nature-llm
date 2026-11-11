"""Replay the frozen development verifier as an exploratory comparison."""
import argparse
import csv
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from ectt_common import completed, digest, finish, inside, load_contract, read, write
from transfer_plan import fingerprint
import verification_pilot as pilot
from verification_pilot_candidate_audit import admission


STAGE = 'shared-transfer-ectt-exploratory-replay'


def summary(plan, comparison):
    lines = ['# Exploratory ECTT development replay', '',
             'All 18 frozen cases replay the same locked verifier. Independent S01 review is pending.', '',
             '| Domain | Source objective | Local MAE | Always-transfer MAE | Replay MAE | Accepted | Mean verification rows read |',
             '|---|---|---:|---:|---:|---:|---:|']
    for domain in plan['recipe']['development_targets']:
        for arm in plan['recipe']['arms']:
            row = comparison['groups'][domain][arm]
            lines.append(f"| {domain} | {arm} | {row['local_MAE']:.3f} | {row['unconditional_transfer_MAE']:.3f} | "
                         f"{row['deployed_MAE']:.3f} | {row['accepted']}/{len(plan['recipe']['seeds'])} | "
                         f"{row['verification_origins_read']:.1f} |")
    lines += ['', 'Per-seed errors, label use and harm are in exploratory_metrics.csv.',
              'Paired descriptive comparisons are in exploratory_comparison.json.',
              'Nonacceptance retains the local forecast and counts the available prefix plus one for restricted evidence time.',
              'The replay is an internal development diagnostic. Confirmatory analysis requires independent review.', '']
    return '\n'.join(lines)


def run(project, code, report):
    import numpy as np
    from transfer_worker import raw_arrays, raw_paths
    from verification_pilot_score import compare, decide, frozen_cases, score_test

    plan = admission(project, code)
    contract, _ = load_contract(code)
    identity = fingerprint({'pilot': plan['identity'], 'scope': 'exploratory',
                            'implementation': digest(code/'scripts/verification_pilot_exploratory.py')})
    store = inside(project, pilot.store_path(project, plan)/'exploratory_replays'/identity)
    store.mkdir(parents=True, exist_ok=True)
    outputs = ('SUMMARY.md', 'exploratory_metrics.csv', 'exploratory_comparison.json', 'decisions.json')

    def progress(phase, count, total):
        state = {'phase': phase, 'cases_complete': count, 'cases_total': total, 'S01_independent_review': 'PENDING'}
        write(report/'transfer_progress.json', state)
        print('ECTT_EXPLORATORY_PROGRESS', state, flush=True)

    with (store/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not completed(store, identity):
            cases = frozen_cases(project, plan, contract)
            raw = raw_arrays(raw_paths(project, code, contract), plan['recipe']['development_targets'], contract)
            decisions = {}
            for case in cases:
                task = case['task']
                decisions[str(task['index'])] = decide(case, raw[task['dataset']], contract)
                progress('VERIFICATION_REPLAY', len(decisions), len(cases))
            decision_file = store/'decisions.json'
            decision_record = {'identity': identity, 'evaluation_scope': 'EXPLORATORY_DEVELOPMENT',
                               'S01_independent_review': 'PENDING', 'decisions': decisions}
            if decision_file.exists() and read(decision_file) != decision_record:
                raise ValueError('Previously frozen exploratory decisions differ')
            write(decision_file, decision_record)
            decision_sha = digest(decision_file)
            rows, errors, truths = [], {}, {}
            for case in cases:
                task = case['task']
                domain = task['dataset']
                start, end = case['manifest']['test_targets']
                if domain not in truths:
                    truths[domain] = np.asarray(raw[domain][:, start:end].T, dtype=np.float64)
                row, values = score_test(case, truths[domain], decisions[str(task['index'])], contract)
                row['evaluation_scope'] = 'EXPLORATORY_DEVELOPMENT'
                row['restricted_first_threshold_crossing'] = row.pop('restricted_first_certificate')
                rows.append(row)
                errors[domain, task['seed'], task['arm']] = values
                progress('TEST_SCORING', len(rows), len(cases))
            if digest(decision_file) != decision_sha:
                raise ValueError('Frozen exploratory decisions changed during test scoring')
            with (store/'exploratory_metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            comparison = compare(rows, errors, plan, contract)
            comparison['exploratory_screen_passed'] = comparison.pop('supports_held_out_design_review')
            comparison.update(evaluation_scope='EXPLORATORY_DEVELOPMENT', S01_independent_review='PENDING')
            write(store/'exploratory_comparison.json', comparison)
            write(store/'transfer_result.json', {'scientific_status': 'ECTT_DEVELOPMENT_EXPLORATORY_REPLAY_COMPLETE',
                  'identity': plan['identity'], 'evaluation_identity': identity, 'evaluation_scope': 'EXPLORATORY_DEVELOPMENT',
                  'method_evaluated': False, 'S01_independent_review': 'PENDING', 'diagnostic_cases_checked': len(rows),
                  'decisions_sha256': decision_sha, 'new_optimizer_steps': 0, 'evaluation_store': str(store),
                  'comparison': comparison})
            (store/'SUMMARY.md').write_text(summary(plan, comparison))
            finish(store, identity)
        for name in (*outputs, 'transfer_result.json'):
            shutil.copyfile(store/name, report/name)
        progress('COMPLETED', len(plan['tasks']), len(plan['tasks']))
    return 0


def dispatch(project, code, report):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/verification_pilot_exploratory.py'),
                            str(project), str(code), str(report)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('project', 'code', 'report'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    report = inside(project, args.report)
    try:
        sys.exit(run(project, inside(project, args.code), report))
    except Exception as exc:
        traceback.print_exc()
        (report/'SUMMARY.md').write_text(f'# Exploratory replay failed\n\n{exc}\n')
        write(report/'transfer_progress.json', {'phase': 'FAILED', 'error': str(exc)})
        sys.exit(2)
