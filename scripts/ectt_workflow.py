#!/usr/bin/env python3
"""Manual sequential admission workflow; completion markers are verified."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from ectt_common import read, write, inside, identity, load_contract, completed, finish, digest

STAGES = ('ectt-preflight', 'ectt-stat-check', 'ectt-gpu-pilot')


def stage_store(project, code, stage):
    return inside(project, project / 'outputs/ectt/admission' / identity(code) / stage)


def require_previous(project, code, stage):
    for previous in STAGES[:STAGES.index(stage)]:
        if not completed(stage_store(project, code, previous), identity(code)):
            raise ValueError('Required stage not completed for this code/contract: '+previous)


def next_stage(project, code):
    for stage in STAGES:
        if not completed(stage_store(project, code, stage), identity(code)):
            return stage
    raise ValueError('Implemented admission complete. S01 independent review and benchmark training kernels '+
                     'remain pending; no unimplemented experiment was submitted.')


def show(report):
    for name in ('ectt_environment_status.json', 'ectt_progress.json', 'ectt_store.json'):
        path = report / name
        if path.exists():
            print('ECTT '+name+':', json.dumps(read(path), indent=2))
    path = report / 'ectt_result.json'
    if path.exists():
        print('ECTT RESULT:', json.dumps({k: v for k, v in read(path).items() if k not in ('datasets', 'scenarios')}, indent=2))
    failure = report / 'failure_traceback.txt'
    if failure.exists():
        print('ECTT FAILURE:', '\n'.join(failure.read_text().splitlines()[-10:]))
    print('Benchmark experiments completed: 0 (admission/engineering only)')


def run(project, code, report, stage):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    inside(project, code)
    inside(project, report)
    c, e = load_contract(code)
    require_previous(project, code, stage)
    store = stage_store(project, code, stage)
    store.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    ident = identity(code)
    state = {'stage': stage, 'identity': ident, 'phase': 'STARTING', 'training_performed': False,
             'benchmark_experiments_completed': 0, 'store': str(store)}
    cache_valid = False
    write(report / 'ectt_store.json', {'identity': ident, 'store': str(store)})
    def progress(**values):
        state.update(values, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(report / 'ectt_progress.json', state)
        if not cache_valid:
            write(store / 'progress.json', state)
        print('ECTT_PROGRESS', json.dumps(state), flush=True)
    with (store / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Exact study stage already has a worker; no duplicate work')
        try:
            cache_valid = completed(store, ident)
            if cache_valid:
                result = read(store / 'result.json')
                if stage == 'ectt-preflight':
                    # Integrity reads stay inside Slurm, not the login launcher.
                    for row in result['datasets'].values():
                        path = inside(project, Path(row['source_path']))
                        if digest(path) != row['container_sha256']:
                            raise ValueError('Canonical asset changed after prior admission: '+str(path))
                result.update(reused_verified_stage=True, training_performed=False)
                if 'fit' in result:
                    result['fit'] = {**result['fit'], 'new_optimizer_steps': 0,
                                     'resumed_from_step': result['fit'].get('optimizer_steps', 0)}
                state.update(phase='COMPLETED', reused_verified_stage=True)
                write(report / 'ectt_progress.json', state)
            else:
                progress(phase='PREPARING_REGISTRY')
                from ectt_registry import save
                matrix = save(code, store)
                write(store / 'contract.json', {'identity': ident, 'scientific': c, 'execution': e})
                if stage == 'ectt-preflight':
                    from ectt_data import run as data_run
                    result = data_run(project, code, store, c, progress)
                elif stage == 'ectt-stat-check':
                    from ectt_statistics import simulate
                    result = simulate(store, e['statistical_simulation'], c['verifier'], progress)
                else:
                    from ectt_pilot import run as pilot_run
                    result = pilot_run(project, code, store, c, e, ident, progress)
                progress(phase='COMPLETED', training_performed=result.get('training_performed', False),
                         family_registry_count=len(matrix['families']))
                finish(store, ident)
            write(report / 'ectt_result.json', result)
            shutil.copyfile(store / 'SUMMARY.md', report / 'SUMMARY.md')
            print((report / 'SUMMARY.md').read_text(), flush=True)
            print('ECTT_STORE', store, flush=True)
            return 0
        except Exception as exc:
            progress(phase='FAILED', error=str(exc))
            (report / 'failure_traceback.txt').write_text(traceback.format_exc())
            (report / 'SUMMARY.md').write_text('# ECTT stage failed\n\n'+str(exc)+
                '\n\nPartial artifacts preserved: '+str(store)+'\nRecovery: resubmit the same stage after resolving its cause.\n')
            print('ECTT_FAILED', str(exc), flush=True)
            return 2


def dispatch(project, code, report, stage):
    load_contract(code)
    require_previous(project, code, stage)
    if stage == 'ectt-gpu-pilot':
        from ectt_env import ensure
        with (project / 'work/ectt_environment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            python = ensure(project, code, report)
    else:
        python = inside(project, project / '.venv/bin/python')
        if not python.exists():
            raise RuntimeError('Existing project acquisition .venv is missing; shared Conda is never modified')
    return subprocess.call([str(python), '-u', str(code / 'scripts/ectt_workflow.py'),
        'run', '--project', str(project), '--code', str(code), '--report', str(report), '--stage', stage])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['plan', 'next', 'run'])
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--report', type=Path)
    p.add_argument('--stage', choices=STAGES)
    args = p.parse_args(argv)
    if args.mode == 'next':
        print(next_stage(args.project.resolve(), args.code.resolve()))
        return 0
    if args.mode == 'plan':
        from ectt_registry import build
        print(json.dumps(build(args.code), indent=2))
        return 0
    if not args.report or not args.stage or not os.environ.get('SLURM_JOB_ID', '').isdigit():
        p.error('run requires --report, --stage and a real Slurm allocation; submit through the bootstrap')
    return run(args.project, args.code, args.report, args.stage)


if __name__ == '__main__':
    sys.exit(main())
