"""Submit one checkpointable transfer batch and print the Slurm receipts."""
import datetime
import fcntl
import json
import os
import shutil
import subprocess

from ectt_common import read, write
from compact_grid_submit import _job_id
import transfer_plan as plan

STEPS = ('data', 'probe', 'source', 'target', 'report')
NAMES = ('ectt-xfer-data', 'ectt-xfer-probe', 'ectt-xfer-source', 'ectt-xfer-target', 'ectt-xfer-report')


def clock(seconds):
    days, left = divmod(seconds, 86400)
    return f'{days}-{left//3600:02}:{left//60%60:02}:{left%60:02}'


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def show(project, state):
    for key in STEPS:
        print(key.upper()+'_JOB_ID='+state['jobs'].get(key, 'NOT_SUBMITTED'))
    if state['jobs'].get('report'):
        print('Collection:', project/'results/slurm_jobs'/state['jobs']['report']/'report/SUMMARY.md')


def submit(project, code, retry=False):
    import slurm_submit as launcher
    from job_notifications import preferences, queued_receipt

    matrix = plan.plan(code)
    plan.foundation(project, code)
    launcher.prepare(project, code)
    if not shutil.which('sbatch'):
        raise ValueError('Run module load slurm on the HPC login node before submitting')
    settings = preferences(project)
    if not settings:
        raise ValueError('Transfer submissions require the saved project notification preferences')
    path = project/'work/transfer_submissions'/(matrix['identity']+'.json')
    intent = path.with_suffix('.intent.json')
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(path) if path.exists() else {'identity': matrix['identity'], 'jobs': {}, 'prior_attempts': []}
        ids = state['jobs']
        if intent.exists():
            pending = read(intent)
            if pending['key'] in ids:
                intent.unlink()
            else:
                raise ValueError(f"Inspect scheduler for ambiguous {pending['key']} submission before retrying. Intent: {intent}")
        jobs_root = project/'results/slurm_jobs'
        all_recorded = len(ids) == len(STEPS) and all(
            (jobs_root/ids[key]/'submission.json').exists()
            and all((jobs_root/f'{ids[key]}_{index}'/'submission.json').exists()
                    for index in range(len(matrix.get(key, []))))
            for key in STEPS)
        if all_recorded and not retry:
            print('Existing transfer submission retained.')
            show(project, state)
            return 0
        recorded = {p.parent.name for p in (project/'results/slurm_jobs').glob('*/submission.json')
                    if p.parent.name.isdigit() and read(p).get('stage') in launcher.STUDY_STAGES}
        live = subprocess.run(['squeue', '--noheader', '--format=%A', '--user='+str(os.getuid())],
                              capture_output=True, text=True, check=True, timeout=20).stdout.split()
        active = recorded.intersection(live)
        if active-set(ids.values()) or (retry and active):
            raise ValueError('Wait for active project study jobs: '+', '.join(sorted(active)))
        if retry:
            if len(ids) != len(STEPS):
                raise ValueError('Repeat the command without --retry-grid to finish the partial submission first')
            previous = read(project/'results/slurm_jobs'/ids['report']/'job_status.json')
            if previous['execution_state'] == 'COMPLETED':
                print('Verified transfer batch already complete.')
                show(project, state)
                return 0
            state['prior_attempts'].append(ids)
            state['jobs'] = ids = {}
            write(path, state)

        limits = {key: launcher.allocation(stage) for key, stage in zip(STEPS, plan.STAGES)}
        root = plan.store_path(project, matrix)
        write(root/'plan.json', matrix)

        def record(key, job_id, row):
            directory = project/'results/slurm_jobs'/job_id
            receipt = directory/'submission.json'
            if receipt.exists():
                row = read(receipt)
            else:
                write(receipt, row)
            for index, task in enumerate(matrix.get(key, [])):
                child = f'{job_id}_{index}'
                write(directory.parent/child/'submission.json',
                      {**row, 'job_id': child, 'array_job_id': job_id, 'task_index': index, 'task': task})
            return receipt

        for index, (key, stage, name) in enumerate(zip(STEPS, plan.STAGES, NAMES)):
            partition, limit, policy = limits[key]
            gpus = int(stage in plan.GPU_STAGES)
            tasks = matrix.get(key)
            options = []
            if gpus:
                count = min(len(tasks), matrix['recipe']['max_concurrent_gpu_tasks'])
                options += ['--gres=gpu:1', f'--array=0-{len(tasks)-1}%{count}',
                            '--output='+str(project/'logs/%x_%A_%a.out'),
                            '--error='+str(project/'logs/%x_%A_%a.err')]
            dependencies = {}
            if key == 'report':
                dependencies = dict(ids)
                dependencies.pop('report', None)
                options.append('--dependency=afterany:'+':'.join(dependencies.values()))
            elif index:
                parent = STEPS[index-1]
                dependencies = {parent: ids[parent]}
                options += ['--dependency=afterok:'+ids[parent], '--kill-on-invalid-dep=yes']
            cmd = ['sbatch', '--parsable', '--job-name='+name, '--partition='+partition, '--cpus-per-task=5',
                   '--mem='+('32G' if gpus else '8G'), '--time='+clock(limit), '--export=ALL',
                   '--mail-user='+settings['mail_user'], '--mail-type='+settings['mail_type'], *options, str(project/'Batch')]
            row = {'project': str(project), 'stage': stage, 'code': str(code), 'job_name': name, 'partition': partition,
                   'cpus': 5, 'gpus': gpus, 'time_limit': clock(limit), 'partition_evidence': policy,
                   'command': cmd, 'matrix_identity': matrix['identity'], 'dependencies': dependencies,
                   'submitted_utc': timestamp(), 'notifications': settings}
            if tasks is not None:
                row['array_tasks'] = tasks
            if key in ids:
                record(key, ids[key], {**row, 'job_id': ids[key]})
                continue
            env = {**os.environ, 'TRAFFIC_CODE_DIR': str(code), 'TRAFFIC_STAGE': stage,
                   'TRAFFIC_SOURCE_TIME_SECONDS': str(limits['source'][1]),
                   'TRAFFIC_TRANSFER_DEPENDENCIES': json.dumps(dependencies)}
            write(intent, {'key': key, 'command': cmd, 'created_utc': timestamp()})
            try:
                response = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as exc:
                if not (exc.stdout or '').strip():
                    intent.unlink()
                detail = (exc.stderr or '').strip()
                raise ValueError(f'sbatch rejected {key}: {detail}') from exc
            job_id = _job_id(response)
            ids[key] = job_id
            write(path, state)
            intent.unlink()
            row['job_id'] = job_id
            receipt = record(key, job_id, row)
            row['queued_email'] = queued_receipt(project, row, settings)
            write(receipt, row)
            print('EMAIL:', json.dumps(row['queued_email']), flush=True)
            print(f'SUBMITTED {job_id} {name}', flush=True)
        (project/'work/LAST_JOB_ID').write_text(ids['report']+'\n')
        show(project, state)
        return 0


def summarize(project, job_id, submission, command):
    rows = submission['array_tasks']
    print('Project transfer array', job_id, submission['stage'], 'tasks='+str(len(rows)))
    live = command(['squeue', '--jobs', job_id, '--array', '--noheader', '--format=%i|%T|%R'])
    accounting = command(['sacct', '-j', job_id, '--parsable2', '--noheader',
                          '--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,Reason'])
    print(live['stdout'] or 'No active queue record')
    print(accounting['stdout'] or accounting['stderr'])
    for idx, task in enumerate(rows):
        child = f'{job_id}_{idx}'
        report = project/'results/slurm_jobs'/child/'report/transfer_progress.json'
        progress = read(report) if report.exists() else {}
        print(child, task['family'], task.get('dataset', task.get('fold')), 'seed='+str(task['seed']),
              progress.get('phase', 'QUEUED_OR_SETUP'), 'step='+str(progress.get('optimizer_step', 0)),
              progress.get('error', ''))
    path = project/'work/transfer_submissions'/(submission['matrix_identity']+'.json')
    if path.exists():
        show(project, read(path))
