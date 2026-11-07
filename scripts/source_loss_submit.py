"""Queue the complete paired loss development experiment with durable IDs."""
import fcntl
import json
import os
import shutil
import subprocess

from compact_grid_submit import _job_id
from ectt_common import read, write
from transfer_submit import clock, timestamp
import source_loss_development as experiment


def show(project, state):
    for key in ('probe', 'fit', 'report'):
        print(key.upper()+'_JOB_ID='+state['jobs'].get(key, 'NOT_SUBMITTED'))
    if state['jobs'].get('report'):
        print('Collection:', project/'results/slurm_jobs'/state['jobs']['report']/'report/SUMMARY.md')


def submit(project, code, retry=False):
    import slurm_submit as launcher
    from job_notifications import preferences, queued_receipt

    plan = experiment.foundation(project, code)
    launcher.prepare(project, code)
    if not shutil.which('sbatch'):
        raise ValueError('Load the Slurm module on the HPC login node')
    settings = preferences(project)
    if not settings:
        raise ValueError('Saved project notification preferences are required')
    journal = project/'work/source_loss_submissions'/(plan['identity']+'.json')
    intent = journal.with_suffix('.intent.json')
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(journal) if journal.exists() else {'identity': plan['identity'], 'jobs': {}, 'receipts': {}, 'previous_attempts': []}
        jobs = state['jobs']
        if intent.exists():
            pending = read(intent)
            if pending['key'] not in jobs:
                raise ValueError('Inspect the scheduler before retrying an ambiguous submission: '+str(intent))
            intent.unlink()
        steps = [('probe', experiment.PROBE, 'ectt-loss-probe', plan['probes']),
                 ('fit', experiment.FIT, 'ectt-loss-fit', plan['tasks']),
                 ('report', experiment.REPORT, 'ectt-loss-report', None)]

        def record(receipt):
            directory = project/'results/slurm_jobs'/receipt['job_id']
            if not (directory/'submission.json').exists():
                write(directory/'submission.json', receipt)
            for index, task in enumerate(receipt.get('array_tasks', [])):
                child = f"{receipt['job_id']}_{index}"
                path = directory.parent/child/'submission.json'
                if not path.exists():
                    write(path, {**receipt, 'job_id': child, 'array_job_id': receipt['job_id'],
                                 'task_index': index, 'task': task})

        for receipt in state['receipts'].values():
            record(receipt)
        if len(jobs) == 3 and not retry:
            show(project, state)
            return 0
        recorded = {p.parent.name for p in (project/'results/slurm_jobs').glob('*/submission.json')
                    if p.parent.name.isdigit() and read(p).get('stage') in launcher.STUDY_STAGES}
        live = subprocess.run(['squeue', '--noheader', '--format=%A', '--user='+str(os.getuid())],
                              capture_output=True, text=True, check=True, timeout=20).stdout.split()
        active = recorded.intersection(live)
        if active-set(jobs.values()) or (retry and active):
            raise ValueError('Wait for active project jobs: '+', '.join(sorted(active)))
        if retry:
            if len(jobs) != 3:
                raise ValueError('Complete the partial submission before requesting a retry')
            previous = read(project/'results/slurm_jobs'/jobs['report']/'job_status.json')
            if previous['execution_state'] == 'COMPLETED':
                show(project, state)
                return 0
            state['previous_attempts'].append(jobs)
            state['jobs'] = jobs = {}
            state['receipts'] = {}
            write(journal, state)
        for key, stage, name, tasks in steps:
            if key in jobs:
                continue
            partition, seconds, policy = launcher.allocation(stage)
            gpu = stage in experiment.GPU_STAGES
            command = ['sbatch', '--parsable', '--job-name='+name, '--partition='+partition,
                       '--cpus-per-task=5', '--mem='+('32G' if gpu else '8G'), '--time='+clock(seconds),
                       '--export=ALL', '--mail-user='+settings['mail_user'], '--mail-type='+settings['mail_type']]
            if tasks:
                command += ['--gres=gpu:1', f"--array=0-{len(tasks)-1}%{plan['recipe']['maximum_concurrent_gpu_tasks']}",
                            '--output='+str(project/'logs/%x_%A_%a.out'), '--error='+str(project/'logs/%x_%A_%a.err')]
            if key == 'fit':
                command += ['--dependency=afterok:'+jobs['probe'], '--kill-on-invalid-dep=yes']
            if key == 'report':
                command += ['--dependency=afterany:'+jobs['probe']+':'+jobs['fit']]
            command += [str(project/'Batch')]
            env = {**os.environ, 'TRAFFIC_CODE_DIR': str(code), 'TRAFFIC_STAGE': stage}
            write(intent, {'key': key, 'command': command, 'created_utc': timestamp()})
            try:
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as exc:
                if not (exc.stdout or '').strip():
                    intent.unlink()
                raise
            job_id = _job_id(result)
            jobs[key] = job_id
            receipt = {'project': str(project), 'job_id': job_id, 'stage': stage, 'job_name': name,
                       'code': str(code), 'partition': partition, 'cpus': 5, 'gpus': int(gpu),
                       'time_limit': clock(seconds), 'command': command, 'partition_evidence': policy,
                       'submitted_utc': timestamp(), 'matrix_identity': plan['identity'], 'notifications': settings}
            if tasks:
                receipt['array_tasks'] = tasks
            state['receipts'][key] = receipt
            write(journal, state)
            intent.unlink()
            record(receipt)
            directory = project/'results/slurm_jobs'/job_id
            receipt['queued_email'] = queued_receipt(project, receipt, settings)
            write(directory/'submission.json', receipt)
            print('SUBMITTED', job_id, name, flush=True)
        (project/'work/LAST_JOB_ID').write_text(jobs['report']+'\n')
        show(project, state)
        return 0


def summarize(project, job_id, submission, command):
    print('Paired development source loss array', job_id, submission['stage'])
    for cmd in (['squeue', '--jobs', job_id, '--array', '--noheader', '--format=%i|%T|%R'],
                ['sacct', '-j', job_id, '--parsable2', '--noheader',
                 '--format=JobID,State,ExitCode,Elapsed,MaxRSS,Reason']):
        result = command(cmd)
        print(result['stdout'] or result['stderr'])
    for index, task in enumerate(submission['array_tasks']):
        report = project/'results/slurm_jobs'/f'{job_id}_{index}'/'report/transfer_progress.json'
        progress = read(report) if report.exists() else {}
        print(f'{job_id}_{index}', task['family'], task['dataset'], task['arm'], task['seed'],
              progress.get('phase', 'QUEUED_OR_SETUP'), progress.get('optimizer_step', 0), progress.get('error', ''))
    state = read(project/'work/source_loss_submissions'/(submission['matrix_identity']+'.json'))
    show(project, state)
