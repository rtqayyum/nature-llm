"""Submit one complete compact array and its dependent CPU collector."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
from ectt_common import read, write, inside
import compact_grid as grid


def _clock(seconds):
    return '%02d:%02d:%02d' % (seconds//3600, seconds%3600//60, seconds%60)


def _job_id(result):
    value = result.stdout.strip().split(';', 1)[0]
    if not value.isdigit():
        raise ValueError('Unexpected sbatch response; inspect scheduler before retrying: '+result.stdout)
    return value


def submit(project, code, retry=False):
    import slurm_submit as launcher
    from job_notifications import preferences, queued_receipt
    project, code = project.resolve(), code.resolve()
    inside(project, code)
    grid.foundation(project, code, full=False)
    matrix = grid.plan(code); recipe, _ = grid.config(code)
    launcher.prepare(project, code)
    if not shutil.which('sbatch'):
        raise ValueError('sbatch unavailable; run module load slurm on the HPC login node')
    settings = preferences(project)
    if not settings:
        raise ValueError('Compact array submissions require saved email preferences')
    journal = inside(project, project/'work/compact_grid_submissions'/(matrix['identity']+'.json'))
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(journal) if journal.exists() else {}
        if previous.get('array_job_id') and previous.get('report_job_id') and not retry:
            print('Existing compact submission retained; no duplicate jobs submitted.')
            show(project, previous); return 0
        # An unrelated project study cannot overlap this batch. Its own array
        # is allowed only while recovering a missing dependent-report submission.
        recorded_ids = {p.parent.name for p in (project/'results/slurm_jobs').glob('*/submission.json')
                        if p.parent.name.isdigit() and read(p).get('stage') in launcher.STUDY_STAGES}
        active = set()
        if recorded_ids:
            live = subprocess.run(['squeue', '--noheader', '--format=%A'], text=True, capture_output=True, timeout=20, check=True)
            active = recorded_ids.intersection(live.stdout.split())
        permitted = {previous.get('array_job_id')} if previous.get('array_job_id') and not previous.get('report_job_id') and not retry else set()
        if active-permitted:
            raise ValueError('Recorded study jobs are still active: '+','.join(sorted(active-permitted)))
        if retry and active:
            raise ValueError('Wait for the existing compact batch to end before retrying')
        if retry:
            previous = {'earlier_submissions': previous.get('earlier_submissions', [])+
                        [{k: v for k, v in previous.items() if k != 'earlier_submissions'}]}
        state = dict(previous, grid_identity=matrix['identity'], code=str(code))
        root = grid.store_path(project, code); root.mkdir(parents=True, exist_ok=True)
        write(root/'plan.json', matrix)

        def submit_one(stage, row, extra, environment, tasks=None):
            partition, limit, policy = launcher.allocation(stage)
            env = {**os.environ, 'TRAFFIC_PROJECT_ROOT': str(project), 'TRAFFIC_WORK_ROOT': str(project/'work'),
                   'TRAFFIC_CODE_DIR': str(code), 'TRAFFIC_STAGE': stage, **environment}
            command = ['sbatch', '--parsable', '--job-name='+row['job_name'], '--partition='+partition,
                       '--cpus-per-task='+str(row['cpus']), '--mem='+row['memory'], '--time='+_clock(limit),
                       '--mail-user='+settings['mail_user'], '--mail-type='+settings['mail_type'], '--export=ALL']
            if row['gpus']:
                command += ['--gres=gpu:1']
            command += extra+[str(project/'Batch')]
            # Keep an intent record before the external scheduler side effect.
            intent = project/'work/compact_grid_submissions'/(matrix['identity']+'.intent.json')
            write(intent, {'stage': stage, 'command': command, 'submitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            job_id = _job_id(result)
            # Persist the returned ID before email or dependent submission.
            state['array_job_id' if stage == grid.STAGE else 'report_job_id'] = job_id
            write(journal, state)
            submission = {'project': str(project), 'job_id': job_id, 'stage': stage, 'code': str(code),
                          'job_name': row['job_name'], 'partition': partition, 'cpus': row['cpus'], 'gpus': row['gpus'],
                          'time_limit': _clock(limit), 'partition_evidence': policy, 'command': command,
                          'submitted_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'grid_identity': matrix['identity'], 'notifications': settings}
            if tasks is not None:
                submission.update(array_tasks=tasks, max_concurrent_tasks=recipe['max_concurrent_tasks'])
            else:
                submission['source_array'] = state['array_job_id']
            directory = project/'results/slurm_jobs'/job_id
            write(directory/'submission.json', submission)
            if tasks is not None:
                for index, task in enumerate(tasks):
                    key = job_id+'_'+str(index)
                    write(project/'results/slurm_jobs'/key/'submission.json',
                          {**submission, 'job_id': key, 'array_job_id': job_id, 'task_index': index, **task})
            submission['queued_email'] = queued_receipt(project, submission, settings)
            write(directory/'submission.json', submission)
            print('EMAIL:', json.dumps(submission['queued_email']), flush=True)
            print('SUBMITTED Slurm job', job_id, 'name='+row['job_name'], 'stage='+stage,
                  'cpus='+str(row['cpus']), 'gpus='+str(row['gpus']), flush=True)
            return job_id

        if not state.get('array_job_id'):
            submit_one(grid.STAGE, recipe['gpu_resources'],
                       ['--array=0-%d%%%d' % (len(matrix['tasks'])-1, recipe['max_concurrent_tasks']),
                        '--output='+str(project/'logs/%x_%A_%a.out'), '--error='+str(project/'logs/%x_%A_%a.err')],
                       {'TRAFFIC_SOURCE_ARRAY': ''}, matrix['tasks'])
        if not state.get('report_job_id'):
            try:
                submit_one(grid.REPORT_STAGE, recipe['report_resources'], ['--dependency=afterany:'+state['array_job_id']],
                           {'TRAFFIC_SOURCE_ARRAY': state['array_job_id']})
            except Exception:
                print('Array', state['array_job_id'], 'is retained. Repeat the same command to submit only its missing collector.', flush=True)
                raise
        write(project/'work/LAST_COMPACT_GRID.json', state)
        (project/'work/LAST_JOB_ID').write_text(state['array_job_id']+'\n')
        show(project, state)
        return 0


def show(project, state):
    print('ARRAY_JOB_ID='+state.get('array_job_id', 'NOT_SUBMITTED'))
    print('REPORT_JOB_ID='+state.get('report_job_id', 'NOT_SUBMITTED'))
    print('Check: bash '+str(project/'Check')+' '+state.get('array_job_id', ''))
    if state.get('report_job_id'):
        print('Collection: '+str(project/'results/slurm_jobs'/state['report_job_id']/'report/SUMMARY.md'))


def summarize(project, array_id, submission, command):
    print('Compact local baseline array', array_id, 'tasks='+str(len(submission['array_tasks'])),
          'max_running='+str(submission['max_concurrent_tasks']))
    for arguments in (['squeue', '--jobs', array_id, '--array', '--noheader', '--format=%i|%T|%R'],
        ['sacct', '-j', array_id, '--parsable2', '--noheader', '--format=JobID%40,JobName,State%30,ExitCode,Elapsed,MaxRSS,Reason']):
        result = command(arguments); print(result['stdout'] or result['stderr'] or 'No current records')
    counts = {'COMPLETED': 0, 'FAILED': 0, 'QUEUED_OR_SETUP': 0, 'RUNNING': 0}
    for index, task in enumerate(submission['array_tasks']):
        key = array_id+'_'+str(index); report = project/'results/slurm_jobs'/key/'report'
        progress = read(report/'compact_progress.json') if (report/'compact_progress.json').exists() else {}
        phase = progress.get('phase', 'QUEUED_OR_SETUP')
        counts[phase if phase in counts else 'RUNNING'] += 1
        print(key, task['dataset'], 'budget='+task['budget'], 'seed='+str(task['seed']), phase,
              'step='+str(progress.get('optimizer_step', 0)), 'new_updates='+str(progress.get('new_optimizer_steps', 0)))
        if progress.get('error'):
            print('FAILURE:', progress['error'], 'traceback:', report/'failure_traceback.txt')
    print('APPLICATION_PROGRESS:', json.dumps(counts), '(Slurm accounting is authoritative for scheduler failures.)')
    journal = project/'work/compact_grid_submissions'/(submission['grid_identity']+'.json')
    if journal.exists():
        show(project, read(journal))
    print('Saved metrics will be collected after all array tasks finish; scientific analysis remains separate.')
