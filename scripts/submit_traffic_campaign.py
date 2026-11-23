"""Submit or recover the registered aggregate-only traffic analysis on the UAEU HPC."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from job_notifications import preferences

RETRY = {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path.home()/'nature-llm')
    args = parser.parse_args()
    project = args.project.resolve()
    src = Path(__file__).resolve().parents[1]
    if not re.fullmatch('[0-9a-f]{40}',src.name) or src.parent != project/'work/code_versions':
        parser.error('Run the launcher from a verified project code snapshot')
    out = project/'results/traffic_publication'
    out.mkdir(parents=True,exist_ok=True)
    receipt = out/(src.name+'_submission.json')
    with (out/(src.name+'.submit.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        attempts = json.loads(receipt.read_text()) if receipt.exists() else []
        bundle = out/(src.name+'_report.tar.gz')
        if attempts:
            job = attempts[-1]
            if bundle.is_file():
                print('Aggregate report:',bundle)
                return
            states = subprocess.check_output(['sacct','-X','-n','-P','-j',job,'-o','JobIDRaw,State'],text=True)
            state = next((row.split('|')[1].split()[0].rstrip('+') for row in states.splitlines()
                          if row.split('|')[0]==job),'ACCOUNTING_PENDING')
            if state not in RETRY:
                print('Existing job',job,state,'log:',out/('campaign-'+job+'.log'))
                return
        py = project/'.venv-ectt-gpu/bin/python'
        if not py.is_file():
            raise ValueError('Missing existing project NumPy environment')
        command = ['sbatch','--parsable','--partition=longq','--nodes=1','--ntasks=1','--cpus-per-task=5',
                   '--mem=12G','--time=03:00:00','--job-name=ectt-paper','--output='+str(out/'campaign-%j.log')]
        notify = preferences(project)
        if notify:
            command += ['--mail-type=BEGIN,END,FAIL,TIME_LIMIT','--mail-user='+notify['mail_user']]
        line = shlex.join([str(py),'-u',str(src/'scripts/traffic_campaign.py'),'--project',str(project)])
        command += ['--wrap', 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; exec '+line]
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected Slurm job ID: '+job)
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(attempts+[job])+'\n')
        tmp.replace(receipt)
        print('Submitted CPU job',job,'\nLog:',out/('campaign-'+job+'.log'),'\nAggregate report:',bundle)


if __name__=='__main__':
    main()
