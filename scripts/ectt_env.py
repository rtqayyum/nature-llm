"""Separate project-local CUDA environment for numerical Qwen admission."""
import os
import shutil
import subprocess
import sys
from ectt_common import inside, digest, write


def ensure(project, code, report):
    if not (3, 9) <= sys.version_info[:2] <= (3, 11):
        raise RuntimeError('Use Python 3.9–3.11 for the pinned Qwen environment')
    environment = inside(project, project / '.venv-ectt-gpu')
    python = environment / 'bin/python'
    requirement = inside(project, code / 'config/ectt_gpu_requirements.txt')
    cache = inside(project, project / '.cache/pip')
    cache.mkdir(parents=True, exist_ok=True)
    status_path = report / 'ectt_environment_status.json'
    def execute(args, **kwargs):
        print('+', ' '.join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), check=True, **kwargs)
    write(status_path, {'phase': 'ENVIRONMENT_CHECK', 'prefix': str(environment)})
    if not python.exists():
        if shutil.disk_usage(project).free < 12*2**30:
            raise RuntimeError('Need at least 12 GiB free project filesystem space for CUDA environment/cache')
        execute([sys.executable, '-m', 'venv', '--copies', environment])
    execute([python, '-c', 'import sys;from pathlib import Path;assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()', environment])
    check = ("import torch,transformers,peft,numpy,huggingface_hub,safetensors,tokenizers,accelerate;"
        "assert (torch.__version__,transformers.__version__,peft.__version__,numpy.__version__,huggingface_hub.__version__,"
        "safetensors.__version__,tokenizers.__version__,accelerate.__version__)=="
        "('2.4.1+cu118','4.45.2','0.13.2','1.26.4','0.26.2','0.4.5','0.20.3','1.0.1')")
    probe = subprocess.run([str(python), '-c', check], capture_output=True, text=True)
    if probe.returncode:
        write(status_path, {'phase': 'INSTALLING_PRIVATE_ENVIRONMENT', 'prefix': str(environment)})
        execute([python, '-m', 'pip', '--isolated', '--cache-dir', cache, 'install', '--no-user',
            '--disable-pip-version-check', 'torch==2.4.1+cu118', '--index-url', 'https://download.pytorch.org/whl/cu118'])
        execute([python, '-m', 'pip', '--isolated', '--cache-dir', cache, 'install', '--no-user',
            '--disable-pip-version-check', '-r', requirement])
    execute([python, '-c', check])
    execute([python, '-m', 'pip', 'check'])
    freeze = report / 'ectt_environment_freeze.txt'
    with freeze.open('w') as stream:
        execute([python, '-m', 'pip', 'freeze'], stdout=stream)
    write(status_path, {'phase': 'READY', 'prefix': str(environment), 'python': str(python),
          'requirements_sha256': digest(requirement), 'freeze_sha256': digest(freeze)})
    return python
