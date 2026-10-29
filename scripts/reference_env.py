#!/usr/bin/env python3
"""Create an isolated CPU reference environment, never the shared Conda base."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def ensure(project, code, report):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    if not (3, 8) <= sys.version_info[:2] <= (3, 11):
        raise RuntimeError("The pinned torch 2.0 / numpy 1.24 environment requires Python 3.8–3.11")
    environment = project / ".venv-reference-cpu"
    environment.resolve().relative_to(project)
    python = environment / "bin/python"
    requirement = code / "config/reference_cpu_requirements.txt"
    cache = project / ".cache/pip"
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(requirement.read_bytes() + b"torch==2.0.0+cpu").hexdigest()
    report.mkdir(parents=True, exist_ok=True)
    status = {"phase": "environment_check", "prefix": str(environment), "contract_sha256": fingerprint}
    status_path = report / "reference_environment_status.json"
    def phase(name):
        status["phase"] = name
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print("REFERENCE_ENV", name, flush=True)
    def execute(arguments, **kwargs):
        print("+", " ".join(map(str, arguments)), flush=True)
        subprocess.run(list(map(str, arguments)), check=True, **kwargs)
    phase("environment_check")
    if not python.exists():
        if shutil.disk_usage(project).free < 3 * 2**30:
            raise RuntimeError("At least 3 GiB of free project filesystem space is requested for the reference environment/cache")
        execute([sys.executable, "-m", "venv", "--copies", environment])
    execute([python, "-c", "import sys; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()", environment])
    check = "import torch,transformers,numpy,scipy,sklearn,tokenizers,safetensors,huggingface_hub; assert (torch.__version__,transformers.__version__,numpy.__version__,scipy.__version__,sklearn.__version__,tokenizers.__version__,safetensors.__version__,huggingface_hub.__version__)==('2.0.0+cpu','4.35.0','1.24.0','1.10.0','1.2.2','0.14.1','0.4.1','0.17.3')"
    probe = subprocess.run([str(python), "-c", check], capture_output=True, text=True)
    if probe.returncode:
        phase("install_torch_cpu")
        execute([python, "-m", "pip", "--isolated", "--cache-dir", cache, "install", "--no-user", "--disable-pip-version-check",
            "torch==2.0.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"])
        phase("install_reference_dependencies")
        execute([python, "-m", "pip", "--isolated", "--cache-dir", cache, "install", "--no-user", "--disable-pip-version-check", "-r", requirement])
    phase("verify_reference_versions")
    execute([python, "-c", check])
    execute([python, "-m", "pip", "check"])
    with (report / "reference_environment_freeze.txt").open("w") as stream:
        execute([python, "-m", "pip", "freeze"], stdout=stream)
    status["phase"] = "READY"
    status["python"] = str(python)
    status["freeze_sha256"] = hashlib.sha256((report / "reference_environment_freeze.txt").read_bytes()).hexdigest()
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return python
