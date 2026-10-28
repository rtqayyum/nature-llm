#!/usr/bin/env python3
"""Create an isolated audit environment, record HPC facts, acquire and audit."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    for name in ("project", "work", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    lock_file = (args.work / "acquisition.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another acquisition is already running in this work directory.", flush=True)
        return 2
    status = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "state": "RUNNING"}
    exit_code = 1
    try:
        if not shutil.which("git"):
            raise RuntimeError("git is not available. Load your HPC git module and rerun.")
        facts = {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
                 "work_root": str(args.work), "free_bytes": shutil.disk_usage(args.work).free,
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "slurm_available": bool(shutil.which("sbatch")), "stage": "CPU download and audit only"}
        for name, command in (("nvidia_smi", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
                              ("quota", ["quota", "-s"]),
                              ("code_commit", ["git", "-C", str(args.project), "rev-parse", "HEAD"])):
            try:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
                facts[name] = {"exit_code": result.returncode, "output": result.stdout[:6000]}
            except (OSError, subprocess.TimeoutExpired) as exc:
                facts[name] = {"unavailable": type(exc).__name__}
        (args.report / "hpc_environment.json").write_text(json.dumps(facts, indent=2))
        print("CPU acquisition only. No reference or Qwen training is started.", flush=True)
        env_dir = args.work / "audit_env"
        python = env_dir / "bin" / "python"
        if not python.exists():
            run([sys.executable, "-m", "venv", env_dir])
        numpy_version = "1.24.4" if sys.version_info[:2] == (3, 8) else "1.26.4"
        probe = subprocess.run([str(python), "-c", "import numpy; assert numpy.__version__ == " + repr(numpy_version)])
        if probe.returncode:
            run([python, "-m", "pip", "install", "--disable-pip-version-check", "numpy==" + numpy_version])
        with (args.report / "audit_environment.txt").open("w") as stream:
            run([python, "-m", "pip", "freeze"], stdout=stream)
        command = [python, "-u", args.project / "scripts" / "acquire.py", "--project", args.project,
                   "--work", args.work, "--report", args.report]
        if args.audit_only:
            command.append("--audit-only")
        result = subprocess.run(list(map(str, command)))
        exit_code = result.returncode
        status["state"] = "COMPLETED" if exit_code == 0 else "NEEDS_ATTENTION"
    except Exception as exc:
        status["state"] = "FAILED"
        status["error"] = str(exc)
        print("SETUP FAILED:", exc, flush=True)
        if not (args.report / "SUMMARY.md").exists():
            (args.report / "SUMMARY.md").write_text("# Acquisition failed\n\n" + str(exc) + "\n\nNo training was performed.\n")
    finally:
        status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status["exit_code"] = exit_code
        (args.report / "status.json").write_text(json.dumps(status, indent=2))
        # This pointer is a path, not a copy of downloaded data.
        (args.work / "LATEST_REPORT.txt").write_text(str(args.report) + "\n")
        print("\nREPORT:", args.report / "SUMMARY.md", flush=True)
        if (args.report / "SUMMARY.md").exists():
            print((args.report / "SUMMARY.md").read_text(), flush=True)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
