#!/usr/bin/env python3
"""Prepare project-contained entrypoints and submit one resumable download job."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path("/home/tariq.qayyum/nature-llm")


def allocation(stage):
    """Use shortq only when its observed maximum fits this small CPU audit."""
    preflight = stage in ("pems08-audit", "beijing-infer", "release-diagnose", "source-inspect", "causality-audit")
    desired = 1800 if preflight else 23 * 3600
    partitions = ("shortq", "longq") if preflight else ("longq",)
    for partition in partitions:
        try:
            result = subprocess.run(["scontrol", "show", "partition", partition, "-o"], capture_output=True, text=True, timeout=20, check=True)
            if "PartitionName=" + partition not in result.stdout:
                continue
            match = re.search(r"\bMaxTime=(\S+)", result.stdout)
            if not match:
                continue
            maximum = seconds(match[1])
            if partition == "shortq" and maximum is not None and maximum < desired:
                continue
            limit = min(desired, maximum) if maximum is not None else desired
            return partition, limit, "Observed " + partition + " MaxTime=" + match[1]
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return "longq", desired, "Partition maximum unavailable; using supplied longq profile"


def read_json(path):
    return json.loads(path.read_text())


def seconds(value):
    if value.upper() in ("INFINITE", "UNLIMITED", "NOT_SET"):
        return None
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    fields = list(map(int, clock.split(":")))
    if len(fields) == 2:
        fields.insert(0, 0)
    if len(fields) != 3:
        raise ValueError("Unrecognized partition time limit")
    return int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]


def validate_process(pid, project, command, proc_root=Path("/proc")):
    actual = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
    actual = [part.decode() for part in actual if part]
    if actual != command or "--work" not in actual or "--project" not in actual:
        return False
    if Path(actual[actual.index("--work") + 1]).resolve() != (project / "work").resolve():
        return False
    source = Path(actual[actual.index("--project") + 1]).resolve()
    source.relative_to(project.resolve())
    if not any(part.endswith("/scripts/pipeline.py") for part in actual):
        return False
    environ = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    return not any(part.startswith(b"SLURM_JOB_ID=") for part in environ)


def handoff(project):
    paused = []
    # Only detached workers with our recorded exact command and project path.
    for launch in (project / "work" / "reports").glob("*/launch.json"):
        data = read_json(launch)
        pid = data.get("pid")
        if not isinstance(pid, int) or pid <= 1:
            continue
        try:
            if not validate_process(pid, project, data["command"]) or os.getsid(pid) != pid:
                continue
            print("Pausing verified unmanaged download process group", pid, "for Slurm handoff", flush=True)
            os.killpg(pid, signal.SIGTERM)
            paused.append(pid)
            (launch.parent / "handoff_to_slurm.json").write_text(json.dumps({"pid": pid, "signal": "SIGTERM", "partial_files_preserved": True}, indent=2))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, KeyError):
            continue
    return paused


def prepare(project, code):
    if project.resolve() != ROOT.resolve():
        raise ValueError("This HPC Batch profile is explicitly rooted at " + str(ROOT))
    code.resolve().relative_to(project.resolve())
    for name in ("logs", "results", "outputs", "figures", "tmp", ".cache", ".config", ".local", "work"):
        directory = project / name
        directory.resolve().relative_to(project.resolve())
        directory.mkdir(parents=True, exist_ok=True)
    stamp = project / "work" / "installed_entrypoints.json"
    previous = read_json(stamp) if stamp.exists() else {}
    updated = {}
    entries = {"Batch": (code / "Batch").read_bytes(),
               "Check": ('#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "' + str(code / "scripts" / "slurm_check.py") + '" --project "' + str(project) + '" "$@"\n').encode()}
    for name, data in entries.items():
        target = project / name
        digest = hashlib.sha256(data).hexdigest()
        if target.exists() and target.read_bytes() != data and hashlib.sha256(target.read_bytes()).hexdigest() != previous.get(name):
            raise ValueError("Existing locally changed entrypoint was preserved: " + str(target))
        temporary = project / (name + ".new")
        temporary.write_bytes(data)
        temporary.chmod(0o755)
        temporary.replace(target)
        updated[name] = digest
    stamp.write_text(json.dumps(updated, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--handoff", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--stage", choices=["acquire", "pems08-audit", "pems08-infer", "beijing-infer", "public8-infer", "release-diagnose", "source-inspect", "causality-audit"], default="acquire")
    args = parser.parse_args(argv)
    project, code = args.project.resolve(), args.code.resolve()
    prepare(project, code)
    if not shutil.which("sbatch"):
        raise SystemExit("sbatch is unavailable; use the cluster login node and run module load slurm before retrying")
    last = project / "work/LAST_JOB_ID"
    if last.exists():
        previous_id = last.read_text().strip()
        recorded = project / "results/slurm_jobs" / previous_id / "submission.json"
        if previous_id.isdigit() and recorded.exists() and read_json(recorded).get("project") == str(project):
            current = subprocess.run(["squeue", "--jobs", previous_id, "--noheader", "--format=%T"], capture_output=True, text=True, timeout=20)
            if current.returncode == 0 and current.stdout.strip():
                print("Project job", previous_id, "is already", current.stdout.strip(), "- no duplicate job submitted")
                print("Check: bash " + str(project / "Check") + " " + previous_id)
                return 0
    # Do not signal a live downloader until Slurm submission is available.
    if args.handoff:
        handoff(project)
    import fcntl
    with (project / "work/acquisition.lock").open("a") as lock:
        acquired = False
        for _ in range(20 if args.handoff else 1):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if args.handoff:
                    time.sleep(0.5)
        if not acquired:
            raise SystemExit("An acquisition still holds this project's lock. No duplicate download job was submitted. Do not stop an unidentified process/job.")
        partition, limit, partition_evidence = allocation(args.stage)
        profiles = {"acquire": ("nutp-dl-audit", 4, "8G"), "pems08-audit": ("nutp-p08-preflight", 1, "4G"),
                    "pems08-infer": ("nutp-p08-infer-cpu", 8, "16G"),
                    "beijing-infer": ("nutp-bj-infer-cpu", 4, "8G"),
                    "public8-infer": ("nutp-public8-infer", 16, "32G"),
                    "release-diagnose": ("nutp-release-diag", 12, "24G"),
                    "source-inspect": ("nutp-source-inspect", 1, "4G"),
                    "causality-audit": ("nutp-causal-audit", 4, "16G")}
        job_name, cpus, memory = profiles[args.stage]
        request_time = "%02d:%02d:%02d" % (limit // 3600, limit % 3600 // 60, limit % 60)
        env = {**os.environ, "TRAFFIC_PROJECT_ROOT": str(project), "TRAFFIC_WORK_ROOT": str(project / "work"), "TRAFFIC_CODE_DIR": str(code),
               "TRAFFIC_AUDIT_ONLY": "1" if args.audit_only else "0", "TRAFFIC_STAGE": args.stage}
        command = ["sbatch", "--parsable", "--job-name=" + job_name, "--partition=" + partition,
            "--cpus-per-task=" + str(cpus), "--mem=" + memory, "--time=" + request_time, "--export=ALL", str(project / "Batch")]
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SystemExit("sbatch returned an unexpected identifier; inspect: " + result.stdout)
        directory = project / "results/slurm_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        submission = {"project": str(project), "job_id": job_id, "job_name": job_name, "stage": args.stage, "code": str(code),
            "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "partition": partition, "cpus": cpus,
            "gpus": 0, "time_limit": request_time, "partition_evidence": partition_evidence, "command": command}
        (directory / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
        (project / "work/LAST_JOB_ID").write_text(job_id + "\n")
        print("SUBMITTED Slurm job", job_id, "name=" + job_name, "partition=" + partition, "cpus=" + str(cpus), "gpus=0 time=" + request_time)
        print("Check: bash " + str(project / "Check") + " " + job_id)
        print("Report: " + str(directory / "report/SUMMARY.md"))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("Slurm submission failed:", str(exc), file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stderr or "", file=sys.stderr)
        sys.exit(2)
