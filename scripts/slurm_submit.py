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


import transfer_diagnostics
import transfer_plan


GPU_STAGES = ("gpu-smoke", "reference-train3", "causal-train3", "ectt-gpu-pilot", "shared-compact-pilot", "shared-compact-baselines", "shared-transfer-check")+transfer_plan.GPU_STAGES
ECTT_STAGES = ("ectt-preflight", "ectt-stat-check", "ectt-gpu-pilot", "ectt-next")
SHARED_STAGES = ('shared-local-baselines',)
COMPACT_STAGES = ('shared-compact-pilot',)
GRID_STAGES = ('shared-compact-baselines', 'shared-compact-report')
TRANSFER_STAGES = ('shared-transfer-check',)
STUDY_STAGES = ECTT_STAGES+SHARED_STAGES+COMPACT_STAGES+GRID_STAGES+TRANSFER_STAGES+transfer_plan.STAGES+(transfer_diagnostics.STAGE,)

ROOT = Path("/home/tariq.qayyum/nature-llm")


def allocation(stage):
    """Use shortq only when its observed maximum fits this small CPU audit."""
    preflight = stage in ("pems08-audit", "beijing-infer", "release-diagnose", "source-inspect", "causality-audit", "baseline-rescore", "reproduction-review", "ectt-preflight", "ectt-stat-check")
    preflight = preflight or stage in SHARED_STAGES or stage == 'shared-compact-report'
    desired = 3600 if stage in SHARED_STAGES or stage == 'shared-compact-report' else 10 * 86400 if stage in ("reference-train3", "causal-train3") else 2 * 3600 if stage in ("gpu-smoke", "ectt-gpu-pilot", "shared-compact-baselines")+COMPACT_STAGES+TRANSFER_STAGES else 1800 if preflight else 23 * 3600
    if stage in transfer_plan.STAGES:
        preflight = stage in (transfer_plan.PREFLIGHT, transfer_plan.REPORT)
        recipe = read_json(Path(__file__).resolve().parents[1]/'config/shared_transfer_v1.json')
        key = stage.rsplit('-', 1)[1]
        desired = recipe[(key if key != 'data' else 'report')+'_wall_seconds']
    if stage == transfer_diagnostics.STAGE:
        preflight, desired = True, 3600
    partitions = ("gpuq",) if stage in GPU_STAGES else ("shortq", "longq") if preflight else ("longq",)
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
    return ("gpuq" if stage in GPU_STAGES else "longq"), desired, "Partition maximum unavailable; using supplied profile"


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
    parser.add_argument("--stage", choices=["acquire", "pems08-audit", "pems08-infer", "beijing-infer", "public8-infer", "release-diagnose", "source-inspect", "causality-audit", "gpu-smoke", "reference-train3", "baseline-rescore", "causal-train3", "reproduction-review"]+list(STUDY_STAGES)+[transfer_plan.LAUNCH], default="acquire")
    parser.add_argument("--source-array", help="Recorded reference-training array to rescore without retraining")
    parser.add_argument('--retry-grid', action='store_true', help='Explicitly submit a new compact array after the previous batch ended; reuse compatible completed cases/checkpoints')
    args = parser.parse_args(argv)
    if args.stage == "baseline-rescore" and (not args.source_array or not re.fullmatch(r"[0-9]+", args.source_array)):
        parser.error("baseline-rescore requires --source-array with a numeric project training array ID")
    if args.source_array and args.stage != "baseline-rescore":
        parser.error("--source-array is only valid with baseline-rescore")
    project, code = args.project.resolve(), args.code.resolve()
    if args.retry_grid and args.stage not in ('shared-compact-baselines', transfer_plan.LAUNCH):
        parser.error('--retry-grid is only valid with a compact or transfer batch')
    if args.stage == transfer_plan.LAUNCH:
        from transfer_submit import submit
        return submit(project, code, retry=args.retry_grid)
    if args.stage in transfer_plan.STAGES:
        parser.error('Submit the complete transfer chain with --stage shared-transfer-baselines')
    if args.stage == 'shared-compact-baselines':
        from compact_grid_submit import submit
        return submit(project, code, retry=args.retry_grid)
    if args.stage == 'shared-compact-report':
        parser.error('The compact collector is submitted with its array; use --stage shared-compact-baselines')
    if args.stage in TRANSFER_STAGES:
        from pretransfer_checks import foundation
        foundation(project, code, full=False)
    if args.stage == transfer_diagnostics.STAGE:
        transfer_diagnostics.foundation(project, code)
    if args.stage in ECTT_STAGES:
        from ectt_workflow import next_stage, require_previous
        if args.stage == "ectt-next":
            args.stage = next_stage(project, code)
        require_previous(project, code, args.stage)
    if args.stage in SHARED_STAGES:
        # Small source/status checks only. Full artifact/data integrity reads
        # run inside the CPU allocation, never as login-node preprocessing.
        from shared_baselines import foundation
        foundation(project, code, full=False)
    if args.stage in COMPACT_STAGES:
        from compact_baselines import foundation
        foundation(project, code, full=False)
    phase_path = code / "config/reproduction_review.json"
    if args.stage == "causal-train3" and phase_path.exists() and read_json(phase_path)["phase"] == "REPRODUCIBILITY_FIRST":
        parser.error("causal-train3 is on hold under the user's reproducibility-first plan; use --stage reproduction-review")
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
        if args.stage in STUDY_STAGES:
            recorded_ids = {p.parent.name for p in (project / 'results/slurm_jobs').glob('*/submission.json')
                if p.parent.name.isdigit() and read_json(p).get('stage') in STUDY_STAGES}
            if recorded_ids:
                live = subprocess.run(['squeue', '--noheader', '--format=%A'], capture_output=True, text=True, timeout=20, check=True)
                active = recorded_ids.intersection(live.stdout.split())
                if active:
                    print('Study job already active; serial execution preserved:', ','.join(sorted(active)))
                    return 0
        partition, limit, partition_evidence = allocation(args.stage)
        profiles = {"acquire": ("nutp-dl-audit", 4, "8G"), "pems08-audit": ("nutp-p08-preflight", 1, "4G"),
                    "pems08-infer": ("nutp-p08-infer-cpu", 8, "16G"),
                    "beijing-infer": ("nutp-bj-infer-cpu", 4, "8G"),
                    "public8-infer": ("nutp-public8-infer", 16, "32G"),
                    "release-diagnose": ("nutp-release-diag", 12, "24G"),
                    "source-inspect": ("nutp-source-inspect", 1, "4G"),
                    "causality-audit": ("nutp-causal-audit", 4, "16G"),
                    "gpu-smoke": ("nutp-s3-gpu-smoke", 20, "16G"),
                    "reference-train3": ("nutp-s3-3seed", 5, "16G"),
                    "baseline-rescore": ("nutp-baseline-fix", 2, "8G"),
                    "causal-train3": ("nutp-causal-3seed", 5, "16G"),
                    "reproduction-review": ("nutp-repro-review", 1, "2G"),
                    transfer_diagnostics.STAGE: ("ectt-xfer-diag", 5, "8G")}
        if args.stage in ECTT_STAGES:
            from ectt_common import load_contract
            _, study_execution = load_contract(code)
            row = study_execution['stages'][args.stage]
            profiles[args.stage] = (row['job_name'], row['cpus'], row['memory'])
        if args.stage in SHARED_STAGES:
            row = read_json(code/'config/shared_baselines_v1.json')['resources']
            profiles[args.stage] = (row['job_name'], row['cpus'], row['memory'])
        if args.stage in COMPACT_STAGES:
            row = read_json(code/'config/compact_local_v1.json')['resources']
            profiles[args.stage] = (row['job_name'], row['cpus'], row['memory'])
        if args.stage in TRANSFER_STAGES:
            row = read_json(code/'config/pretransfer_checks_v1.json')['resources']
            profiles[args.stage] = (row['job_name'], row['cpus'], row['memory'])
        job_name, cpus, memory = profiles[args.stage]
        request_time = "%02d:%02d:%02d" % (limit // 3600, limit % 3600 // 60, limit % 60)
        env = {**os.environ, "TRAFFIC_PROJECT_ROOT": str(project), "TRAFFIC_WORK_ROOT": str(project / "work"), "TRAFFIC_CODE_DIR": str(code),
               "TRAFFIC_AUDIT_ONLY": "1" if args.audit_only else "0", "TRAFFIC_STAGE": args.stage,
               "TRAFFIC_SOURCE_ARRAY": args.source_array or ""}
        command = ["sbatch", "--parsable", "--job-name=" + job_name, "--partition=" + partition,
            "--cpus-per-task=" + str(cpus), "--mem=" + memory, "--time=" + request_time, "--export=ALL", str(project / "Batch")]
        if args.stage in GPU_STAGES:
            command.insert(-1, "--gres=gpu:1")
        from job_notifications import preferences, queued_receipt
        notification_settings = preferences(project)
        if args.stage in STUDY_STAGES and not notification_settings:
            raise ValueError('Study submissions require saved email preferences or TRAFFIC_MAIL_USER; no silent notification omission')
        if notification_settings:
            command[1:1] = ["--mail-user=" + notification_settings["mail_user"],
                            "--mail-type=" + notification_settings["mail_type"]]
        tasks = []
        if args.stage in ("reference-train3", "causal-train3"):
            config = read_json(code / "config" / ("causal_reference.json" if args.stage == "causal-train3" else "reference_training.json"))
            tasks = [{"dataset": dataset, "seed": seed} for dataset in config["datasets"] for seed in config["seeds"]]
            command[1:1] = ["--array=0-%d%%%d" % (len(tasks) - 1, config["max_concurrent_tasks"]),
                "--output=" + str(project / "logs/%x_%A_%a.out"),
                "--error=" + str(project / "logs/%x_%A_%a.err")]
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SystemExit("sbatch returned an unexpected identifier; inspect: " + result.stdout)
        directory = project / "results/slurm_jobs" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        submission = {"project": str(project), "job_id": job_id, "job_name": job_name, "stage": args.stage, "code": str(code),
            "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "partition": partition, "cpus": cpus,
            "gpus": 1 if args.stage in GPU_STAGES else 0, "time_limit": request_time, "partition_evidence": partition_evidence, "command": command}
        if args.source_array:
            submission["source_array"] = args.source_array
        if tasks:
            submission["protocol"] = config["protocol"]
            submission["array_tasks"] = tasks
            submission["max_concurrent_tasks"] = config["max_concurrent_tasks"]
            for index, task in enumerate(tasks):
                child_id = job_id + "_" + str(index)
                child = project / "results/slurm_jobs" / child_id
                child.mkdir(parents=True, exist_ok=True)
                (child / "submission.json").write_text(json.dumps({**submission, "job_id": child_id,
                    "array_job_id": job_id, "task_index": index, **task}, indent=2) + "\n")
        submission["notifications"] = notification_settings or {"state": "NOT_CONFIGURED"}
        submission["queued_email"] = queued_receipt(project, submission, notification_settings)
        print("EMAIL:", json.dumps(submission["queued_email"]), flush=True)
        (directory / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
        (project / "work/LAST_JOB_ID").write_text(job_id + "\n")
        print("SUBMITTED Slurm job", job_id, "name=" + job_name, "partition=" + partition, "cpus=" + str(cpus), "gpus=" + str(submission["gpus"]) + " time=" + request_time)
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
