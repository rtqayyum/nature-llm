#!/usr/bin/env python3
"""Show queue/accounting reasons, asset progress, utilization and failed logs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from slurm_submit import seconds


def command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        return {"command": arguments, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": arguments, "exit_code": -1, "stdout": "", "stderr": str(exc)}


def cpu_utilization(accounting, job_id):
    for line in accounting.splitlines():
        fields = line.split("|")
        if len(fields) >= 9 and fields[0] == job_id:
            try:
                elapsed, cpus = int(fields[5]), int(fields[6])
                # CPU durations can contain fractional seconds.
                total_cpu = seconds(fields[7].split(".", 1)[0])
                if elapsed > 0 and cpus > 0 and total_cpu is not None:
                    return round(100 * total_cpu / (elapsed * cpus), 2)
            except (ValueError, TypeError):
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("job_id", nargs="?")
    args = parser.parse_args()
    project = args.project.resolve()
    job_id = args.job_id or (project / "work/LAST_JOB_ID").read_text().strip()
    if not job_id.isdigit():
        raise SystemExit("A numeric project Slurm job ID is required")
    job = project / "results/slurm_jobs" / job_id
    submission = json.loads((job / "submission.json").read_text())
    if Path(submission["project"]).resolve() != project or submission["job_id"] != job_id:
        raise SystemExit("Job is not recorded as belonging to this project")
    queue = command(["squeue", "--jobs", job_id, "--noheader", "--format=%i|%T|%R"])
    accounting = command(["sacct", "-j", job_id, "--parsable2", "--noheader",
        "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,Reason"])
    statistics = command(["sstat", "--allsteps", "-j", job_id, "--format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize"])
    print("Project job:", job_id, "name=" + submission["job_name"], "queue=" + submission["partition"])
    print("\nQUEUE (JobID | State | Reason):\n" + (queue["stdout"] or queue["stderr"] or "No active queue record; inspect accounting below"))
    print("ACCOUNTING (JobID | Name | State | ExitCode | Elapsed | ElapsedRaw | CPUs | TotalCPU | MaxRSS | Reason):\n" + (accounting["stdout"] or accounting["stderr"]))
    print("LIVE CPU/MEMORY:\n" + (statistics["stdout"] or statistics["stderr"] or "Not running / accounting metrics unavailable"))
    cpu = cpu_utilization(accounting["stdout"], job_id)
    if cpu is not None:
        print("CPU capacity used over elapsed allocation: %.2f%%" % cpu)
    status_path = job / "job_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"execution_state": "No final job report yet; use queue/accounting state"}
    print("PROJECT STATUS:", json.dumps(status, indent=2))
    progress_path = job / "report/asset_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    print("ASSET WORKERS completed=%s/%s verified=%s failed=%s" %
          (progress.get("completed_assets", 0), progress.get("total_assets", 37), progress.get("verified_assets", 0), progress.get("failed_assets", 0)))
    print("Training experiments completed: 0 (this job is download/audit only)")
    if progress:
        print("Verified asset GiB: %.3f / %.3f" % (progress["verified_bytes"] / 2**30, progress["expected_bytes"] / 2**30))
    for failure in progress.get("failures", []):
        print("FAILED ASSET:", failure["path"], "reason:", failure["error"])
        if failure.get("worker_log"):
            path = Path(failure["worker_log"])
            path.resolve().relative_to(project)
            print("FAILED WORKER LOG:", path)
            if path.exists():
                print("\n".join(path.read_text(errors="replace").splitlines()[-10:]))
    for suffix in ("out", "err"):
        path = project / "logs" / (submission["job_name"] + "_" + job_id + "." + suffix)
        if path.exists():
            print("RECENT", suffix.upper(), str(path))
            print("\n".join(path.read_text(errors="replace").splitlines()[-15:]))
    print("Partial/final assets:", project / "work")
    print("Detailed report:", job / "report/SUMMARY.md")
    observed = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "queue": queue,
        "accounting": accounting, "live_statistics": statistics, "cpu_capacity_percent": cpu,
        "job_status": status, "asset_counts": {key: progress.get(key) for key in
            ("total_assets", "completed_assets", "verified_assets", "failed_assets")}}
    (job / "monitor.json").write_text(json.dumps(observed, indent=2) + "\n")


if __name__ == "__main__":
    main()
