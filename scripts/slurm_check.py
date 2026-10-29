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
                whole, _, fractional = fields[7].partition(".")
                total_cpu = seconds(whole)
                if total_cpu is not None and fractional:
                    total_cpu += float("0." + fractional)
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
    stage = submission.get("stage", "acquire")
    progress_path = job / "report/asset_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if stage == "public8-infer":
        environment = job / "report/reference_environment_status.json"
        if environment.exists():
            print("REFERENCE ENVIRONMENT:", json.dumps(json.loads(environment.read_text()), indent=2))
        path = job / "report/public8_progress.json"
        batch = json.loads(path.read_text()) if path.exists() else {}
        print("PUBLIC DATASETS completed=%s/8 failed=%s" % (batch.get("completed_datasets", 0), batch.get("failed_datasets", 0)))
        for key, row in batch.get("datasets", {}).items():
            print("DATASET", key, row["state"], row.get("reason", ""))
            if row.get("report"):
                directory = Path(row["report"])
                directory.resolve().relative_to(project)
                detail = directory / "inference_progress.json"
                if detail.exists():
                    item = json.loads(detail.read_text())
                    print("  phase=%s batches=%s/%s origins=%s resumed=%s" % (item.get("phase"),
                        item.get("completed_batches"), item.get("total_batches"), item.get("completed_origins"), item.get("resumed_batches")))
            if row["state"] == "FAILED" and row.get("stderr"):
                error = Path(row["stderr"])
                error.resolve().relative_to(project)
                print("FAILED WORKER LOG:", error)
                if error.exists():
                    print("\n".join(error.read_text(errors="replace").splitlines()[-10:]))
        path = job / "report/public8_result.json"
        if path.exists():
            for row in json.loads(path.read_text())["datasets"]:
                print("DATASET RESULT", row["title"], row["state"], json.dumps(row.get("metrics_author", {})))
                if row.get("paper_anchor_comparisons"):
                    print("  S1 rounding (diagnostic only):", json.dumps(row["paper_anchor_comparisons"]["S1"]["same_two_decimal_rounding"]))
    elif stage in ("pems08-infer", "beijing-infer"):
        environment_file = job / "report/reference_environment_status.json"
        if environment_file.exists():
            print("REFERENCE ENVIRONMENT:", json.dumps(json.loads(environment_file.read_text()), indent=2))
        inference_file = job / "report/inference_progress.json"
        if inference_file.exists():
            print("INFERENCE PROGRESS:", json.dumps(json.loads(inference_file.read_text()), indent=2))
        result_file = job / "report/inference_result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            print("INFERENCE RESULT:", json.dumps({key: value for key, value in result.items()
                if key in ("dataset", "execution_status", "error", "metrics_author", "metrics_independent", "scored_origins", "checkpoint_strategy", "strict_load_performed", "output_store", "limitations")}, indent=2))
            if result.get("paper_anchor_comparisons"):
                print("DIAGNOSTIC PAPER ROUNDING (not a reproduction pass):", json.dumps({key: value["same_two_decimal_rounding"]
                    for key, value in result["paper_anchor_comparisons"].items()}))
    elif stage == "pems08-audit":
        preflight_file = job / "report/preflight_progress.json"
        preflight = json.loads(preflight_file.read_text()) if preflight_file.exists() else {}
        print("PEMS08 PREFLIGHT checks=%s/%s" % (preflight.get("completed_checks", 0), preflight.get("total_checks", 5)))
        details = job / "report/pems08_audit.json"
        if details.exists():
            print("PREFLIGHT SUMMARY:", json.dumps({key: value for key, value in json.loads(details.read_text()).items()
                if key in ("execution_status", "error", "overlaps", "observations")}, indent=2))
    else:
        print("ASSET WORKERS completed=%s/%s verified=%s failed=%s" %
              (progress.get("completed_assets", 0), progress.get("total_assets", 37), progress.get("verified_assets", 0), progress.get("failed_assets", 0)))
    print("Training experiments completed: 0 (reference inference/audits do not train parameters)")
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
