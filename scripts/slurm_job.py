#!/usr/bin/env python3
"""Foreground Slurm worker and durable final execution/science status."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def contained(root, path):
    path.resolve().relative_to(root.resolve())
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "finalize"])
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--raw-exit", type=int, default=-1)
    parser.add_argument("--shell-exit", type=int, default=1)
    args = parser.parse_args(argv)
    root = args.project.resolve()
    code = contained(root, Path(os.environ["TRAFFIC_CODE_DIR"]))
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise SystemExit("A real Slurm job ID is required; use sbatch, not bash Batch")
    job = root / "results" / "slurm_jobs" / job_id
    report = job / "report"
    report.mkdir(parents=True, exist_ok=True)
    if args.mode == "run":
        work = contained(root, Path(os.environ.get("TRAFFIC_WORK_ROOT", str(root / "work"))))
        contained(root, Path(os.environ.get("TRAFFIC_ENV_DIR", str(root / ".venv"))))
        workers = root / "logs" / ("nutp-dl-audit_" + job_id + "_workers")
        workers.mkdir(parents=True, exist_ok=True)
        os.environ["TRAFFIC_WORKER_LOG_DIR"] = str(workers)
        (job / "started.json").write_text(json.dumps({"started_utc": stamp(), "job_id": job_id,
             "code": str(code), "work": str(work), "report": str(report)}, indent=2))
        from pipeline import main as pipeline_main
        sys.argv = ["pipeline.py", "--project", str(code), "--work", str(work), "--report", str(report)]
        if os.environ.get("TRAFFIC_AUDIT_ONLY") == "1":
            sys.argv.append("--audit-only")
        result = pipeline_main()
        (job / "pipeline_result.json").write_text(json.dumps({"pipeline_exit_code": result,
            "execution_exit_code": 0 if result == 3 else result}, indent=2) + "\n")
        # A completed download with a known scientific review requirement is
        # operationally successful, including the srun step's accounting state.
        return 0 if result == 3 else result
    state = "COMPLETED_WITH_REVIEW" if args.raw_exit == 3 and args.shell_exit == 0 else "COMPLETED" if args.shell_exit == 0 else "FAILED"
    status = {"job_id": job_id, "finished_utc": stamp(), "execution_state": state,
              "shell_exit_code": args.shell_exit, "pipeline_exit_code": args.raw_exit,
              "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "report": str(report),
              "training_experiments_completed": 0,
              "reason": "Known duplicate source domains need review" if state == "COMPLETED_WITH_REVIEW" else
                        "Inspect Slurm accounting, stderr and report/status.json" if state == "FAILED" else "Acquisition completed"}
    (job / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (job / "JOB_REPORT.md").write_text("# Download and audit job\n\nExecution: " + state + "\n\n" + status["reason"] +
        "\n\nScientific status: REPRODUCTION_NOT_ESTABLISHED\n\nDetailed report: " + str(report / "SUMMARY.md") + "\n")
    print("FINAL_STATUS", json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
