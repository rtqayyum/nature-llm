#!/usr/bin/env python3
"""Foreground Slurm worker and durable final execution/science status."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import subprocess
import fcntl


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
    stage = os.environ.get("TRAFFIC_STAGE", "acquire")
    if args.mode == "run":
        work = contained(root, Path(os.environ.get("TRAFFIC_WORK_ROOT", str(root / "work"))))
        contained(root, Path(os.environ.get("TRAFFIC_ENV_DIR", str(root / ".venv"))))
        job_name = os.environ.get("SLURM_JOB_NAME", "nutp-p08-preflight" if stage == "pems08-audit" else "nutp-dl-audit")
        workers = root / "logs" / (job_name + "_" + job_id + "_workers")
        workers.mkdir(parents=True, exist_ok=True)
        os.environ["TRAFFIC_WORKER_LOG_DIR"] = str(workers)
        (job / "started.json").write_text(json.dumps({"started_utc": stamp(), "job_id": job_id,
             "code": str(code), "work": str(work), "report": str(report)}, indent=2))
        if stage == "pems08-audit":
            python = root / ".venv/bin/python"
            if not python.exists():
                raise SystemExit("Project .venv is missing; complete acquisition setup before this preflight")
            with (work / "acquisition.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SystemExit("Another project worker is accessing these assets; preflight stopped without changing them")
                result = subprocess.call([str(python), "-u", str(code / "scripts/pems08_audit.py"),
                    "--project", str(root), "--code", str(code), "--report", str(report)])
        else:
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
    status = {"job_id": job_id, "stage": stage, "finished_utc": stamp(), "execution_state": state,
              "shell_exit_code": args.shell_exit, "pipeline_exit_code": args.raw_exit,
              "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "report": str(report),
              "training_experiments_completed": 0,
              "reason": ("PEMS08 split/scaling/checkpoint provenance need review" if stage == "pems08-audit" else "Known duplicate source domains need review") if state == "COMPLETED_WITH_REVIEW" else
                        "Inspect Slurm accounting, stderr and report/" if state == "FAILED" else "Stage completed"}
    (job / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (job / "JOB_REPORT.md").write_text("# Traffic reproduction job\n\nStage: " + stage + "\n\nExecution: " + state + "\n\n" + status["reason"] +
        "\n\nScientific status: REPRODUCTION_NOT_ESTABLISHED\n\nDetailed report: " + str(report / "SUMMARY.md") + "\n")
    print("FINAL_STATUS", json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
