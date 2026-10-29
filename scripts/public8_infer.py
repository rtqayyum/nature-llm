#!/usr/bin/env python3
"""Evaluate all public released checkpoints with bounded isolated CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from acquire import atomic_json
from reference_datasets import PROFILES, PUBLIC_ORDER


def run(project, code, report, python=None):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    report.mkdir(parents=True, exist_ok=True)
    python = str(python or sys.executable)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    workers = min(4, max(1, cpus // 4))
    threads = max(1, cpus // workers)
    job_id = os.environ.get("SLURM_JOB_ID", "local-test")
    name = os.environ.get("SLURM_JOB_NAME", "nutp-public8-infer")
    log_dir = project / "logs" / (name + "_" + job_id + "_workers")
    log_dir.resolve().relative_to(project)
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = {"stage": "public8-infer", "phase": "RUNNING", "total_datasets": len(PUBLIC_ORDER),
                "completed_datasets": 0, "failed_datasets": 0, "parallel_workers": workers,
                "threads_per_worker": threads, "training_experiments_completed": 0,
                "datasets": {key: {"state": "PENDING"} for key in PUBLIC_ORDER}}
    mutex = threading.Lock()

    def update(key=None, **values):
        with mutex:
            if key:
                progress["datasets"][key].update(values)
            progress["completed_datasets"] = sum(row["state"] == "COMPLETED_WITH_REVIEW" for row in progress["datasets"].values())
            progress["failed_datasets"] = sum(row["state"] == "FAILED" for row in progress["datasets"].values())
            atomic_json(report / "public8_progress.json", progress)
            print("PUBLIC8 completed=%d/%d failed=%d latest=%s" %
                  (progress["completed_datasets"], len(PUBLIC_ORDER), progress["failed_datasets"], key or "setup"), flush=True)

    def evaluate(key):
        profile = PROFILES[key]
        destination = report / "datasets" / profile["slug"]
        destination.mkdir(parents=True, exist_ok=True)
        freeze = report / "reference_environment_freeze.txt"
        (destination / freeze.name).write_bytes(freeze.read_bytes())
        output, error = log_dir / (profile["slug"] + ".out"), log_dir / (profile["slug"] + ".err")
        update(key, state="RUNNING", report=str(destination), stdout=str(output), stderr=str(error))
        try:
            env = {**os.environ, "SLURM_CPUS_PER_TASK": str(threads), "OMP_NUM_THREADS": str(threads),
                   "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
            with output.open("a") as out, error.open("a") as err:
                result = subprocess.run([python, "-u", str(code / "scripts/pems08_infer.py"),
                    "--project", str(project), "--code", str(code), "--report", str(destination),
                    "--dataset", key, "--preflight"], env=env, stdout=out, stderr=err)
            path = destination / "inference_result.json"
            measured = json.loads(path.read_text()) if path.exists() else {}
            required = ("metrics_author", "metrics_independent", "scored_origins", "output_store", "paper_anchor_comparisons")
            passed = (result.returncode == 3 and measured.get("execution_status") == "COMPLETED_WITH_REVIEW"
                      and measured.get("strict_load_performed") is True and measured.get("dataset") == key
                      and all(field in measured for field in required))
            if not passed:
                preflight = destination / (profile["slug"] + "_audit.json")
                details = json.loads(preflight.read_text()) if preflight.exists() else {}
                raise RuntimeError(measured.get("error") or details.get("error") or
                                   "Worker exited %d without a complete strict inference result" % result.returncode)
            row = {"dataset": key, "title": profile["title"], "state": "COMPLETED_WITH_REVIEW",
                   "report": str(destination), **{field: measured[field] for field in required}}
            update(key, state=row["state"])
            return row
        except Exception as exc:
            reason = str(exc)
            update(key, state="FAILED", reason=reason)
            return {"dataset": key, "title": profile["title"], "state": "FAILED", "reason": reason,
                    "report": str(destination), "stderr": str(error)}

    update()
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, key): key for key in PUBLIC_ORDER}
        for future in as_completed(futures):
            key = futures[future]
            try:
                rows[key] = future.result()
            except Exception as exc:
                # Even a setup failure in one worker must not hide other results.
                rows[key] = {"dataset": key, "title": PROFILES[key]["title"], "state": "FAILED", "reason": str(exc)}
                update(key, state="FAILED", reason=str(exc))
    failed = any(row["state"] == "FAILED" for row in rows.values())
    progress["phase"] = "FAILED" if failed else "COMPLETED_WITH_REVIEW"
    update()
    summary = {"stage": "public8-infer", "execution_status": progress["phase"],
        "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "training_experiments_completed": 0,
        "completed_datasets": progress["completed_datasets"], "failed_datasets": progress["failed_datasets"],
        "datasets": [rows[key] for key in PUBLIC_ORDER],
        "limitations": ["Released-checkpoint CPU evaluation only; checkpoint training strategy remains unresolved",
            "Original scaling, masks, batches and splits are retained; timestamps and service gaps are unverified",
            "Stage-one bike/taxi duplicate payloads remain a joint-training blocker; distinct stage-two evaluations do not resolve it",
            "Three restricted full datasets are not available; example arrays are excluded",
            "Paper S1 rounding comparison is diagnostic only, never a checkpoint strategy assignment"]}
    atomic_json(report / "public8_result.json", summary)
    lines = ["# Eight public released-checkpoint CPU evaluations", "", "Execution: " + progress["phase"],
             "", "Completed datasets: %d/8. Failed datasets: %d. Training runs: 0." %
             (summary["completed_datasets"], summary["failed_datasets"]), "",
             "| Dataset | MAE | RMSE | WMAPE % | S1 rounding matches / 3 | State |",
             "|---|---:|---:|---:|---:|---|"]
    for key in PUBLIC_ORDER:
        row = rows[key]
        if row["state"] == "FAILED":
            lines += ["| %s | — | — | — | — | FAILED |" % row["title"]]
        else:
            metric = row["metrics_author"]
            count = sum(row["paper_anchor_comparisons"]["S1"]["same_two_decimal_rounding"].values())
            lines.append("| %s | %.8f | %.8f | %.8f | %d | completed with review |" %
                         (row["title"], metric["MAE"], metric["RMSE"], metric["WMAPE_percent"], count))
    lines += ["", "## Scope and limitations", ""] + ["- " + value for value in summary["limitations"]]
    for row in rows.values():
        if row["state"] == "FAILED":
            lines += ["", "Failed " + row["title"] + ": " + row["reason"]]
    lines += ["", "Per-dataset details: datasets/SLUG/inference_result.json and SUMMARY.md.",
              "Prediction shards: PROJECT/outputs/SLUG_reference_cpu/RUN_HASH/.",
              "Recovery: resubmit public8-infer; compatible shards are verified and reused.", ""]
    (report / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return 2 if failed else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.project, args.code, args.report))
