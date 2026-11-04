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
import re


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
    job_id = os.environ.get("TRAFFIC_JOB_KEY", os.environ.get("SLURM_JOB_ID", ""))
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise SystemExit("A real Slurm job ID is required; use sbatch, not bash Batch")
    job = root / "results" / "slurm_jobs" / job_id
    report = job / "report"
    report.mkdir(parents=True, exist_ok=True)
    stage = os.environ.get("TRAFFIC_STAGE", "acquire")
    if args.mode == "run":
        phase_path = code / "config/reproduction_review.json"
        if stage == "causal-train3" and phase_path.exists() and json.loads(phase_path.read_text())["phase"] == "REPRODUCIBILITY_FIRST":
            raise SystemExit("Causal adaptation is on hold; submit reproduction-review first")
        work = contained(root, Path(os.environ.get("TRAFFIC_WORK_ROOT", str(root / "work"))))
        contained(root, Path(os.environ.get("TRAFFIC_ENV_DIR", str(root / ".venv"))))
        job_name = os.environ.get("SLURM_JOB_NAME", "nutp-p08-preflight" if stage == "pems08-audit" else "nutp-dl-audit")
        workers = root / "logs" / (job_name + "_" + job_id + "_workers")
        workers.mkdir(parents=True, exist_ok=True)
        os.environ["TRAFFIC_WORKER_LOG_DIR"] = str(workers)
        (job / "started.json").write_text(json.dumps({"started_utc": stamp(), "job_id": job_id,
             "actual_slurm_job_id": os.environ["SLURM_JOB_ID"],
             "code": str(code), "work": str(work), "report": str(report)}, indent=2))
        if stage == 'shared-compact-pilot':
            try:
                from compact_baselines import dispatch
                result = dispatch(root, code, report)
            except Exception as exc:
                result = 2
                (report/'compact_progress.json').write_text(json.dumps({'phase': 'FAILED', 'error': str(exc)}))
                (report/'SUMMARY.md').write_text('# Compact pilot setup failed\n\n'+str(exc)+'\n')
        elif stage == 'shared-local-baselines':
            try:
                from shared_baselines import dispatch
                result = dispatch(root, code, report)
            except Exception as exc:
                result = 2
                (report/'baseline_progress.json').write_text(json.dumps({'phase':'FAILED', 'error':str(exc)}))
                (report/'SUMMARY.md').write_text('# Shared baseline setup failed\n\n'+str(exc)+'\n')
        elif stage.startswith('ectt-'):
            try:
                from ectt_workflow import dispatch
                result = dispatch(root, code, report, stage)
            except Exception as exc:
                result = 2
                (report / 'ectt_progress.json').write_text(json.dumps({'phase': 'FAILED', 'error': str(exc)}, indent=2))
                (report / 'SUMMARY.md').write_text('# ECTT setup failed\n\n'+str(exc)+'\n')
                print('ECTT_SETUP_FAILED:', str(exc), flush=True)
        elif stage == "reproduction-review":
            # Read-only historical evidence consolidation; no environment or
            # research-asset downloads and no Torch import are necessary.
            result = subprocess.call([sys.executable, "-u", str(code / "scripts/reproduction_review.py"),
                "--project", str(root), "--code", str(code), "--report", str(report)])
        elif stage in ("reference-train3", "causal-train3"):
            # Only environment preparation is serialized. Training tasks must
            # not hold the global acquisition/environment lock while fitting.
            try:
                from reference_env import ensure
                with (work / "acquisition.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    python = ensure(root, code, report, flavor="gpu")
                arguments = [str(python), "-u", str(code / "scripts/reference_train.py"),
                    "--project", str(root), "--code", str(code), "--report", str(report)]
                if stage == "causal-train3":
                    arguments.append("--causal")
                result = subprocess.call(arguments)
            except Exception as exc:
                result = 2
                (report / "training_progress.json").write_text(json.dumps({"phase": "FAILED", "error": str(exc)}, indent=2))
                (report / "SUMMARY.md").write_text("# Training setup failed\n\n" + str(exc) + "\n")
                print("TRAINING_SETUP_FAILED:", str(exc), flush=True)
        elif stage == "pems08-audit":
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
        elif stage == "source-inspect":
            with (work / "acquisition.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SystemExit("Another project worker is accessing these assets")
                try:
                    from source_audit_env import ensure
                    python = ensure(root, code, report)
                    result = subprocess.call([str(python), "-u", str(code / "scripts/source_inspect.py"),
                        "--project", str(root), "--code", str(code), "--report", str(report)])
                except Exception as exc:
                    result = 2
                    (report / "source_inspection.json").write_text(json.dumps({"execution_status": "FAILED", "error": str(exc), "items": {}}, indent=2))
                    (report / "SUMMARY.md").write_text("# Source inspection setup failed\n\n" + str(exc) + "\n")
                    print("SOURCE_SETUP_FAILED:", str(exc), flush=True)
        elif stage in ("pems08-infer", "beijing-infer", "public8-infer", "release-diagnose", "causality-audit", "gpu-smoke", "baseline-rescore"):
            with (work / "acquisition.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SystemExit("Another project worker is accessing these assets")
                try:
                    from reference_env import ensure
                    python = ensure(root, code, report, flavor="gpu" if stage == "gpu-smoke" else "cpu")
                    script = {"public8-infer": "public8_infer.py", "release-diagnose": "release_diagnose.py", "causality-audit": "causality_audit.py", "gpu-smoke": "gpu_smoke.py", "baseline-rescore": "baseline_rescore.py"}.get(stage, "pems08_infer.py")
                    arguments = [str(python), "-u", str(code / "scripts" / script),
                        "--project", str(root), "--code", str(code), "--report", str(report)]
                    if stage == "baseline-rescore":
                        arguments += ["--source-array", os.environ.get("TRAFFIC_SOURCE_ARRAY", "")]
                    if stage == "beijing-infer":
                        arguments += ["--dataset", "bj", "--preflight"]
                    result = subprocess.call(arguments)
                except Exception as exc:
                    result = 2
                    (report / "inference_result.json").write_text(json.dumps({"execution_status": "FAILED", "error": str(exc)}, indent=2))
                    (report / "SUMMARY.md").write_text("# Reference environment or inference setup failed\n\n" + str(exc) + "\n")
                    print("REFERENCE_SETUP_FAILED:", str(exc), flush=True)
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
    review_reasons = {"pems08-audit": "PEMS08 split/scaling/checkpoint provenance need review",
        "reproduction-review": "Historical reproduction evidence consolidated; review protocol gaps before further experiments or method design",
        "pems08-infer": "Released-checkpoint CPU evaluation complete; training strategy and paper comparison remain unresolved",
        "beijing-infer": "Beijing Metro checkpoint evaluation complete; training strategy and timestamp provenance remain unresolved",
        "public8-infer": "Public checkpoint evaluations complete; review per-dataset comparisons and provenance before training",
        "release-diagnose": "Fixed released-input/split diagnostic complete; inspect all controls before a reproduction decision",
        "source-inspect": "Source workbook and RAR directory inspection complete; verify experiment correspondence before reproduction claims",
        "causality-audit": "Real-checkpoint input-dependence and zero-update gradient audit complete; keep faithful and causal training tracks distinct",
        "gpu-smoke": "Independent training-batch smoke complete; no causal baseline or benchmark training result is established",
        "reference-train3": "Independent single-domain training complete; inspect held-out metrics and protocol limitations",
        "causal-train3": "Index-causal reference adaptation evaluated; inspect causal gates, results and timestamp limitations",
        "baseline-rescore": "Saved baseline forecasts rescored with verified min-max inverse; causal protocol and exact S3 reproduction remain unresolved",
        "acquire": "Known duplicate source domains need review"}
    status = {"job_id": job_id, "stage": stage, "finished_utc": stamp(), "execution_state": state,
              "shell_exit_code": args.shell_exit, "pipeline_exit_code": args.raw_exit,
              "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "report": str(report),
              "training_experiments_completed": 0,
              "reason": review_reasons.get(stage, "Scientific review required") if state == "COMPLETED_WITH_REVIEW" else
                        "Inspect Slurm accounting, stderr and report/" if state == "FAILED" else "Stage completed"}
    if stage in ("reference-train3", "causal-train3"):
        path = report / "training_progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        status.update(training_experiments_completed=int((report / "training_result.json").exists()),
            training_performed=progress.get("new_optimizer_steps", 0) > 0,
            training_optimizer_steps=progress.get("optimizer_steps", 0),
            new_optimizer_steps=progress.get("new_optimizer_steps", 0))
    if stage == "causal-train3":
        path = report / "training_result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        status["causal_protocol_status"] = result.get("protocol_status", "NOT_ESTABLISHED")
        status["paper_reproduction_claim"] = False
    if stage == "baseline-rescore":
        path = report / "rescore_progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        status.update(rescored_experiments_completed=progress.get("completed_tasks", 0),
            rescored_experiments_failed=progress.get("failed_tasks", 0), training_performed=False, model_inference_performed=False)
    if stage == "reproduction-review":
        path = report / "reproduction_review.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        status.update(training_performed=False, model_inference_performed=False, rescoring_performed=False,
            full_paper_reproduction=False, checkpoint_datasets_reviewed=len(result.get("checkpoint_datasets", [])),
            training_experiments_reviewed=len(result.get("training_experiments", [])),
            causal_adaptation_on_hold=True, proposed_method_started=False)
    if stage == "gpu-smoke":
        progress_path = report / "gpu_smoke_progress.json"
        smoke = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        status["training_smoke_fits_completed"] = smoke.get("completed_datasets", 0)
        status["training_experiments_completed"] = smoke.get("completed_datasets", 0)
        status["benchmark_experiments_completed"] = 0
        status["training_smoke_optimizer_steps"] = sum(row.get("completed_steps", 0) for row in smoke.get("datasets", {}).values())
        status["new_training_smoke_optimizer_steps"] = sum(row.get("new_optimizer_steps", max(0, row.get("completed_steps", 0) - row.get("resumed_from_step", 0)))
            for row in smoke.get("datasets", {}).values())
        status["training_performed"] = status["new_training_smoke_optimizer_steps"] > 0
    if stage.startswith('ectt-'):
        path = report / 'ectt_result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        fit = result.get('fit', {})
        status.update(scientific_status='METHOD_NOT_EVALUATED', benchmark_experiments_completed=0,
            training_performed=result.get('training_performed', False),
            training_smoke_optimizer_steps=fit.get('optimizer_steps', 0),
            new_training_smoke_optimizer_steps=fit.get('new_optimizer_steps', 0),
            reason='ECTT admission stage only; full benchmark training and S01 independent review remain pending' if args.shell_exit == 0 else 'Inspect ECTT progress, worker logs and Slurm accounting')
    if stage == 'shared-compact-pilot':
        path = report/'compact_result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        progress_path = report/'compact_progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        status.update(scientific_status=result.get('scientific_status', 'COMPACT_DEVELOPMENT_PILOT_INCOMPLETE'),
                      benchmark_experiments_completed=result.get('completed_cases', progress.get('completed_cases', 0)),
                      training_experiments_completed=result.get('completed_cases', progress.get('completed_cases', 0)),
                      training_performed=result.get('training_performed', progress.get('training_performed', False)),
                      optimizer_steps=result.get('optimizer_steps', 0),
                      new_optimizer_steps=result.get('new_optimizer_steps', progress.get('new_optimizer_steps', 0)),
                      development_only=True, method_evaluated=False, S01_independent_review='PENDING',
                      reason='Compact development pilot only; all-domain B02, LLM-UTP/Qwen and S01 review remain pending'
                          if args.shell_exit == 0 else 'Inspect compact_progress.json, environment status and failure traceback')
    if stage == 'shared-local-baselines':
        path = report/'baseline_result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        progress_path = report/'baseline_progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        status.update(scientific_status=result.get('scientific_status', 'LOCAL_BASELINE_FOUNDATION_INCOMPLETE'),
                      benchmark_experiments_completed=result.get('completed_cases', progress.get('completed_cases', 0)),
                      deterministic_dataset_budget_cases=True,
                      training_performed=result.get('training_performed', progress.get('training_performed', False)),
                      new_ridge_fits=result.get('new_ridge_fits', progress.get('new_ridge_fits', 0)),
                      optimizer_steps=0, method_evaluated=False, S01_independent_review='PENDING',
                      reason='Local persistence/ridge foundation; compact, LLM-UTP/Qwen and independent S01 review remain pending'
                          if args.shell_exit == 0 else 'Inspect baseline_progress.json and failure traceback')
    (job / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (job / "JOB_REPORT.md").write_text("# Traffic reproduction job\n\nStage: " + stage + "\n\nExecution: " + state + "\n\n" + status["reason"] +
        "\n\nScientific status: " + status['scientific_status'] + "\n\nDetailed report: " + str(report / "SUMMARY.md") + "\n")
    print("FINAL_STATUS", json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
