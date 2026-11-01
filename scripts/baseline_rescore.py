#!/usr/bin/env python3
"""CPU rescore of saved normalized forecasts. Never train or modify source runs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import traceback

import numpy as np

from acquire import atomic_json, prepare_reference, verify
from pems08_audit import derive_splits, intersection_count, literal_assignment
from pems08_infer import module_from
from reference_contract import atomic_npz, file_sha, independent_metrics, inverse_author, inverse_minmax, load_batch
from reference_datasets import PROFILES

INVERSE = "TRAINING_MINMAX_AFFINE_V1"


def contained(root, path):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    return path


def metrics_pair(module, truth, prediction):
    rmse, r2, mae, wmape = module.Metrics(truth, prediction).evaluate_performance()
    author = {"MAE": float(mae), "RMSE": float(rmse), "WMAPE_percent": float(wmape) * 100, "R2": float(r2)}
    independent = independent_metrics(truth, prediction)
    if not all(np.isfinite(list(author.values()))) or not all(np.isclose(author[key], value, rtol=1e-5, atol=1e-6) for key, value in independent.items()):
        raise ValueError("Author/independent metrics disagree or are nonfinite")
    return {"author": author, "independent": independent}


def corrected_shard(destination, original, source_sha, minimum, maximum, scale, identity_hash):
    """Persist both transforms; verify compatible cached outputs on recovery."""
    expected = {"target_indices": original["target_indices"], "truth": original["truth"],
        "normalized_predictions": original["normalized_predictions"],
        "predictions_legacy": inverse_author(original["normalized_predictions"], scale),
        "predictions_corrected": inverse_minmax(original["normalized_predictions"], minimum, maximum)}
    record = destination.with_suffix(".json")
    if destination.exists() and record.exists():
        previous = json.loads(record.read_text())
        if previous.get("identity_hash") != identity_hash or previous.get("source_sha256") != source_sha or file_sha(destination) != previous["sha256"]:
            raise ValueError("Saved rescore shard identity/hash mismatch; output preserved")
        with np.load(destination, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected) or not all(np.array_equal(arrays[key], value) for key, value in expected.items()):
                raise ValueError("Saved rescore shard differs from its recorded transform")
        return expected, True
    atomic_npz(destination, **expected)
    atomic_json(record, {"sha256": file_sha(destination), "source_sha256": source_sha, "identity_hash": identity_hash})
    return expected, False


def rescore_task(project, code, report, group, index, task, lock, repo, metrics_module, update):
    locks = project / "work/baseline_rescore_locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (group + "_" + str(index) + ".lock")).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This source task is already being rescored; no duplicate writer started")
        return _rescore_task(project, code, report, group, index, task, lock, repo, metrics_module, update)


def _rescore_task(project, code, report, group, index, task, lock, repo, metrics_module, update):
    task_id = group + "_" + str(index)
    source_report = contained(project, project / "results/slurm_jobs" / task_id / "report")
    submission_path = source_report.parent / "submission.json"
    contract_path = source_report / "training_contract.json"
    result_path = source_report / "training_result.json"
    submission = json.loads(submission_path.read_text())
    status = json.loads((source_report.parent / "job_status.json").read_text())
    contract = json.loads(contract_path.read_text())
    original_result = json.loads(result_path.read_text())
    if status["execution_state"] not in ("COMPLETED", "COMPLETED_WITH_REVIEW") or status["shell_exit_code"] != 0:
        raise ValueError("Source training task has not completed successfully")
    if submission["job_id"] != task_id or submission["stage"] != "reference-train3" or contained(project, submission["project"]) != project:
        raise ValueError("Source task is not a registered reference-training task of this project")
    for key in ("dataset", "seed"):
        if any(row.get(key) != task[key] for row in (submission, contract, original_result)):
            raise ValueError("Source task dataset/seed identity mismatch")
    if contract["config"]["array"] != "stage1" or contract["config"]["route"] != "pretrain":
        raise ValueError("This correction is explicitly defined for stage1/pretrain source runs")
    if original_result["source_array"] != "stage1" or original_result["split_route"] != "pretrain":
        raise ValueError("Source result and training contract name different input protocols")
    if contract["reference_commit"] != lock["reference_commit"]:
        raise ValueError("Source author revision differs from the pinned release")
    source_code = contained(project, submission["code"])
    if file_sha(source_code / "scripts/reference_train.py") != contract["trainer_sha256"]:
        raise ValueError("Original training source no longer matches its contract")
    for name, digest in contract["helper_sha256"].items():
        if file_sha(contained(project, source_code / "scripts" / name)) != digest:
            raise ValueError("Original training helper changed: " + name)
    store = contained(project, original_result["output_store"])
    if json.loads((store / "identity.json").read_text()) != contract:
        raise ValueError("Original run store and task contract differ")
    store_result = json.loads((store / "training_result.json").read_text())
    if store_result != original_result or file_sha(store / "training_result.json") != json.loads((store / "COMPLETED.json").read_text())["result_sha256"]:
        raise ValueError("Original completed result fails integrity verification")
    best = original_result["best"]
    if file_sha(contained(project, store / best["file"])) != best["sha256"]:
        raise ValueError("Validation-selected model hash mismatch")
    profile = PROFILES[task["dataset"]]
    asset = next(row for row in lock["lfs_files"] if row["path"] == profile["stage1"])
    values_path = contained(project, repo / profile["stage1"])
    actual = verify(values_path, asset)
    if not actual or actual["sha256"] != contract["source_sha256"]:
        raise ValueError("Original training array hash mismatch")
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != profile["shape"] or not np.isfinite(values).all():
        raise ValueError("Training array shape/finite-value verification failed")
    minimum, maximum = float(values.min()), float(values.max())
    if (minimum, maximum) != (contract["min"], contract["max"]):
        raise ValueError("Saved normalization extrema differ from verified original array")
    registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")[task["dataset"]]
    splits = derive_splits((repo / "data/unified_dataset.py").read_text(), registry, values, "pretrain")
    if splits != contract["splits"] or splits["test"]["scored_windows"] != original_result["scored_origins"]:
        raise ValueError("Source test origins/splits differ from the recorded training contract")
    scale = registry["finetune"]["max_inflow"]
    expected_batches = splits["test"]["inner_batches"]
    shards = contained(project, store / "test_predictions")
    if len(list(shards.glob("batch-*.npz"))) != expected_batches:
        raise ValueError("Missing or extra original prediction shards")
    identity = {"source_task": task_id, "dataset": task["dataset"], "seed": task["seed"],
        "source_contract_sha256": file_sha(contract_path), "source_result_sha256": file_sha(result_path),
        "selected_model_sha256": best["sha256"], "source_array_sha256": actual["sha256"],
        "minimum": minimum, "maximum": maximum, "legacy_registry_scale": scale,
        "corrected_transform": INVERSE, "clipping": "negative physical predictions clipped to zero after inversion",
        "rescore_script_sha256": file_sha(Path(__file__)),
        "contract_helper_sha256": file_sha(code / "scripts/reference_contract.py")}
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = contained(project, project / "outputs/baseline_rescore" / run_hash)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "identity.json", identity)
    original_hashes = {str(path): file_sha(path) for path in (result_path, contract_path, store / "identity.json", store / "COMPLETED.json")}
    truth_rows, legacy_rows, corrected_rows = [], [], []
    reused = 0
    batch_sources = []
    for batch in range(expected_batches):
        start = splits["test"]["start_target_index"] + batch * 16
        targets = np.arange(start, start + 16, dtype="int64")
        truth = np.array([values[:, i, 0] for i in targets], dtype="float32")
        meta = json.loads((shards / ("batch-%04d.json" % batch)).read_text())
        if meta["batch_index"] != batch:
            raise ValueError("Prediction shard ordinal mismatch")
        original = load_batch(shards, batch, targets, truth, scale)
        if original is None:
            raise ValueError("A required saved normalized prediction shard is missing")
        source_sha = meta["sha256"]
        derived, cached = corrected_shard(output / ("batch-%04d.npz" % batch), original, source_sha,
            minimum, maximum, scale, run_hash)
        reused += cached
        truth_rows.append(derived["truth"])
        legacy_rows.append(derived["predictions_legacy"])
        corrected_rows.append(derived["predictions_corrected"])
        batch_sources.append({"batch": batch, "source_sha256": source_sha})
        if (batch + 1) % 10 == 0 or batch + 1 == expected_batches:
            update(phase="RESCORING", completed_batches=batch + 1, total_batches=expected_batches, reused_batches=reused, output_store=str(output))
    truth, legacy, corrected = map(np.concatenate, (truth_rows, legacy_rows, corrected_rows))
    legacy_metrics = metrics_pair(metrics_module, truth, legacy)
    corrected_metrics = metrics_pair(metrics_module, truth, corrected)
    primary = corrected_metrics if original_result.get("inverse_transform") == INVERSE else legacy_metrics
    if not all(np.isclose(primary["author"][key], original_result["metrics_author"][key], rtol=1e-5, atol=1e-6) for key in ("MAE", "RMSE", "WMAPE_percent")):
        raise ValueError("Replayed source scores differ from the original stored results")
    unchanged_control = minimum == 0 and maximum == scale
    if unchanged_control and not np.array_equal(legacy, corrected):
        raise ValueError("Zero-minimum control changed under equivalent inverse formulas")
    for path, digest in original_hashes.items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Original experiment files changed during rescore")
    # This audit documents source-defined route alternatives; it never selects
    # a route based on test scores or claims to resolve the published S3 policy.
    target_splits = derive_splits((repo / "data/unified_dataset.py").read_text(), registry, values, "finetune")
    protocol = {"original_array": "stage1", "original_route": "pretrain", "source_splits": splits,
        "released_target_route_splits": target_splits,
        "source_train_val_target_overlap": intersection_count(splits["train"], splits["val"]),
        "target_route_train_val_target_overlap": intersection_count(target_splits["train"], target_splits["val"]),
        "scaling_fit_scope": "entire stage-one supplied array, including held-out periods and both channels",
        "cross_window_dependence": "retained in original weights and saved forecasts; rescore cannot repair it",
        "clean_protocol_status": "NOT_ESTABLISHED",
        "next_gate": "Reconcile S3 input/trainability evidence and predeclare matched causal baseline before new-method comparisons"}
    result = {"source_task": task_id, **task, "execution_status": "COMPLETED_WITH_REVIEW", "inverse_transform": INVERSE,
        "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "minimum": minimum, "maximum": maximum,
        "legacy_metrics": legacy_metrics, "corrected_metrics": corrected_metrics,
        "scored_origins": len(truth), "best_epoch": best["epoch"], "training_epochs": original_result["epochs"],
        "selected_model_sha256": best["sha256"], "output_store": str(output), "normalized_predictions_reused": True,
        "training_performed": False, "model_inference_performed": False, "source_outputs_preserved": True,
        "zero_minimum_equivalence_control": unchanged_control, "completed_batches": expected_batches, "reused_batches": reused,
        "mean_prediction_shift_corrected_minus_legacy": float(np.mean(corrected.astype("float64") - legacy)),
        "max_absolute_prediction_shift": float(np.max(np.abs(corrected.astype("float64") - legacy))),
        "source_inputs": identity, "source_shards": batch_sources, "protocol_audit": protocol}
    atomic_json(output / "rescore_result.json", result)
    case_report = report / "tasks" / task_id
    case_report.mkdir(parents=True, exist_ok=True)
    atomic_json(case_report / "rescore_result.json", result)
    (case_report / "SUMMARY.md").write_text("# Saved-forecast inverse-scaling correction\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    return result


def aggregate(results):
    output = {}
    for dataset in sorted({row["dataset"] for row in results}):
        rows = [row for row in results if row["dataset"] == dataset]
        output[dataset] = {"completed_seeds": len(rows), **{variant: {
            key: {"mean": statistics.mean(row[variant]["author"][key] for row in rows),
                  "sample_std": statistics.stdev(row[variant]["author"][key] for row in rows) if len(rows) > 1 else None}
            for key in ("MAE", "RMSE", "WMAPE_percent")} for variant in ("legacy_metrics", "corrected_metrics")}}
    return output


def run(project, code, report, source_array):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    contained(project, code)
    contained(project, report)
    if not re.fullmatch(r"[0-9]+", source_array):
        raise ValueError("A numeric, recorded source training array ID is required")
    report.mkdir(parents=True, exist_ok=True)
    worker_logs = contained(project, os.environ.get("TRAFFIC_WORKER_LOG_DIR", str(report / "worker_logs")))
    worker_logs.mkdir(parents=True, exist_ok=True)
    progress = {"source_array": source_array, "completed_tasks": 0, "failed_tasks": 0, "phase": "VERIFYING_SOURCE", "tasks": {}}
    def update(key=None, **fields):
        (progress["tasks"].setdefault(key, {}) if key else progress).update(fields)
        atomic_json(report / "rescore_progress.json", progress)
        if key:
            with (worker_logs / (key + ".log")).open("a") as stream:
                stream.write(json.dumps(progress["tasks"][key]) + "\n")
        print("BASELINE_RESCORE", json.dumps(progress), flush=True)
    try:
        submission = json.loads((project / "results/slurm_jobs" / source_array / "submission.json").read_text())
        if submission["stage"] != "reference-train3" or submission["job_id"] != source_array or contained(project, submission["project"]) != project:
            raise ValueError("Source array is not registered as reference training in this project")
        tasks = submission["array_tasks"]
        if not tasks or len({(row["dataset"], row["seed"]) for row in tasks}) != len(tasks):
            raise ValueError("Empty/duplicate source task identities")
        progress["total_tasks"] = len(tasks)
        atomic_json(report / "rescore_plan.json", {"source_array": source_array, "tasks": tasks, "transforms": ["legacy max-only", INVERSE],
            "selection": "same original validation-selected weights, normalized forecasts, target rows and clipping; no test-driven selection",
            "training_performed": False, "model_inference_performed": False})
        update()
        lock = json.loads((code / "config/sources.lock.json").read_text())
        repo = prepare_reference(project / "work", lock, audit_only=True)
        expected = next(row for row in lock["lfs_files"] if row["path"] == "utils/metrics.py")
        if not verify(repo / "utils/metrics.py", expected):
            raise ValueError("Pinned author metric source hash mismatch")
        metrics_module = module_from(repo / "utils/metrics.py", "_rescore_author_metrics")
        results, failures = [], []
        for index, task in enumerate(tasks):
            key = source_array + "_" + str(index)
            update(key, phase="VERIFYING", **task)
            try:
                result = rescore_task(project, code, report, source_array, index, task, lock, repo, metrics_module,
                    lambda **kw: update(key, **kw))
                results.append(result)
                progress["completed_tasks"] += 1
                update(key, phase="COMPLETED_WITH_REVIEW", result_summary={name: result[name] for name in
                    ("legacy_metrics", "corrected_metrics", "output_store", "source_outputs_preserved")})
            except Exception as exc:
                failures.append({"task": key, "error": str(exc)})
                case_report = report / "tasks" / key
                case_report.mkdir(parents=True, exist_ok=True)
                (case_report / "failure_traceback.txt").write_text(traceback.format_exc())
                (worker_logs / (key + ".err")).write_text(traceback.format_exc())
                progress["failed_tasks"] += 1
                update(key, phase="FAILED", error=str(exc), traceback=str(case_report / "failure_traceback.txt"))
        summary = {"execution_status": "FAILED" if failures else "COMPLETED_WITH_REVIEW", "source_array": source_array,
            "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "completed_tasks": len(results), "total_tasks": len(tasks),
            "failed_tasks": len(failures), "failures": failures, "tasks": results, "aggregates": aggregate(results),
            "training_performed": False, "model_inference_performed": False,
            "scope": "Inverse-scaling correction of existing single-domain release-behavior baseline; not a clean/causal baseline or exact S3 reproduction"}
        atomic_json(report / "rescore_summary.json", summary)
        lines = ["# Reference baseline inverse-scaling correction", "", "Execution: " + summary["execution_status"], "",
            "Rescored %d/%d tasks; failures %d. No training or model inference." % (len(results), len(tasks), len(failures)), "",
            "| Dataset | Seed | Legacy MAE | Corrected MAE | Legacy RMSE | Corrected RMSE | Legacy WMAPE % | Corrected WMAPE % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in results:
            a, b = row["legacy_metrics"]["author"], row["corrected_metrics"]["author"]
            lines.append("| %s | %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |" %
                (row["dataset"], row["seed"], a["MAE"], b["MAE"], a["RMSE"], b["RMSE"], a["WMAPE_percent"], b["WMAPE_percent"]))
        lines += ["", "Aggregate means and sample standard deviations:", "", "```json", json.dumps(summary["aggregates"], indent=2), "```", "",
            "All original weights, forecasts and reports are preserved. Corrected forecasts use the recorded full-array min/max and the same zero clipping.",
            "Full-array scaling, second-channel provenance and cross-window dependence remain unresolved scientific limitations.",
            "Next: reconcile the single-domain training protocol and establish a matched causal reference baseline before evaluating the proposed method."]
        for row in failures:
            lines += ["", "FAILURE " + row["task"] + ": " + row["error"]]
        (report / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        update(phase=summary["execution_status"])
        print("\n".join(lines), flush=True)
        return 2 if failures else 3
    except Exception as exc:
        update(phase="FAILED", error=str(exc))
        atomic_json(report / "rescore_summary.json", {"execution_status": "FAILED", "error": str(exc), "completed_tasks": 0, "tasks": []})
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        (report / "SUMMARY.md").write_text("# Baseline rescore failed\n\n" + str(exc) + "\n")
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-array", required=True)
    args = parser.parse_args()
    sys.exit(run(args.project, args.code, args.report, args.source_array))
