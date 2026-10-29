#!/usr/bin/env python3
"""Fixed released-input/split controls; never select a baseline by test score."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import zipfile

import numpy as np

from acquire import atomic_json, prepare_reference, verify
from pems08_audit import derive_splits, literal_assignment
from reference_contract import file_sha, independent_metrics
from reference_datasets import PROFILES

TARGETS = ("pems04", "sh", "bj")
VARIANTS = ("stage2-finetune", "stage1-finetune", "stage2-pretrain", "stage1-pretrain")


def inventory_source_data(project, code, report):
    lock = json.loads((code / "config/sources.lock.json").read_text())
    asset = next(row for row in lock["lfs_files"] if row["path"] == "Source Data/Source Data.zip")
    path = project / "work/reference/snapshots" / lock["reference_commit"] / asset["path"]
    path.resolve().relative_to(project)
    actual = verify(path, asset)
    if not actual:
        raise ValueError("Existing Source Data archive fails its pinned digest")
    with zipfile.ZipFile(path) as archive:
        rows = [{"name": row.filename, "bytes": row.file_size, "compressed_bytes": row.compress_size}
                for row in archive.infolist()]
    result = {"sha256": actual["sha256"], "member_count": len(rows), "members": rows,
        "inspection": "verified archive directory only; no extraction or file execution"}
    atomic_json(report / "source_data_inventory.json", result)
    print("SOURCE_DATA archive members=%d; inventory saved" % len(rows), flush=True)
    return result


def make_plan(project, code):
    lock = json.loads((code / "config/sources.lock.json").read_text())
    repo = prepare_reference(project / "work", lock, audit_only=True)
    registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")
    source = (repo / "data/unified_dataset.py").read_text()
    cases, aliases, assets = [], {}, {}
    for key in TARGETS:
        profile = PROFILES[key]
        arrays = {}
        for field in ("stage1", "stage2", "checkpoint"):
            name = profile[field]
            expected = next(item for item in lock["lfs_files"] if item["path"] == name)
            actual = verify(repo / name, expected)
            if not actual:
                raise ValueError("Existing diagnostic asset fails its pinned digest: " + name)
            assets[name] = actual["sha256"]
            if field != "checkpoint":
                arrays[field] = np.load(repo / name, mmap_mode="r", allow_pickle=False)
                if list(arrays[field].shape) != profile["shape"]:
                    raise ValueError("Diagnostic array shape differs from its released profile")
        if not np.array_equal(arrays["stage1"][..., 0], arrays["stage2"][..., 0]):
            raise ValueError("Raw target channels differ; this fixed paired-input diagnostic is inapplicable")
        seen = {}
        for variant in VARIANTS:
            stage, route = variant.split("-")
            row = derive_splits(source, registry[key], arrays[stage], route)["test"]
            case_id = key + "__" + variant
            # In the pinned author loader, stage is a label, while these
            # coordinates, batch size and supplied array determine all inputs.
            signature = (stage, row["start_target_index"], row["end_target_index_exclusive"], row["scored_windows"])
            if signature in seen:
                aliases[case_id] = seen[signature]
                continue
            seen[signature] = case_id
            cases.append({"case_id": case_id, "dataset": key, "variant": variant, "split": row})
    return {"plan_version": 1, "cases": cases, "aliases": aliases,
        "assets": assets, "reference_commit": lock["reference_commit"],
        "factors": ["released supplied array: stage1/stage2", "released split route: pretrain/finetune"],
        "checkpoint_policy": "same strictly loaded target checkpoint in every case",
        "selection_policy": "fixed source-defined cases; report all results; no metric-based selection or replacement baseline"}


def read_forecasts(project, result):
    path = Path(result["output_store"]) / "forecasts.npz"
    path.resolve().relative_to(project.resolve())
    if file_sha(path) != result["forecast_file_sha256"]:
        raise ValueError("Diagnostic forecast archive fails its recorded digest")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("target_indices", "truth", "predictions")}


def paired_comparison(left, right):
    common, first, second = np.intersect1d(left["target_indices"], right["target_indices"], return_indices=True)
    if not len(common) or not np.array_equal(left["truth"][first], right["truth"][second]):
        raise ValueError("Paired diagnostic has no common targets or its raw truths disagree")
    delta = np.abs(left["predictions"][first].astype("float64") - right["predictions"][second])
    return {"common_origins": len(common), "common_target_indices_sha256": hashlib.sha256(common.tobytes()).hexdigest(),
        "mean_absolute_prediction_difference": float(delta.mean()), "max_absolute_prediction_difference": float(delta.max()),
        "left_common_metrics": independent_metrics(left["truth"][first], left["predictions"][first]),
        "right_common_metrics": independent_metrics(right["truth"][second], right["predictions"][second])}


def run(project, code, report):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    report.mkdir(parents=True, exist_ok=True)
    plan = make_plan(project, code)
    inventory = inventory_source_data(project, code, report)
    plan["source_data_archive_sha256"] = inventory["sha256"]
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    atomic_json(report / "diagnostic_plan.json", plan)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "12"))
    workers = min(3, max(1, cpus // 4))
    threads = max(1, cpus // workers)
    job_id, name = os.environ.get("SLURM_JOB_ID", "synthetic"), os.environ.get("SLURM_JOB_NAME", "nutp-release-diag")
    logs = project / "logs" / (name + "_" + job_id + "_workers")
    logs.resolve().relative_to(project)
    logs.mkdir(parents=True, exist_ok=True)
    progress = {"phase": "RUNNING", "total_cases": len(plan["cases"]), "completed_cases": 0, "failed_cases": 0,
        "parallel_workers": workers, "threads_per_worker": threads,
        "cases": {row["case_id"]: {"state": "PENDING"} for row in plan["cases"]}}
    mutex = threading.Lock()
    def update(case_id=None, **values):
        with mutex:
            if case_id:
                progress["cases"][case_id].update(values)
            progress["completed_cases"] = sum(row["state"] == "COMPLETED_WITH_REVIEW" for row in progress["cases"].values())
            progress["failed_cases"] = sum(row["state"] == "FAILED" for row in progress["cases"].values())
            atomic_json(report / "diagnostic_progress.json", progress)
            print("DIAGNOSTIC completed=%d/%d failed=%d latest=%s" % (progress["completed_cases"], len(plan["cases"]),
                progress["failed_cases"], case_id or "setup"), flush=True)
    def evaluate(case):
        key, variant, case_id = case["dataset"], case["variant"], case["case_id"]
        destination = report / "cases" / case_id
        destination.mkdir(parents=True, exist_ok=True)
        freeze = report / "reference_environment_freeze.txt"
        (destination / freeze.name).write_bytes(freeze.read_bytes())
        output, error = logs / (case_id + ".out"), logs / (case_id + ".err")
        update(case_id, state="RUNNING", report=str(destination), stdout=str(output), stderr=str(error))
        try:
            env = {**os.environ, "SLURM_CPUS_PER_TASK": str(threads), "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}
            with output.open("a") as out, error.open("a") as err:
                child = subprocess.run([sys.executable, "-u", str(code / "scripts/pems08_infer.py"),
                    "--project", str(project), "--code", str(code), "--report", str(destination),
                    "--dataset", key, "--preflight", "--release-variant", variant], env=env, stdout=out, stderr=err)
            path = destination / "inference_result.json"
            measured = json.loads(path.read_text()) if path.exists() else {}
            if (child.returncode != 3 or measured.get("strict_load_performed") is not True or
                measured.get("release_variant") != variant or measured.get("dataset") != key):
                raise ValueError(measured.get("error") or "Case did not complete strict diagnostic inference; inspect worker logs")
            measured.update(case_id=case_id, report=str(destination), state="COMPLETED_WITH_REVIEW")
            arrays = read_forecasts(project, measured)
            expected = case["split"]
            targets = np.arange(expected["start_target_index"], expected["scored_end_target_index_exclusive"], dtype="int64")
            if not np.array_equal(arrays["target_indices"], targets):
                raise ValueError("Saved targets differ from the frozen diagnostic plan")
            measured["configuration"] = json.loads((destination / "evaluation_config.json").read_text())
            update(case_id, state="COMPLETED_WITH_REVIEW")
            return measured
        except Exception as exc:
            update(case_id, state="FAILED", reason=str(exc))
            return {"case_id": case_id, "dataset": key, "release_variant": variant, "state": "FAILED",
                "error": str(exc), "report": str(destination)}
    update()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, case): case["case_id"] for case in plan["cases"]}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                results[case_id] = future.result()
            except Exception as exc:
                results[case_id] = {"case_id": case_id, "state": "FAILED", "error": str(exc)}
                update(case_id, state="FAILED", reason=str(exc))
    pairs = []
    for key in TARGETS:
        def resolve(variant):
            name = key + "__" + variant
            return plan["aliases"].get(name, name)
        for factor, first, second in (
            ("supplied_array_at_finetune_origins", "stage2-finetune", "stage1-finetune"),
            ("supplied_array_at_pretrain_origins", "stage2-pretrain", "stage1-pretrain"),
            ("split_route_with_stage2_input", "stage2-finetune", "stage2-pretrain"),
            ("split_route_with_stage1_input", "stage1-finetune", "stage1-pretrain")):
            left, right = resolve(first), resolve(second)
            row = {"dataset": key, "factor": factor, "left": left, "right": right}
            if results[left]["state"] == results[right]["state"] == "COMPLETED_WITH_REVIEW":
                try:
                    row.update(paired_comparison(read_forecasts(project, results[left]), read_forecasts(project, results[right])))
                except Exception as exc:
                    row["error"] = str(exc)
            else:
                row["error"] = "A required paired case failed"
            pairs.append(row)
    failed = bool(progress["failed_cases"] or any("error" in row for row in pairs))
    progress["phase"] = "FAILED" if failed else "COMPLETED_WITH_REVIEW"
    update()
    summary = {"execution_status": progress["phase"], "scientific_status": "REPRODUCTION_NOT_ESTABLISHED",
        "training_experiments_completed": 0, "plan_sha256": plan["plan_sha256"], "cases": results, "aliases": plan["aliases"],
        "paired_comparisons": pairs, "selection_policy": plan["selection_policy"],
        "limitations": ["Same released checkpoint in every case; training strategy is not identified",
            "Changing a supplied array can change both its second channel and its full-array normalization; compare recorded extrema",
            "Changing a split route changes the scored cohort and potentially inner-batch grouping; common-origin scores are reported separately",
            "No arbitrary offsets, scaler fitting, layer changes, metric tuning or baseline replacement",
            "Matching a paper row does not establish which released input or split was used in the published experiment"]}
    atomic_json(report / "diagnostic_result.json", summary)
    lines = ["# Released-input and split diagnostics", "", "Execution: " + progress["phase"], "",
        "Fixed variants only. All comparisons retained; no best-matching baseline selected.", "",
        "| Case | MAE | RMSE | WMAPE % | Origins | State |", "|---|---:|---:|---:|---:|---|"]
    for case in plan["cases"]:
        row = results[case["case_id"]]
        if row["state"] != "COMPLETED_WITH_REVIEW":
            lines.append("| %s | — | — | — | — | FAILED |" % case["case_id"])
            continue
        metric = row["metrics_author"]
        lines.append("| %s | %.8f | %.8f | %.8f | %d | completed with review |" %
            (case["case_id"], metric["MAE"], metric["RMSE"], metric["WMAPE_percent"], row["scored_origins"]))
    lines += ["", "Identical input/range cases are aliases, not additional model runs: " + json.dumps(plan["aliases"]),
        "", "Paired common-origin comparisons: diagnostic_result.json.", "", "## Limitations", ""]
    lines += ["- " + value for value in summary["limitations"]]
    for row in results.values():
        if row["state"] == "FAILED":
            lines += ["", "FAILED " + row["case_id"] + ": " + row["error"]]
    for row in pairs:
        if "error" in row:
            lines += ["", "FAILED paired comparison " + row["dataset"] + " " + row["factor"] + ": " + row["error"]]
    lines += ["", "Recovery: resubmit release-diagnose; compatible saved prediction shards are verified and reused.", ""]
    (report / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return 2 if failed else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        sys.exit(run(args.project, args.code, args.report))
    except Exception as exc:
        args.report.resolve().relative_to(args.project.resolve())
        args.report.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report / "diagnostic_result.json", {"execution_status": "FAILED", "error": str(exc),
            "cases": {}, "aliases": {}, "paired_comparisons": [], "training_experiments_completed": 0})
        (args.report / "failure_traceback.txt").write_text(traceback.format_exc())
        (args.report / "SUMMARY.md").write_text("# Release diagnostic failed\n\n" + str(exc) + "\n")
        print("DIAGNOSTIC_FAILED:", str(exc), flush=True)
        sys.exit(2)
