#!/usr/bin/env python3
"""Read-only PEMS08 checkpoint/split preflight. No Torch import or unpickling.

The pinned loader's integer range methods are evaluated without importing its
Torch modules. Tensor values, strict model loading and prediction remain later
steps. Released preprocessing is documented without silently repairing it.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickletools
import re
import sys
import zipfile

import numpy as np

from acquire import atomic_json, prepare_reference, verify


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing literal assignment: " + name)


def loader_class(source):
    names = {"_total_len", "_build_start_end", "_ratio_tail", "_ratio_window", "__len__"}
    original = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "UnifiedTrafficDataset")
    functions = [copy.deepcopy(node) for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Pinned loader's split methods are incomplete")
    namespace = {"__builtins__": {"int": int, "float": float, "len": len, "ValueError": ValueError}}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, "pinned_author_split_methods", "exec"), namespace)
    return type("AuthorSplitAudit", (), {name: namespace[name] for name in names})


def derive_splits(source, registry, values, stage, lookback=12, horizon=1, batch=16):
    cls = loader_class(source)
    rows = {}
    for split in ("train", "val", "test"):
        instance = cls()
        instance.inflow_data = values
        instance.split_config = registry[stage]["split_config"]
        instance.time_lag, instance.pre_len = lookback, horizon
        instance.inner_batch_size, instance.split = batch, split
        start, end = instance._build_start_end()
        start, end = max(0, start), min(end, values.shape[1])
        if start < lookback or end > values.shape[1] - horizon + 1 or end <= start:
            raise ValueError("Author split has an invalid input/target boundary")
        instance.X = range(end - start)
        kept = len(instance) * batch
        rows[split] = {"start_target_index": start, "end_target_index_exclusive": end,
            "constructed_windows": end - start, "inner_batches": len(instance), "scored_windows": kept,
            "scored_end_target_index_exclusive": start + kept, "dropped_tail_windows": end - start - kept,
            "lookback_steps": lookback, "horizon_steps": horizon, "inner_batch_size": batch,
            "time_coordinates": "row indices only; absolute timestamps are not verified"}
    return rows


def intersection_count(left, right, scored=True):
    end = "scored_end_target_index_exclusive" if scored else "end_target_index_exclusive"
    return max(0, min(left[end], right[end]) - max(left["start_target_index"], right["start_target_index"]))


def checkpoint_metadata(path):
    if not zipfile.is_zipfile(path):
        raise ValueError("Checkpoint is not the supported PyTorch ZIP format; no deserialization was attempted")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [name for name in names if name.endswith("/data.pkl") or name == "data.pkl"]
        if len(pickles) != 1:
            raise ValueError("Checkpoint must contain exactly one data.pkl")
        info = archive.getinfo(pickles[0])
        if info.file_size > 16 * 2**20:
            raise ValueError("Checkpoint metadata pickle exceeds the static inspection limit")
        # pickletools disassembles opcodes. It never invokes GLOBAL/REDUCE targets.
        strings, globals_seen = [], []
        for opcode, argument, _ in pickletools.genops(archive.read(pickles[0])):
            if opcode.name in ("UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"):
                strings.append(argument)
            elif opcode.name == "GLOBAL":
                globals_seen.append(argument)
        layers = sorted({int(match[1]) for value in strings if (match := re.match(r"^(?:module\.)?gpt2\.h\.(\d+)(?:\.|$)", value))})
        candidates = sorted({value for value in strings if "." in value and value.endswith((".weight", ".bias"))})
        storage = [member for member in archive.infolist() if re.search(r"/data/\d+$", member.filename)]
        return {"format": "PyTorch ZIP", "inspection": "static pickle opcode inspection only",
            "pickle_executed": False, "tensor_values_loaded": False, "strict_model_load_performed": False,
            "gpt_block_indices_in_serialized_names": layers, "parameter_name_candidates": candidates,
            "candidate_parameter_names_count": len(candidates), "storage_records": len(storage),
            "storage_bytes": sum(member.file_size for member in storage), "pickle_globals": sorted(set(globals_seen)),
            "checkpoint_strategy": "UNRESOLVED: filename/parameter names do not identify S1, S2 or S3",
            "tensor_shapes": "not decoded by this static preflight"}


def channel_stats(values):
    if values.ndim != 3 or values.shape[0] != 170 or values.shape[2] != 2:
        raise ValueError("Expected PEMS08 node,time,feature array with 170 nodes and two channels")
    result = []
    for channel in range(values.shape[2]):
        data = values[:, :, channel]
        if not np.isfinite(data).all():
            raise ValueError("PEMS08 contains nonfinite values")
        result.append({"channel": channel, "minimum": float(data.min()), "maximum": float(data.max()),
            "zeros": int((data == 0).sum()), "negative_values": int((data < 0).sum())})
    return result


def write_origins(path, row):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["forecast_origin_index", "input_start", "input_end_exclusive", "target_index", "included_by_author_inner_batches"])
        for target in range(row["start_target_index"], row["end_target_index_exclusive"]):
            writer.writerow([target - 1, target - row["lookback_steps"], target, target,
                             int(target < row["scored_end_target_index_exclusive"])])


def run(project, code, report):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    report.mkdir(parents=True, exist_ok=True)
    progress = {"stage": "pems08-audit", "completed_checks": 0, "total_checks": 5, "checks": []}
    def complete(label):
        progress["checks"].append(label)
        progress["completed_checks"] += 1
        atomic_json(report / "preflight_progress.json", progress)
        print("PREFLIGHT checks=%d/%d completed=%s" % (progress["completed_checks"], progress["total_checks"], label), flush=True)
    try:
        lock = json.loads((code / "config/sources.lock.json").read_text())
        repo = prepare_reference(project / "work", lock, audit_only=True)
        paths = ["data/stage1/08.npy", "data/stage2/08fine-tuning.npy", "checkpoint/08/best_model_dict.pth", "utils/metrics.py"]
        assets = {}
        for name in paths:
            expected = next(item for item in lock["lfs_files"] if item["path"] == name)
            actual = verify(repo / name, expected)
            if not actual:
                raise ValueError("Pinned file is missing or fails integrity verification: " + name)
            assets[name] = {**actual, "path": str(repo / name)}
        complete("reference_and_selected_asset_hashes")
        stage1 = np.load(repo / paths[0], mmap_mode="r", allow_pickle=False)
        stage2 = np.load(repo / paths[1], mmap_mode="r", allow_pickle=False)
        arrays = {"stage1": {"shape": list(stage1.shape), "dtype": str(stage1.dtype), "channels": channel_stats(stage1)},
                  "stage2": {"shape": list(stage2.shape), "dtype": str(stage2.dtype), "channels": channel_stats(stage2)}}
        if stage1.shape != stage2.shape:
            raise ValueError("Stage-one/stage-two shapes differ; resolve before comparing their origin indices")
        arrays["stage_comparison"] = {"channel_0_identical": bool(np.array_equal(stage1[..., 0], stage2[..., 0])),
            "channel_1_identical": bool(np.array_equal(stage1[..., 1], stage2[..., 1])),
            "raw_channel_max_absolute_difference": float(np.abs(stage1[..., 0].astype("float64") - stage2[..., 0]).max())}
        complete("pems08_arrays_and_stage_alignment")
        registry_source = (repo / "data/dataset_registry.py").read_text()
        registry = literal_assignment(registry_source, "DATASET_REGISTRY")["pems08"]
        loader_source = (repo / "data/unified_dataset.py").read_text()
        splits = {"pretrain": derive_splits(loader_source, registry, stage1, "pretrain"),
                  "finetune": derive_splits(loader_source, registry, stage2, "finetune")}
        origins = report / "origin_ledgers"
        origins.mkdir(exist_ok=True)
        for stage, rows in splits.items():
            for split, row in rows.items():
                path = origins / (stage + "_" + split + ".csv")
                write_origins(path, row)
                row["origin_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        overlaps = {"finetune_train_val_constructed_target_origins": intersection_count(splits["finetune"]["train"], splits["finetune"]["val"], False),
            "finetune_train_val_scored_target_origins": intersection_count(splits["finetune"]["train"], splits["finetune"]["val"]),
            "pretrain_train_finetune_test_scored_target_origins": intersection_count(splits["pretrain"]["train"], splits["finetune"]["test"]),
            "test_start_offset_finetune_minus_pretrain": splits["finetune"]["test"]["start_target_index"] - splits["pretrain"]["test"]["start_target_index"]}
        complete("exact_author_split_and_tail_ledgers")
        minimum, maximum = float(stage2.min()), float(stage2.max())
        registry_max = registry["finetune"]["max_inflow"]
        scaling = {"author_normalization": "(full_array - full_array.min()) / (full_array.max() - full_array.min())",
            "fit_scope": "entire supplied stage array, including held-out periods and both channels",
            "stage2_global_minimum": minimum, "stage2_global_maximum": maximum,
            "prediction_script_inverse": "normalized_prediction * registry_max_inflow, then clamp negatives to zero",
            "registry_max_inflow": registry_max,
            "inverse_matches_minmax_algebra": minimum == 0.0 and maximum == registry_max,
            "wmape": "sum absolute errors only where truth > 0, divided by total truth sum; MAE/RMSE use all cells",
            "timestamps": "not supplied in the .npy payload; absolute dates and gaps remain unverified"}
        first = splits["finetune"]["test"]["start_target_index"]
        inputs = np.asarray(stage2[:, first - 12:first, :])
        normalized = (inputs - minimum) / (maximum - minimum) if maximum != minimum else np.zeros_like(inputs)
        np.savez_compressed(report / "first_test_window.npz", raw_inputs=inputs, normalized_inputs=normalized,
                            raw_targets=np.asarray(stage2[:, first:first + 1, 0]), target_index=first)
        complete("scaling_metrics_and_one_debug_window")
        checkpoint = checkpoint_metadata(repo / paths[2])
        complete("checkpoint_static_inventory")
        observations = ["This is a verified-release preflight, not checkpoint inference or training reproduction.",
            "Checkpoint strategy S1/S2/S3 and original GPT-2 initialization provenance remain unresolved.",
            "Released constructor retains six GPT-2 blocks; its forward loop executes two. Strict tensor loading is not tested here.",
            "The release normalizes the supplied full array; faithful and clean evaluation must be reported separately.",
            "All split coordinates are indices, not verified timestamps. Do not infer dates or repair gaps silently.",
            "Published S1 anchors are MAE 11.87, RMSE 18.53 and WMAPE 5.16%; no metric is reproduced in this job."]
        if overlaps["finetune_train_val_scored_target_origins"]:
            observations.append("The released finetune split routes overlap train/validation target indices; report and resolve this before independent training claims.")
        if not scaling["inverse_matches_minmax_algebra"]:
            observations.append("The prediction script's inverse scaling differs from the algebraic inverse of its loader normalization; do not silently substitute a corrected scaler.")
        result = {"stage": "pems08-audit", "execution_status": "COMPLETED_WITH_REVIEW",
            "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "reference_commit": lock["reference_commit"],
            "assets": assets, "arrays": arrays, "registry": registry, "splits": splits, "overlaps": overlaps,
            "scaling_and_metrics": scaling, "checkpoint": checkpoint, "observations": observations,
            "inference_performed": False, "training_performed": False, "downloads_performed": False,
            "proposed_prediction_task": "12 native 5-minute observations -> next 5-minute node flow"}
        atomic_json(report / "pems08_audit.json", result)
        lines = ["# PEMS08 checkpoint and split preflight", "", "Execution: COMPLETED_WITH_REVIEW", "Scientific status: REPRODUCTION_NOT_ESTABLISHED", "",
            "| Route | Split | Start | End exclusive | Constructed | Scored | Dropped |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for stage, rows in splits.items():
            for split, row in rows.items():
                lines.append("| %s | %s | %d | %d | %d | %d | %d |" % (stage, split, row["start_target_index"], row["end_target_index_exclusive"], row["constructed_windows"], row["scored_windows"], row["dropped_tail_windows"]))
        lines += ["", "## Target overlaps", "", json.dumps(overlaps, indent=2), "", "## Scaling", "", json.dumps(scaling, indent=2),
            "", "## Checkpoint", "", "Stored GPT block indices from static metadata: " + str(checkpoint["gpt_block_indices_in_serialized_names"]),
            "No tensor deserialization, strict load, inference or training was performed.", "", "## Interpretation", ""] + ["- " + item for item in observations]
        lines += ["", "Details: pems08_audit.json; origin_ledgers/; first_test_window.npz.", "Next: review these discrepancies and implement strictly identified checkpoint inference.", ""]
        (report / "SUMMARY.md").write_text("\n".join(lines))
        print((report / "SUMMARY.md").read_text(), flush=True)
        return 3
    except Exception as exc:
        atomic_json(report / "pems08_audit.json", {"execution_status": "FAILED", "error": str(exc), "scientific_status": "REPRODUCTION_NOT_ESTABLISHED"})
        (report / "SUMMARY.md").write_text("# PEMS08 preflight failed\n\n" + str(exc) + "\n")
        print("PREFLIGHT_FAILED:", str(exc), flush=True)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.project, args.code, args.report))
