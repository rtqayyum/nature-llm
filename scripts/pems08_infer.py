#!/usr/bin/env python3
"""Evaluate the released checkpoint on CPU under the pinned author protocol.

Import/paths are supplied by a wrapper; author model/loader/metric source is
unchanged. No alternate layers, dropped keys, corrected splits or fitted data
transformations are selected to chase a published number.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from acquire import atomic_json, prepare_reference, verify
from pems08_audit import derive_splits, literal_assignment
from reference_datasets import PROFILES
from reference_contract import anchor_comparison, atomic_npz, file_sha, independent_metrics, inverse_author, load_batch, save_batch


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_diagnostics(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict):
        raise ValueError("The released checkpoint is not a direct state_dict; no automatic extraction was attempted")
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatched = []
    for name in sorted(set(expected) & set(state)):
        if not hasattr(state[name], "shape") or tuple(expected[name].shape) != tuple(state[name].shape):
            mismatched.append({"key": name, "expected": list(expected[name].shape),
                               "checkpoint": list(state[name].shape) if hasattr(state[name], "shape") else "not a tensor"})
    return {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatched,
            "model_key_count": len(expected), "checkpoint_key_count": len(state),
            "compatible": not (missing or unexpected or mismatched),
            "strict_load_performed": False}


def matching_preflight(project, expected, dataset="pems08", current_report=None):
    slug = PROFILES[dataset]["slug"]
    reports = sorted((project / "results/slurm_jobs").glob("*/report/" + slug + "_audit.json"),
                     key=lambda path: int(path.parents[1].name) if path.parents[1].name.isdigit() else -1, reverse=True)
    if current_report is not None:
        current = current_report / (slug + "_audit.json")
        if current.exists():
            reports.insert(0, current)
    for path in reports:
        item = json.loads(path.read_text())
        if item.get("execution_status") != "COMPLETED_WITH_REVIEW":
            continue
        if all(item.get("assets", {}).get(name, {}).get("sha256") == value["sha256"] for name, value in expected.items()):
            return path, item
    raise ValueError("No completed " + PROFILES[dataset]["title"] + " preflight matches these existing asset hashes")


def run(project, code, report, dataset="pems08"):
    profile = PROFILES[dataset]
    title, slug = profile["title"], profile["slug"]
    nodes = profile["shape"][0]
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    report.mkdir(parents=True, exist_ok=True)
    progress = {"stage": slug + "-infer", "phase": "verify_inputs", "completed_batches": 0, "total_batches": None,
                "completed_origins": 0, "resumed_batches": 0, "inference_performed": False, "training_performed": False}
    def update(phase, **values):
        progress.update(values)
        progress["phase"] = phase
        atomic_json(report / "inference_progress.json", progress)
        print("INFERENCE", json.dumps(progress), flush=True)
    try:
        import torch
        import transformers
        import scipy
        import sklearn
        if (torch.__version__, transformers.__version__, np.__version__, scipy.__version__, sklearn.__version__) != (
                "2.0.0+cpu", "4.35.0", "1.24.0", "1.10.0", "1.2.2"):
            raise ValueError("Reference core package versions differ from the pinned CPU contract")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        np.random.seed(11)
        torch.manual_seed(11)
        lock = json.loads((code / "config/sources.lock.json").read_text())
        repo = prepare_reference(project / "work", lock, audit_only=True)
        selected = [profile["stage2"], profile["checkpoint"], "utils/metrics.py"]
        assets = {}
        for name in selected:
            item = next(item for item in lock["lfs_files"] if item["path"] == name)
            actual = verify(repo / name, item)
            if not actual:
                raise ValueError("Selected existing asset failed its pinned digest: " + name)
            assets[name] = actual
        preflight_path, preflight = matching_preflight(project, assets, dataset, report)
        base = project / "work/models/gpt2" / lock["gpt2"]["revision"]
        for item in lock["gpt2"]["files"]:
            if not verify(base / item["path"], item):
                raise ValueError("Existing GPT-2 file failed verification: " + item["path"])
        update("load_original_modules")
        model_module = module_from(repo / "model/LLM_UTP.py", "_author_llm_utp")
        dataset_module = module_from(repo / "data/unified_dataset.py", "_author_utp_dataset")
        metrics_module = module_from(repo / "utils/metrics.py", "_author_utp_metrics")
        registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")[dataset]
        values = np.load(repo / selected[0], mmap_mode="r", allow_pickle=False)
        if list(values.shape) != profile["shape"]:
            raise ValueError("Verified " + title + " dimensions changed unexpectedly")
        config = {"device": "cpu", "dtype": "float32", "threads": threads, "seed_for_initialization": 11,
            "time_lag": 12, "pre_len": 1, "inner_batch_size": 16, "outer_batch_size": 1,
            "prediction_script_time_interval": 15, "native_interval_minutes": profile["interval_minutes"],
            "time_interval_note": "The released script's 15 is retained as an unused loader attribute; arrays are not resampled",
            "normalization": "author full-array minmax across both channels", "inverse": "prediction * registry max_inflow, negative predictions clipped",
            "dataset": dataset, "checkpoint_strategy": "UNRESOLVED", "reference_commit": lock["reference_commit"],
            "gpt2_base_revision": lock["gpt2"]["revision"], "preflight_report": str(preflight_path)}
        atomic_json(report / "evaluation_config.json", config)
        expected = derive_splits((repo / "data/unified_dataset.py").read_text(), registry, values, "finetune")["test"]
        test_set = dataset_module.UnifiedTrafficDataset(dataset_key=dataset, stage="predict", split="test", time_interval=15,
            tg_in_one_day=registry["finetune"]["tg_in_one_day"], time_lag=12, pre_len=1, inflow_data=values,
            split_config=registry["finetune"]["split_config"], inner_batch_size=16)
        if (test_set.start_index, test_set.end_index, len(test_set) * 16) != (
                expected["start_target_index"], expected["end_target_index_exclusive"], expected["scored_windows"]):
            raise ValueError("Actual author loader origins differ from the independently audited ranges")
        preflight_test = preflight["splits"]["finetune"]["test"]
        if any(expected[key] != preflight_test[key] for key in ("start_target_index", "end_target_index_exclusive", "scored_windows")):
            raise ValueError("Actual evaluation origins differ from completed preflight")
        update("strict_checkpoint_load", total_batches=len(test_set), total_origins=len(test_set) * 16)
        model = model_module.TrafficPredictionModel(nodes, 12, 64, [1, 2, 4], 12, 0, str(base), 768)
        state = torch.load(repo / selected[1], map_location="cpu", weights_only=True)
        diagnostic = strict_diagnostics(model, state)
        atomic_json(report / "strict_load.json", diagnostic)
        if not diagnostic["compatible"]:
            raise ValueError("Strict checkpoint compatibility failed; inspect strict_load.json. No keys were removed or rewritten")
        model.load_state_dict(state, strict=True)
        diagnostic["strict_load_performed"] = True
        diagnostic["model_trainable_parameter_count_from_constructor"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diagnostic["total_model_parameters"] = sum(p.numel() for p in model.parameters())
        diagnostic["checkpoint_key_shapes"] = {name: list(value.shape) for name, value in state.items()}
        atomic_json(report / "strict_load.json", diagnostic)
        del state
        model.eval()
        loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        first_x, _, first_y = next(iter(loader))
        first_x = first_x.to(dtype=torch.float32)
        expected_truth = np.asarray(values[:, test_set.start_index:test_set.start_index + 16, 0].T, dtype="float32")
        if not np.array_equal(first_y.squeeze(0).squeeze(-1).to(torch.float32).numpy(), expected_truth):
            raise ValueError("First author batch target indices differ from direct source values")
        executed = []
        hooks = [block.register_forward_hook(lambda module, inputs, output, index=index: executed.append(index)) for index, block in enumerate(model.gpt2.h)]
        with torch.no_grad():
            first_prediction = model(first_x).squeeze(-1).cpu().numpy()
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            repeated = model(first_x).squeeze(-1).cpu().numpy()
        if executed != [0, 1] or first_prediction.shape != (16, nodes) or not np.isfinite(first_prediction).all():
            raise ValueError("First forward pass does not match the pinned two-block/shape contract")
        repeated_difference = float(np.abs(first_prediction - repeated).max())
        if not np.allclose(first_prediction, repeated, rtol=1e-5, atol=1e-6):
            raise ValueError("Repeated evaluation of the first batch is numerically inconsistent")
        runtime = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": sys.version, "platform": platform.platform(), "device": "cpu", "cpu_threads": torch.get_num_threads(),
            "retained_gpt_blocks": len(model.gpt2.h), "executed_gpt_block_indices": executed,
            "first_forward_repeat_max_absolute_difference": repeated_difference, "torch_config": torch.__config__.show()}
        atomic_json(report / "runtime.json", runtime)
        freeze = report / "reference_environment_freeze.txt"
        identity = {"config": {key: value for key, value in config.items() if key != "preflight_report"},
            "assets": {key: value["sha256"] for key, value in assets.items()}, "environment_freeze": file_sha(freeze),
            "inference_source": file_sha(Path(__file__)), "contract_source": file_sha(code / "scripts/reference_contract.py"),
            "dataset_profile": profile}
        run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        store = project / "outputs" / (slug + "_reference_cpu") / run_hash
        store.mkdir(parents=True, exist_ok=True)
        atomic_json(store / "identity.json", identity)
        update("forecasting", output_store=str(store), strict_load=True, inference_performed=True)
        started = time.monotonic()
        resumed = 0
        scale = registry["finetune"]["max_inflow"]
        for index, (x, _, y) in enumerate(loader):
            targets = np.arange(test_set.start_index + index * 16, test_set.start_index + (index + 1) * 16, dtype="int64")
            truth = y.squeeze(0).squeeze(-1).to(torch.float32).numpy()
            saved = load_batch(store, index, targets, truth, scale)
            if saved is None:
                if index == 0:
                    predicted = first_prediction
                else:
                    with torch.no_grad():
                        predicted = model(x.to(dtype=torch.float32)).squeeze(-1).cpu().numpy()
                if predicted.shape != (16, nodes) or not np.isfinite(predicted).all():
                    raise ValueError("Invalid prediction in batch " + str(index))
                save_batch(store, index, targets, truth, predicted, scale)
            else:
                resumed += 1
            update("forecasting", completed_batches=index + 1, completed_origins=(index + 1) * 16,
                   resumed_batches=resumed, forecasting_elapsed_seconds=time.monotonic() - started)
        update("independent_metric_recalculation")
        all_truth, all_prediction, all_normalized, all_targets = [], [], [], []
        # Re-read the persistent shards and reconstruct their target values
        # directly from the original array, independently of DataLoader.
        for index in range(len(test_set)):
            targets = np.arange(test_set.start_index + index * 16, test_set.start_index + (index + 1) * 16, dtype="int64")
            direct_truth = np.asarray(values[:, targets, 0].T, dtype="float32")
            shard = load_batch(store, index, targets, direct_truth, scale)
            if shard is None:
                raise ValueError("A required prediction shard is missing")
            all_targets.append(targets)
            all_truth.append(shard["truth"])
            all_prediction.append(shard["predictions"])
            all_normalized.append(shard["normalized_predictions"])
        truth, prediction = np.concatenate(all_truth), np.concatenate(all_prediction)
        targets = np.concatenate(all_targets)
        if len(np.unique(targets)) != expected["scored_windows"] or not np.array_equal(targets, np.arange(expected["start_target_index"], expected["scored_end_target_index_exclusive"])):
            raise ValueError("Final prediction coverage differs from the fixed author scoring origins")
        rmse, r2, mae, wmape = metrics_module.Metrics(truth, prediction).evaluate_performance()
        measured = {"MAE": float(mae), "RMSE": float(rmse), "WMAPE_percent": float(wmape) * 100, "R2": float(r2)}
        independent = independent_metrics(truth, prediction)
        if not all(np.isclose(measured[key], value, rtol=1e-5, atol=1e-6) for key, value in independent.items()):
            raise ValueError("Author and independent metric implementations disagree beyond declared numeric tolerance")
        atomic_npz(store / "forecasts.npz", target_indices=targets, truth=truth, predictions=prediction, normalized_predictions=np.concatenate(all_normalized))
        result = {"execution_status": "COMPLETED_WITH_REVIEW", "scientific_status": "REPRODUCTION_NOT_ESTABLISHED",
            "dataset": dataset, "checkpoint_strategy": "UNRESOLVED", "inference_performed": True, "training_performed": False,
            "strict_load_performed": True, "device": "cpu", "metrics_author": measured, "metrics_independent": independent,
            "paper_anchor_comparisons": anchor_comparison(measured, profile["anchors"]), "scored_origins": len(targets), "nodes": truth.shape[1],
            "scored_target_cells": int(truth.size), "author_test_tail_dropped": expected["dropped_tail_windows"],
            "output_store": str(store), "forecast_file_sha256": file_sha(store / "forecasts.npz"), "run_identity": run_hash,
            "limitations": ["CPU diagnostic evaluation; reference paper used a GPU and its complete runtime/seed history is unknown",
                "Training strategy/checkpoint identity and absolute test timestamps remain unresolved",
                "Original full-array scaling, mask conventions and dropped tail are preserved; this is not a clean-protocol result",
                "Rounding agreement with a paper row alone does not establish reproduction"]}
        atomic_json(report / "inference_result.json", result)
        atomic_json(store / "completed.json", result)
        update("COMPLETED_WITH_REVIEW")
        text = "# " + title + " released checkpoint CPU evaluation\n\nExecution: COMPLETED_WITH_REVIEW\n\nScientific status: REPRODUCTION_NOT_ESTABLISHED\n\n"
        text += "Strict checkpoint load: passed. Inference completed on %d origins x %d nodes.\n\n" % (len(targets), truth.shape[1])
        text += "| Metric | Author function | Independent calculation |\n|---|---:|---:|\n"
        text += "\n".join("| %s | %.8f | %.8f |" % (key, measured[key], value) for key, value in independent.items())
        text += "\n\nForecast store: " + str(store) + "\n\n" + "\n".join("- " + item for item in result["limitations"]) + "\n"
        (report / "SUMMARY.md").write_text(text)
        print(text, flush=True)
        return 3
    except Exception as exc:
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        update("FAILED", error=str(exc))
        atomic_json(report / "inference_result.json", {"execution_status": "FAILED", "error": str(exc), "scientific_status": "REPRODUCTION_NOT_ESTABLISHED"})
        (report / "SUMMARY.md").write_text("# " + title + " inference failed\n\n" + str(exc) + "\n\nExisting inputs and completed prediction shards were preserved.\n")
        print("INFERENCE_FAILED:", str(exc), flush=True)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--dataset", choices=list(PROFILES), default="pems08")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        from pems08_audit import run as preflight
        status = preflight(args.project, args.code, args.report, args.dataset)
        if status != 3:
            sys.exit(status)
        summary = args.report / "SUMMARY.md"
        (args.report / "preflight_SUMMARY.md").write_bytes(summary.read_bytes())
    sys.exit(run(args.project, args.code, args.report, args.dataset))
