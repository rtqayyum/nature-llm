#!/usr/bin/env python3
"""Checkpoint batch-dependence and gradient probes, with zero weight updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import traceback

import numpy as np

from acquire import atomic_json, prepare_reference, verify
from pems08_audit import derive_splits, literal_assignment
from pems08_infer import module_from, strict_diagnostics
from reference_contract import atomic_npz, inverse_author
from reference_datasets import PROFILES, PUBLIC_ORDER


def future_mask(batch=16, lookback=12):
    # Target T+j has input rows T+j-lookback ... T+j-1.
    # The first forecast is issued at T-1; rows >= T are future to it.
    return np.arange(batch)[:, None] - lookback + np.arange(lookback)[None, :] >= 0


def probe(model, evaluation_inputs, training_inputs, training_targets, scale):
    import torch
    import torch.nn.functional as functional
    if evaluation_inputs.ndim != 5 or evaluation_inputs.shape[0:2] != (1, 16) or evaluation_inputs.shape[-2:] != (12, 2):
        raise ValueError("Expected original outer-1/inner-16/12-step/two-channel inputs")
    versions = {name: value._version for name, value in model.named_parameters()}
    model.eval()
    calls = []
    original_softmax = functional.softmax
    def traced(value, dim=None, _stacklevel=3, dtype=None):
        if dim is None:
            calls.append({"shape": list(value.shape), "effective_dimension_in_torch_2": 0 if value.ndim in (0, 1, 3) else 1})
        return original_softmax(value, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    try:
        functional.softmax = traced
        with torch.no_grad():
            baseline = model(evaluation_inputs).detach()
    finally:
        functional.softmax = original_softmax
    with torch.no_grad():
        repeat = model(evaluation_inputs).detach()
    if not torch.allclose(baseline, repeat, rtol=1e-5, atol=1e-6):
        raise ValueError("Repeated unchanged evaluation differs beyond numeric tolerance")
    mask = torch.from_numpy(future_mask()).to(evaluation_inputs.device)[None, :, None, :, None]
    cases, saved = [], {"future_time_mask": future_mask(), "baseline_prediction": baseline.cpu().numpy()}
    for fill in (0.0, 1.0):
        altered = torch.where(mask, torch.full_like(evaluation_inputs, fill), evaluation_inputs)
        if not torch.equal(altered[:, 0], evaluation_inputs[:, 0]) or not torch.equal(altered.masked_select(~mask), evaluation_inputs.masked_select(~mask)):
            raise ValueError("Probe changed the fixed first history or an observation available at its issue time")
        with torch.no_grad():
            predicted = model(altered).detach()
        if not torch.isfinite(predicted).all():
            raise ValueError("Probe produced nonfinite predictions")
        first, changed = baseline[0].cpu().numpy(), predicted[0].cpu().numpy()
        delta = np.abs(changed.astype("float64") - first)
        flow_delta = np.abs(inverse_author(changed, scale).astype("float64") - inverse_author(first, scale))
        cases.append({"future_fill_normalized": fill, "first_history_unchanged": True, "all_available_past_unchanged": True,
            "max_absolute_normalized_difference": float(delta.max()), "mean_absolute_flow_difference": float(flow_delta.mean()),
            "max_absolute_flow_difference": float(flow_delta.max()),
            "changed_first_output_values": int(np.count_nonzero(~np.isclose(changed, first, rtol=1e-5, atol=1e-6))),
            "changed_clipped_forecast_values": int(np.count_nonzero(~np.isclose(inverse_author(changed, scale),
                inverse_author(first, scale), rtol=1e-5, atol=1e-6 * scale)))})
        saved["first_prediction_future_fill_%d" % int(fill)] = changed
    model.train()
    torch.manual_seed(11)
    model.zero_grad(set_to_none=True)
    prediction = model(training_inputs)
    if prediction.shape != training_targets.shape:
        raise ValueError("Training-batch output and target shapes differ")
    loss = functional.mse_loss(prediction, training_targets)
    if not torch.isfinite(loss):
        raise ValueError("Gradient-probe loss is nonfinite")
    loss.backward()
    parameters = []
    for name, value in model.named_parameters():
        if value.grad is not None and not torch.isfinite(value.grad).all():
            raise ValueError("Gradient probe produced nonfinite gradients: " + name)
        parameters.append({"name": name, "elements": value.numel(), "requires_grad": value.requires_grad,
            "received_gradient": value.grad is not None,
            "nonzero_gradient": value.grad is not None and bool(torch.count_nonzero(value.grad))})
    if any(value._version != versions[name] for name, value in model.named_parameters()):
        raise ValueError("A parameter changed during the zero-update probe")
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"evaluation_mode": True, "softmax_without_explicit_dimension": calls,
        "first_target_normalized_dependence_observed": any(row["changed_first_output_values"] for row in cases),
        "first_target_future_input_dependence_observed": any(row["changed_clipped_forecast_values"] for row in cases),
        "normalized_rtol": 1e-5, "normalized_atol": 1e-6,
        "counterfactuals": cases, "gradient_probe_loss": float(loss.detach()), "optimizer_steps": 0,
        "parameter_versions_unchanged": True, "parameter_contract": parameters,
        "declared_trainable_elements": sum(row["elements"] for row in parameters if row["requires_grad"]),
        "elements_receiving_gradients": sum(row["elements"] for row in parameters if row["received_gradient"])}, saved


def worker(project, code, report, dataset):
    report.mkdir(parents=True, exist_ok=True)
    try:
        import torch, transformers
        if (torch.__version__, transformers.__version__, np.__version__) != ("2.0.0+cpu", "4.35.0", "1.24.0"):
            raise ValueError("Pinned CPU reference runtime is required")
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
        torch.set_num_interop_threads(1)
        torch.manual_seed(11)
        profile = PROFILES[dataset]
        lock = json.loads((code / "config/sources.lock.json").read_text())
        repo = prepare_reference(project / "work", lock, audit_only=True)
        assets = {}
        for key in ("stage2", "checkpoint"):
            name = profile[key]
            expected = next(row for row in lock["lfs_files"] if row["path"] == name)
            assets[name] = verify(repo / name, expected)
            if not assets[name]:
                raise ValueError("Pinned probe input fails verification: " + name)
        base = project / "work/models/gpt2" / lock["gpt2"]["revision"]
        for item in lock["gpt2"]["files"]:
            if not verify(base / item["path"], item):
                raise ValueError("GPT-2 base file fails verification: " + item["path"])
        os.environ.update(TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1")
        original = module_from(repo / "model/LLM_UTP.py", "_causality_author_model")
        loader = module_from(repo / "data/unified_dataset.py", "_causality_author_loader")
        values = np.load(repo / profile["stage2"], mmap_mode="r", allow_pickle=False)
        if list(values.shape) != profile["shape"]:
            raise ValueError("Probe array dimensions differ from the pinned profile")
        registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")[dataset]
        cfg = registry["finetune"]
        test = loader.UnifiedTrafficDataset(dataset, "predict", "test", 15, cfg["tg_in_one_day"], 12, 1, values, cfg["split_config"], 16)
        x = test[0][0].unsqueeze(0).to(torch.float32)
        # Build a training-route batch using exactly the released normalization
        # and window expressions, without materializing the whole training set.
        training = derive_splits((repo / "data/unified_dataset.py").read_text(), registry, values, "pretrain")["train"]
        start = training["start_target_index"]
        if training["scored_windows"] < 16:
            raise ValueError("Training route has no complete probe batch")
        normalized = test.norm_inflow
        train_x = torch.from_numpy(np.array([normalized[:, index - 12:index, :] for index in range(start, start + 16)])).unsqueeze(0).to(torch.float32)
        train_y = torch.from_numpy(np.array([normalized[:, index:index + 1, 0] for index in range(start, start + 16)])).to(torch.float32)
        model = original.TrafficPredictionModel(profile["shape"][0], 12, 64, [1, 2, 4], 12, 0, str(base), 768)
        state = torch.load(repo / profile["checkpoint"], map_location="cpu", weights_only=True)
        strict = strict_diagnostics(model, state)
        atomic_json(report / "strict_load.json", strict)
        if not strict["compatible"]:
            raise ValueError("Strict checkpoint compatibility failed")
        model.load_state_dict(state, strict=True)
        del state
        result, arrays = probe(model, x, train_x, train_y, cfg["max_inflow"])
        result.update(execution_status="COMPLETED_WITH_REVIEW", dataset=dataset, strict_load_performed=True,
            test_first_target_index=test.start_index, training_probe_start_target_index=start,
            training_probe_input_stage="stage2", training_probe_split_route="pretrain", training_performed=False,
            assets=assets, reference_commit=lock["reference_commit"],
            interpretation="Functional input-dependence and gradient diagnostics; no forecast-accuracy or clean-protocol claim")
        atomic_json(report / "causality_result.json", result)
        atomic_npz(report / "probe_predictions.npz", **arrays)
        return 3
    except Exception as exc:
        atomic_json(report / "causality_result.json", {"execution_status": "FAILED", "dataset": dataset, "error": str(exc)})
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        return 2


def run(project, code, report):
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    workers = min(2, max(1, cpus // 2))
    threads = max(1, cpus // workers)
    report.mkdir(parents=True, exist_ok=True)
    logs = Path(os.environ.get("TRAFFIC_WORKER_LOG_DIR", str(report / "worker_logs")))
    logs.resolve().relative_to(project)
    logs.mkdir(parents=True, exist_ok=True)
    progress = {"completed_datasets": 0, "failed_datasets": 0, "total_datasets": 8, "phase": "RUNNING",
        "datasets": {key: {"state": "PENDING"} for key in PUBLIC_ORDER}}
    mutex = threading.Lock()
    def update(key, **fields):
        with mutex:
            progress["datasets"][key].update(fields)
            progress["completed_datasets"] = sum(row["state"] == "COMPLETED_WITH_REVIEW" for row in progress["datasets"].values())
            progress["failed_datasets"] = sum(row["state"] == "FAILED" for row in progress["datasets"].values())
            atomic_json(report / "causality_progress.json", progress)
            print("CAUSALITY completed=%d/8 failed=%d latest=%s" % (progress["completed_datasets"], progress["failed_datasets"], key), flush=True)
    def evaluate(key):
        destination = report / "datasets" / PROFILES[key]["slug"]
        destination.mkdir(parents=True, exist_ok=True)
        out, err = logs / (key + ".out"), logs / (key + ".err")
        update(key, state="RUNNING", report=str(destination), stderr=str(err))
        try:
            with out.open("a") as output, err.open("a") as error:
                child = subprocess.run([sys.executable, "-u", str(code / "scripts/causality_audit.py"),
                    "--project", str(project), "--code", str(code), "--report", str(destination), "--dataset", key],
                    env={**os.environ, "SLURM_CPUS_PER_TASK": str(threads), "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)},
                    stdout=output, stderr=error)
            result = json.loads((destination / "causality_result.json").read_text())
            if (child.returncode != 3 or result.get("optimizer_steps") != 0 or not result.get("parameter_versions_unchanged")
                or result.get("strict_load_performed") is not True or result.get("dataset") != key):
                raise ValueError(result.get("error", "Probe did not satisfy its zero-update contract"))
            update(key, state="COMPLETED_WITH_REVIEW")
            return result
        except Exception as exc:
            update(key, state="FAILED", error=str(exc))
            return {"dataset": key, "execution_status": "FAILED", "error": str(exc)}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, key): key for key in PUBLIC_ORDER}
        results = {}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = {"dataset": key, "execution_status": "FAILED", "error": str(exc)}
                update(key, state="FAILED", error=str(exc))
    failed = bool(progress["failed_datasets"])
    progress["phase"] = "FAILED" if failed else "COMPLETED_WITH_REVIEW"
    atomic_json(report / "causality_progress.json", progress)
    summary = {"execution_status": progress["phase"], "scientific_status": "REPRODUCTION_NOT_ESTABLISHED",
        "datasets": results, "training_experiments_completed": 0, "optimizer_steps": 0,
        "archive_reader_candidates": {name: shutil.which(name) for name in ("unrar", "unar", "7zz", "7z", "bsdtar")},
        "limitations": ["Synthetic future-input counterfactuals applied to real checkpoints; this is not an accuracy experiment",
            "No dependence observed in one batch would not establish causality for all inputs",
            "Supplied second-channel preprocessing and full-array scaling remain separately unresolved",
            "Constructor gradient policy is measured, not assumed to reproduce the paper's stage-specific policy",
            "Time ordering is relative to released row indices; absolute timestamps remain unverified"]}
    atomic_json(report / "causality_summary.json", summary)
    lines = ["# Released checkpoint causality and gradient audit", "", "Execution: " + progress["phase"], "",
        "All original weights preserved. Optimizer steps: 0.", "",
        "| Dataset | Future dependence observed | Declared trainable elements | Elements receiving gradients |",
        "|---|---|---:|---:|"]
    for key in PUBLIC_ORDER:
        row = results[key]
        if row["execution_status"] == "FAILED":
            lines.append("| %s | FAILED | — | — |" % PROFILES[key]["title"])
        else:
            lines.append("| %s | %s | %d | %d |" % (PROFILES[key]["title"], row["first_target_future_input_dependence_observed"],
                row["declared_trainable_elements"], row["elements_receiving_gradients"]))
    for key, row in results.items():
        if row["execution_status"] == "FAILED":
            lines += ["", "FAILED " + key + ": " + row["error"]]
    lines += ["", "Archive reader candidates (capability unverified): " + json.dumps(summary["archive_reader_candidates"])]
    lines += ["", "## Scope", ""] + ["- " + item for item in summary["limitations"]]
    (report / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 2 if failed else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("project", "code", "report"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--dataset", choices=list(PROFILES))
    args = parser.parse_args()
    project, code, report = args.project.resolve(), args.code.resolve(), args.report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    try:
        sys.exit(worker(project, code, report, args.dataset) if args.dataset else run(project, code, report))
    except Exception as exc:
        report.mkdir(parents=True, exist_ok=True)
        atomic_json(report / "causality_summary.json", {"execution_status": "FAILED", "error": str(exc), "datasets": {}, "optimizer_steps": 0})
        (report / "SUMMARY.md").write_text("# Causality audit failed\n\n" + str(exc) + "\n")
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        sys.exit(2)
