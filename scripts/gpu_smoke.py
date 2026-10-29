#!/usr/bin/env python3
"""Two independent training-batch smoke fits; not a forecasting benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import traceback

import numpy as np

from acquire import atomic_json, prepare_reference, verify
from pems08_audit import derive_splits, literal_assignment
from pems08_infer import module_from
from reference_contract import file_sha
from reference_datasets import PROFILES

DATASETS = ("pems08", "bj")
STEPS = 30


def fit_batch(model, inputs, targets, store, identity, update, steps=STEPS):
    """Resume our own full checkpoints, including optimizer and dropout RNG."""
    import torch
    store.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    pointer = store / "LATEST.json"
    start = 0
    initial = None
    losses = []
    if pointer.exists():
        recorded = json.loads(pointer.read_text())
        path = store / recorded["file"]
        path.resolve().relative_to(store.resolve())
        if file_sha(path) != recorded["sha256"]:
            raise ValueError("Smoke checkpoint digest differs; checkpoint preserved")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Smoke checkpoint belongs to a different experiment")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if inputs.is_cuda:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        start, initial, losses = state["step"], state["initial_eval_mse"], state["losses"]
        if not 0 <= start <= steps:
            raise ValueError("Checkpoint step is outside the locked smoke budget")
    model.eval()
    with torch.no_grad():
        current = float(torch.nn.functional.mse_loss(model(inputs), targets))
    if initial is None:
        initial = current
    if not np.isfinite(initial) or initial <= 1e-12:
        raise ValueError("Initial training-batch loss is nonfinite or too small for an informative smoke check")
    received = 0
    for step in range(start, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        if prediction.shape != targets.shape:
            raise ValueError("Smoke prediction and target shapes differ")
        loss = torch.nn.functional.mse_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite smoke loss")
        loss.backward()
        received = sum(value.numel() for value in model.parameters() if value.grad is not None)
        if received == 0 or any(not torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None):
            raise ValueError("Smoke gradients are absent or nonfinite")
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0 or step + 1 == steps:
            state = {"identity": identity, "step": step + 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if inputs.is_cuda else [],
                "initial_eval_mse": initial, "losses": losses}
            checkpoint = store / ("step-%04d.pt" % (step + 1))
            temporary = checkpoint.with_suffix(".pt.part")
            torch.save(state, temporary)
            temporary.replace(checkpoint)
            atomic_json(pointer, {"file": checkpoint.name, "sha256": file_sha(checkpoint), "step": step + 1})
        update(completed_steps=step + 1, total_steps=steps, latest_train_mse=losses[-1], resumed_from_step=start)
    model.eval()
    with torch.no_grad():
        final = float(torch.nn.functional.mse_loss(model(inputs), targets))
    if not np.isfinite(final):
        raise ValueError("Final smoke loss is nonfinite")
    result = {"optimizer_steps": steps, "new_optimizer_steps": steps - start, "resumed_from_step": start,
        "initial_eval_training_batch_mse": initial, "final_eval_training_batch_mse": final,
        "relative_training_batch_loss_reduction": (initial - final) / initial,
        "elements_receiving_gradients_last_step": received if steps > start else None,
        "overfit_smoke_pass": final <= initial * 0.9,
        "scope": "same training batch before/after fitting; no validation/test performance claim"}
    atomic_json(store / "smoke_result.json", result)
    return result


def run(project, code, report):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    report.mkdir(parents=True, exist_ok=True)
    progress = {"completed_datasets": 0, "failed_datasets": 0, "total_datasets": 2,
        "phase": "gpu_preflight", "datasets": {key: {"state": "PENDING"} for key in DATASETS}}
    def update(key=None, **values):
        if key:
            progress["datasets"][key].update(values)
        progress["completed_datasets"] = sum(row["state"] == "COMPLETED_WITH_REVIEW" for row in progress["datasets"].values())
        progress["failed_datasets"] = sum(row["state"] == "FAILED" for row in progress["datasets"].values())
        atomic_json(report / "gpu_smoke_progress.json", progress)
        print("GPU_SMOKE", json.dumps(progress), flush=True)
    try:
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15)
            (report / "nvidia-smi.txt").write_text(smi.stdout + smi.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            (report / "nvidia-smi.txt").write_text(str(exc))
        import torch, transformers
        if (str(torch.__version__), transformers.__version__, np.__version__) != ("2.0.0+cu118", "4.35.0", "1.24.0"):
            raise ValueError("Pinned CUDA reference environment is required")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expected exactly one Slurm-visible CUDA GPU; no CPU fallback")
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "20")))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Original GRU CUDA determinism is checked empirically by smoke outputs;
        # do not claim bitwise equivalence across devices or CUDA versions.
        device = torch.device("cuda:0")
        hardware = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "visible_gpus": torch.cuda.device_count(),
            "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "20"))}
        atomic_json(report / "gpu_runtime.json", hardware)
        lock = json.loads((code / "config/sources.lock.json").read_text())
        repo = prepare_reference(project / "work", lock, audit_only=True)
        base = project / "work/models/gpt2" / lock["gpt2"]["revision"]
        for item in lock["gpt2"]["files"]:
            if not verify(base / item["path"], item):
                raise ValueError("Pinned base file failed verification: " + item["path"])
        os.environ.update(TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1")
        original = module_from(repo / "model/LLM_UTP.py", "_smoke_author_model")
        registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")
        results = {}
        logs = Path(os.environ.get("TRAFFIC_WORKER_LOG_DIR", str(report / "worker_logs")))
        logs.resolve().relative_to(project)
        logs.mkdir(parents=True, exist_ok=True)
        for key in DATASETS:
            update(key, state="RUNNING")
            try:
                torch.manual_seed(11)
                torch.cuda.manual_seed_all(11)
                profile = PROFILES[key]
                expected = next(row for row in lock["lfs_files"] if row["path"] == profile["stage1"])
                actual = verify(repo / profile["stage1"], expected)
                if not actual:
                    raise ValueError("Stage-one source array failed verification")
                values = np.load(repo / profile["stage1"], mmap_mode="r", allow_pickle=False)
                if list(values.shape) != profile["shape"]:
                    raise ValueError("Unexpected stage-one source dimensions")
                split = derive_splits((repo / "data/unified_dataset.py").read_text(), registry[key], values, "pretrain")["train"]
                start = split["start_target_index"]
                lower, upper = float(values.min()), float(values.max())
                if upper <= lower or split["scored_windows"] < 16:
                    raise ValueError("Invalid scaler or insufficient training windows")
                # Preserve the author's full-array normalization for this
                # faithful-behavior engineering smoke; not a causal baseline.
                normalized = (values - lower) / (upper - lower)
                x = torch.from_numpy(np.array([normalized[:, i - 12:i, :] for i in range(start, start + 16)])).unsqueeze(0).to(device=device, dtype=torch.float32)
                y = torch.from_numpy(np.array([normalized[:, i:i + 1, 0] for i in range(start, start + 16)])).to(device=device, dtype=torch.float32)
                identity = {"dataset": key, "seed": 11, "source_sha256": actual["sha256"], "reference_commit": lock["reference_commit"],
                    "gpt2_revision": lock["gpt2"]["revision"], "smoke_source": file_sha(Path(__file__)), "runtime": hardware,
                    "freeze_sha256": file_sha(report / "reference_environment_freeze.txt"), "steps": STEPS,
                    "learning_rate": 0.0001, "inner_batch": 16, "outer_batch": 1, "training_start_target": start,
                    "stage": "pretrain", "array": "stage1", "normalization": "released full-array minmax", "min": lower, "max": upper,
                    "initialization": "official pinned GPT-2 base plus seeded fresh author encoder/head; no target checkpoint loaded"}
                run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                store = project / "outputs/gpu_smoke" / profile["slug"] / run_hash
                store.resolve().relative_to(project)
                store.mkdir(parents=True, exist_ok=True)
                atomic_json(store / "identity.json", identity)
                model = original.TrafficPredictionModel(profile["shape"][0], 12, 64, [1, 2, 4], 12, 0, str(base), 768).to(device)
                result = fit_batch(model, x, y, store, identity, lambda **kw: update(key, **kw))
                result.update(dataset=key, output_store=str(store), forecast_evaluation_performed=False,
                    execution_status="COMPLETED_WITH_REVIEW" if result["overfit_smoke_pass"] else "FAILED")
                if not result["overfit_smoke_pass"]:
                    result["error"] = "Training-batch evaluation MSE did not decrease by the predeclared 10% in 30 updates"
                results[key] = result
                (logs / (key + ".json")).write_text(json.dumps(result, indent=2) + "\n")
                update(key, state=result["execution_status"], result=result, completed_steps=result["optimizer_steps"],
                    new_optimizer_steps=result["new_optimizer_steps"], resumed_from_step=result["resumed_from_step"])
                del model, x, y
                torch.cuda.empty_cache()
            except Exception as exc:
                results[key] = {"dataset": key, "execution_status": "FAILED", "error": str(exc)}
                (logs / (key + ".err")).write_text(traceback.format_exc())
                update(key, state="FAILED", error=str(exc))
        progress["phase"] = "FAILED" if progress["failed_datasets"] else "COMPLETED_WITH_REVIEW"
        update()
        summary = {"execution_status": progress["phase"], "scientific_status": "REPRODUCTION_NOT_ESTABLISHED",
            "datasets": results, "completed_training_smoke_fits": progress["completed_datasets"], "benchmark_experiments_completed": 0,
            "limitations": ["One fixed training minibatch per domain, not an epoch or a test evaluation",
                "Original cross-window softmax and full-array scaling are retained and are not a causal baseline",
                "Fresh single-domain smoke initialization does not certify original S1/S2/S3 training settings",
                "CUDA environment, driver and numeric behavior are recorded; CPU synthetic tests do not verify this GPU execution"]}
        atomic_json(report / "gpu_smoke_summary.json", summary)
        lines = ["# Independent GPU training-batch smoke", "", "Execution: " + progress["phase"], "",
            "Benchmark experiments: 0. One fixed training batch, 30 Adam updates per dataset.", ""]
        for key, result in results.items():
            lines += [key + ": " + json.dumps(result), ""]
        lines += ["## Scope", ""] + ["- " + value for value in summary["limitations"]]
        (report / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        return 2 if progress["failed_datasets"] else 3
    except Exception as exc:
        atomic_json(report / "gpu_smoke_summary.json", {"execution_status": "FAILED", "error": str(exc), "datasets": {}})
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        (report / "SUMMARY.md").write_text("# GPU smoke failed\n\n" + str(exc) + "\n")
        print("GPU_SMOKE_FAILED:", str(exc), flush=True)
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.project, args.code, args.report))
