#!/usr/bin/env python3
"""Resumable single-domain reference training, one dataset/seed per array task."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np

from acquire import atomic_json, prepare_reference, verify
from pems08_audit import derive_splits, literal_assignment, write_origins
from pems08_infer import module_from
from reference_contract import (anchor_comparison, file_sha, independent_metrics,
                                load_batch, save_batch)
from reference_datasets import PROFILES


def task_plan(config):
    return [{"dataset": dataset, "seed": seed} for dataset in config["datasets"] for seed in config["seeds"]]


class WindowBatches:
    """Lazy equivalent of released contiguous inner batches; no window reordering."""
    def __init__(self, normalized, raw, split, inner=16):
        self.normalized, self.raw, self.split, self.inner = normalized, raw, split, inner
        if split["inner_batch_size"] != inner or split["horizon_steps"] != 1 or split["lookback_steps"] != 12:
            raise ValueError("Released batch contract changed")
        if len(self) < 1:
            raise ValueError("A required train/validation/test split has no full batches")

    def __len__(self):
        return self.split["inner_batches"]

    def get(self, index, device):
        import torch
        start = self.split["start_target_index"] + index * self.inner
        origins = np.arange(start, start + self.inner)
        x = np.array([self.normalized[:, i - 12:i, :] for i in origins])
        y = np.array([self.normalized[:, i:i + 1, 0] for i in origins])
        return (torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32),
                torch.from_numpy(y).to(device=device, dtype=torch.float32))


def load_checkpoint(store, identity):
    import torch
    pointer = store / "LATEST.json"
    if not pointer.exists():
        return None
    record = json.loads(pointer.read_text())
    path = store / record["file"]
    path.resolve().relative_to(store.resolve())
    if file_sha(path) != record["sha256"]:
        raise ValueError("Training checkpoint digest mismatch; files preserved")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Checkpoint identity differs; refusing cross-experiment resume")
    return state


def save_checkpoint(store, identity, model, optimizer, state):
    import torch
    previous = json.loads((store / "LATEST.json").read_text()) if (store / "LATEST.json").exists() else None
    name = "state-%09d-%04d-%05d.pt" % (state["updates"], state["epoch"], state["cursor"])
    checkpoint = store / name
    payload = {**state, "identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}
    temporary = checkpoint.with_suffix(".pt.part")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    pointer = {"file": name, "sha256": file_sha(checkpoint), "epoch": state["epoch"], "cursor": state["cursor"],
        "updates": state["updates"], "best": state["best"]}
    atomic_json(store / "LATEST.json", pointer)
    # Keep two resumable states and the best weights each references. Never
    # remove checkpoints outside this experiment's own narrowly named files.
    keep_states = {name} | ({previous["file"]} if previous else set())
    keep_best = {row["best"]["file"] for row in (pointer, previous) if row and row.get("best")}
    for candidate in store.glob("state-*.pt"):
        if candidate.name not in keep_states:
            candidate.unlink()
    for candidate in store.glob("best-*.pt"):
        if candidate.name not in keep_best:
            candidate.unlink()


def fit(model, train, validation, store, identity, config, update, stop_after_updates=None):
    """Resume at a saved batch boundary, preserving Adam/dropout state."""
    import torch
    store.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    state = {"epoch": 0, "cursor": 0, "updates": 0, "epoch_loss_sum": 0.0,
        "best": None, "bad_epochs": 0, "history": [], "training_complete": False}
    saved = load_checkpoint(store, identity)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if next(model.parameters()).is_cuda:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        state.update({key: saved[key] for key in state})
    resumed_updates = state["updates"]
    device = next(model.parameters()).device
    started = time.monotonic()
    last_checkpoint_time = started
    def report(**fields):
        update(epoch=state["epoch"] + (0 if state["training_complete"] else 1), completed_epochs=state["epoch"],
            max_epochs=config["max_epochs"], batch=state["cursor"], train_batches=len(train),
            optimizer_steps=state["updates"], resumed_updates=resumed_updates,
            new_optimizer_steps=state["updates"] - resumed_updates, best=state["best"],
            bad_epochs=state["bad_epochs"], elapsed_this_attempt_seconds=time.monotonic() - started, **fields)
    report(phase="RESUMED" if saved else "TRAINING")
    while state["epoch"] < config["max_epochs"] and not state["training_complete"]:
        epoch_start = time.monotonic()
        # Like the released DataLoader, shuffle whole contiguous inner batches.
        # Epoch-specific seeds make order explicit and resumable; original seed
        # and sampler stream are unknown, so bitwise paper reproduction is not claimed.
        generator = torch.Generator().manual_seed(identity["seed"] + state["epoch"] * 1000003)
        order = torch.randperm(len(train), generator=generator).tolist()
        model.train()
        for cursor in range(state["cursor"], len(order)):
            x, y = train.get(order[cursor], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if prediction.shape != y.shape:
                raise ValueError("Training prediction shape differs from targets")
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError("Missing or nonfinite training gradients")
            optimizer.step()
            state["updates"] += 1
            state["cursor"] = cursor + 1
            state["epoch_loss_sum"] += float(loss.detach())
            periodic = state["updates"] % config["checkpoint_batches"] == 0 and time.monotonic() - last_checkpoint_time >= config.get("checkpoint_seconds", 0)
            if periodic or state["cursor"] == len(train) or state["updates"] == stop_after_updates:
                save_checkpoint(store, identity, model, optimizer, state)
                last_checkpoint_time = time.monotonic()
            if state["updates"] % config["progress_batches"] == 0 or state["cursor"] == len(train):
                report(phase="TRAINING", latest_train_batch_mse=float(loss.detach()),
                    running_train_mse=state["epoch_loss_sum"] / state["cursor"])
            if stop_after_updates is not None and state["updates"] >= stop_after_updates:
                report(phase="PAUSED_TEST_FIXTURE")
                return state
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for index in range(len(validation)):
                x, y = validation.get(index, device)
                prediction = model(x)
                if prediction.shape != y.shape:
                    raise ValueError("Validation prediction shape differs from targets")
                value = torch.nn.functional.mse_loss(prediction, y)
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite validation loss")
                validation_loss += float(value)
                if (index + 1) % config["progress_batches"] == 0:
                    report(phase="VALIDATING", validation_batch=index + 1, validation_batches=len(validation))
        val_mse = validation_loss / len(validation)
        epoch = state["epoch"] + 1
        # main_run.py calls EarlyStopping only when zero-based epoch > 0.
        # First epoch is logged but cannot select the final test checkpoint.
        if epoch >= 2:
            if state["best"] is None or val_mse <= state["best"]["validation_mse"]:
                best_path = store / ("best-%04d.pt" % epoch)
                temporary = best_path.with_suffix(".pt.part")
                torch.save(model.state_dict(), temporary)
                temporary.replace(best_path)
                state["best"] = {"file": best_path.name, "sha256": file_sha(best_path),
                    "epoch": epoch, "validation_mse": val_mse}
                state["bad_epochs"] = 0
            else:
                state["bad_epochs"] += 1
        row = {"epoch": epoch, "train_mse": state["epoch_loss_sum"] / len(train), "validation_mse": val_mse,
            "optimizer_steps": state["updates"], "seconds_this_epoch_attempt": time.monotonic() - epoch_start}
        state["history"].append(row)
        state.update(epoch=epoch, cursor=0, epoch_loss_sum=0.0,
            training_complete=epoch >= config["max_epochs"] or state["bad_epochs"] >= config["patience"])
        save_checkpoint(store, identity, model, optimizer, state)
        last_checkpoint_time = time.monotonic()
        atomic_json(store / "loss_history.json", state["history"])
        (store / "loss_history.csv").write_text("epoch,train_mse,validation_mse,optimizer_steps,seconds_this_epoch_attempt\n" +
            "".join(",".join(str(row[key]) for key in ("epoch", "train_mse", "validation_mse", "optimizer_steps", "seconds_this_epoch_attempt")) + "\n" for row in state["history"]))
        report(phase="TRAINING_FINISHED" if state["training_complete"] else "EPOCH_COMPLETE", latest_epoch=row)
        print("EPOCH", json.dumps(row), flush=True)
    if not state["best"]:
        raise ValueError("No validation-selected checkpoint; at least two epochs are required")
    return state


def run(project, code, report):
    import torch, transformers
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    report.mkdir(parents=True, exist_ok=True)
    config = json.loads((code / "config/reference_training.json").read_text())
    task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task = task_plan(config)[task_index]
    progress = {**task, "task_index": task_index, "phase": "PREFLIGHT", "optimizer_steps": 0}
    store = None
    def update(**fields):
        progress.update(fields)
        progress["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(report / "training_progress.json", progress)
        if store:
            atomic_json(store / "progress.json", progress)
        (report / "SUMMARY.md").write_text("# Single-domain reference training\n\n" +
            "Live status; no test metric until validation-selected training finishes.\n\n```json\n" + json.dumps(progress, indent=2) + "\n```\n")
        print("TRAIN_PROGRESS", json.dumps(progress), flush=True)
    try:
        update()
        if (str(torch.__version__), transformers.__version__, np.__version__) != ("2.0.0+cu118", "4.35.0", "1.24.0"):
            raise ValueError("Pinned CUDA reference runtime required")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expected exactly one visible CUDA GPU; no CPU fallback")
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "5")))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        torch.cuda.manual_seed_all(task["seed"])
        hardware = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        atomic_json(report / "gpu_runtime.json", hardware)
        lock = json.loads((code / "config/sources.lock.json").read_text())
        repo = prepare_reference(project / "work", lock, audit_only=True)
        profile = PROFILES[task["dataset"]]
        for name in (profile["stage1"], "utils/metrics.py"):
            expected = next(row for row in lock["lfs_files"] if row["path"] == name)
            if not verify(repo / name, expected):
                raise ValueError("Required released file hash failed: " + name)
        base = project / "work/models/gpt2" / lock["gpt2"]["revision"]
        for item in lock["gpt2"]["files"]:
            if not verify(base / item["path"], item):
                raise ValueError("Pinned GPT-2 file hash failed")
        values = np.load(repo / profile["stage1"], mmap_mode="r", allow_pickle=False)
        if list(values.shape) != profile["shape"] or not np.isfinite(values).all():
            raise ValueError("Invalid source array shape/values")
        registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")
        splits = derive_splits((repo / "data/unified_dataset.py").read_text(), registry[task["dataset"]], values, config["route"])
        lower, upper = float(values.min()), float(values.max())
        if upper <= lower:
            raise ValueError("Degenerate full-array scaler")
        normalized = (values - lower) / (upper - lower)
        scheduling = {"datasets", "seeds", "max_concurrent_tasks", "checkpoint_batches", "checkpoint_seconds", "progress_batches"}
        identity = {**task, "config": {key: value for key, value in config.items() if key not in scheduling},
            "source_sha256": file_sha(repo / profile["stage1"]),
            "reference_commit": lock["reference_commit"], "gpt2_revision": lock["gpt2"]["revision"],
            "trainer_sha256": file_sha(Path(__file__)), "splits": splits, "min": lower, "max": upper,
            "helper_sha256": {name: file_sha(code / "scripts" / name) for name in
                ("reference_contract.py", "pems08_audit.py", "pems08_infer.py", "reference_datasets.py")},
            "environment_sha256": file_sha(report / "reference_environment_freeze.txt"), "runtime": hardware,
            "initialization": "official pinned GPT-2 plus fresh seeded author encoder/head; no released target checkpoint"}
        run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        store = project / "outputs/reference_training" / profile["slug"] / ("seed-%s" % task["seed"]) / run_hash
        store.resolve().relative_to(project)
        store.mkdir(parents=True, exist_ok=True)
        with (store / "run.lock").open("a") as run_lock:
            try:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("This exact dataset/seed experiment is already running")
            atomic_json(store / "identity.json", identity)
            atomic_json(report / "training_contract.json", identity)
            update(output_store=str(store), train_batches=splits["train"]["inner_batches"],
                validation_batches=splits["val"]["inner_batches"], test_batches=splits["test"]["inner_batches"])
            for split, row in splits.items():
                write_origins(report / (split + "_origins.csv"), row)
            if (store / "training_result.json").exists() and (store / "COMPLETED.json").exists():
                completed = json.loads((store / "COMPLETED.json").read_text())
                if file_sha(store / "training_result.json") != completed["result_sha256"]:
                    raise ValueError("Completed result digest mismatch")
                result = json.loads((store / "training_result.json").read_text())
                if file_sha(store / result["best"]["file"]) != result["best"]["sha256"]:
                    raise ValueError("Completed experiment's selected model digest mismatch")
                atomic_json(report / "training_result.json", result)
                update(phase="COMPLETED_WITH_REVIEW", result=result, reused_completed_result=True,
                    optimizer_steps=result["optimizer_steps"], completed_epochs=result["epochs"], new_optimizer_steps=0)
                return 3
            if shutil.disk_usage(project).free < 3 * 2**30:
                raise ValueError("Less than 3 GiB free for this run's retained checkpoints; free project space before retrying")
            os.environ.update(TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1")
            original = module_from(repo / "model/LLM_UTP.py", "_train_author_model")
            model = original.TrafficPredictionModel(profile["shape"][0], 12, 64, [1, 2, 4], 12, 0, str(base), 768).cuda()
            batches = {key: WindowBatches(normalized, values, row) for key, row in splits.items()}
            state = fit(model, batches["train"], batches["val"], store, identity, config, update)
            best = state["best"]
            if file_sha(store / best["file"]) != best["sha256"]:
                raise ValueError("Best model digest mismatch")
            model.load_state_dict(torch.load(store / best["file"], map_location="cpu", weights_only=True), strict=True)
            model.eval()
            forecasts = store / "test_predictions"
            forecasts.mkdir(exist_ok=True)
            scale = registry[task["dataset"]]["finetune"]["max_inflow"]
            truth_list, prediction_list = [], []
            with torch.no_grad():
                for index in range(len(batches["test"])):
                    start = splits["test"]["start_target_index"] + index * 16
                    targets = np.arange(start, start + 16)
                    truth = np.array([values[:, i, 0] for i in targets], dtype="float32")
                    shard = load_batch(forecasts, index, targets, truth, scale)
                    if shard is None:
                        x, _ = batches["test"].get(index, "cuda:0")
                        prediction = model(x).squeeze(-1).cpu().numpy()
                        save_batch(forecasts, index, targets, truth, prediction, scale)
                        shard = load_batch(forecasts, index, targets, truth, scale)
                    truth_list.append(shard["truth"])
                    prediction_list.append(shard["predictions"])
                    if (index + 1) % 10 == 0 or index + 1 == len(batches["test"]):
                        update(phase="TESTING_SELECTED_CHECKPOINT", test_batches_completed=index + 1)
            truth, prediction = np.concatenate(truth_list), np.concatenate(prediction_list)
            metrics = module_from(repo / "utils/metrics.py", "_train_author_metrics")
            rmse, r2, mae, wmape = metrics.Metrics(truth, prediction).evaluate_performance()
            measured = {"MAE": float(mae), "RMSE": float(rmse), "WMAPE_percent": float(wmape) * 100, "R2": float(r2)}
            independent = independent_metrics(truth, prediction)
            if not all(np.isclose(measured[key], value, rtol=1e-5, atol=1e-6) for key, value in independent.items()):
                raise ValueError("Author and independent metrics disagree")
            result = {**task, "execution_status": "COMPLETED_WITH_REVIEW", "scientific_status": "REPRODUCTION_NOT_ESTABLISHED",
                "epochs": state["epoch"], "optimizer_steps": state["updates"], "best": best,
                "stop_reason": "patience" if state["bad_epochs"] >= config["patience"] else "max_epochs",
                "metrics_author": measured, "metrics_independent": independent, "output_store": str(store),
                "scored_origins": len(truth), "source_array": "stage1", "split_route": config["route"],
                "paper_comparisons_diagnostic_only": anchor_comparison(measured, profile["anchors"]),
                "limitations": ["Independent single-domain released-behavior experiment; exact published S3 settings unresolved",
                    "Original full-array scaling and cross-window dependence retained; not a causal baseline",
                    "Test cohort is the recorded stage1/pretrain route; not interchangeable with stage2/finetune results",
                    "Epoch-specific batch-shuffle seeds are explicit; original author seeds/order are unknown"]}
            atomic_json(store / "training_result.json", result)
            atomic_json(store / "COMPLETED.json", {"result_sha256": file_sha(store / "training_result.json")})
            atomic_json(report / "training_result.json", result)
            update(phase="COMPLETED_WITH_REVIEW", result=result)
            return 3
    except Exception as exc:
        update(phase="FAILED", error=str(exc))
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.project, args.code, args.report))
