"""Locked causal input/scaler/fusion rules and fail-closed execution gates."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from acquire import atomic_json
from reference_contract import atomic_npz, file_sha, inverse_minmax


def train_scaler(raw, splits):
    if raw.ndim != 2 or raw.shape[0] < 2:
        raise ValueError("Expected nonnegative flow as node-by-time observations")
    for name in ("train", "val", "test"):
        row = splits[name]
        if row["inner_batch_size"] != 16 or row["lookback_steps"] != 12 or row["horizon_steps"] != 1 or row["inner_batches"] < 1:
            raise ValueError("Causal baseline requires the locked 12/1/16 window contract")
        if not 12 <= row["start_target_index"] < row["scored_end_target_index_exclusive"] <= raw.shape[1]:
            raise ValueError("Invalid scored target boundaries")
        if row["scored_end_target_index_exclusive"] - row["start_target_index"] != 16 * row["inner_batches"]:
            raise ValueError("Scored target coverage differs from the fixed batch count")
    if splits["train"]["scored_end_target_index_exclusive"] > splits["val"]["start_target_index"] or splits["val"]["scored_end_target_index_exclusive"] > splits["test"]["start_target_index"]:
        raise ValueError("Training, validation and test targets must be chronological and disjoint")
    start = splits["train"]["start_target_index"] - 12
    end = splits["train"]["scored_end_target_index_exclusive"]
    support = raw[:, start:end]
    if not np.isfinite(support).all() or np.any(support < 0):
        raise ValueError("Training raw flows must be finite and nonnegative")
    lower, upper = float(support.min()), float(support.max())
    if upper <= lower:
        raise ValueError("Degenerate training-only scaler")
    return {"minimum": lower, "maximum": upper, "fit_start_index": start, "fit_end_index_exclusive": end,
        "fit_support": "union of scored training input rows and target rows; global over nodes",
        "heldout_rows_used_for_fit": 0, "normalized_values_clipped": False}


def features(history, scaler):
    """Only the supplied history is visible; prefix means use no extra context."""
    history = np.asarray(history, dtype="float64")
    if history.ndim != 3 or history.shape[-1] != 12 or not np.isfinite(history).all():
        raise ValueError("Expected finite batch-by-node-by-12 histories")
    trend = np.cumsum(history, axis=-1) / np.arange(1, 13, dtype="float64")
    pair = np.stack((history, trend), axis=-1)
    normalized = ((pair - scaler["minimum"]) / (scaler["maximum"] - scaler["minimum"])).astype("float32")
    if not np.isfinite(normalized).all():
        raise ValueError("Causal features are nonfinite")
    return normalized


class CausalBatches:
    def __init__(self, raw, split, scaler):
        self.raw, self.split, self.scaler = raw, split, scaler

    def __len__(self):
        return self.split["inner_batches"]

    def arrays(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = self.split["start_target_index"] + index * 16
        targets = np.arange(start, start + 16, dtype="int64")
        history = np.array([self.raw[:, i - 12:i] for i in targets], dtype="float64")
        truth = np.array([self.raw[:, i:i + 1] for i in targets], dtype="float64")
        y = ((truth - self.scaler["minimum"]) / (self.scaler["maximum"] - self.scaler["minimum"])).astype("float32")
        return features(history, self.scaler), y, targets

    def get(self, index, device):
        import torch
        x, y, _ = self.arrays(index)
        return torch.from_numpy(x).unsqueeze(0).to(device), torch.from_numpy(y).to(device)


def data_gates(raw, splits, scaler):
    """Mutation checks use no accuracy scores and never fit on held-out rows."""
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Raw source flow contains negative/nonfinite observations")
    target = splits["val"]["start_target_index"]
    mutated = raw.copy()
    # Alter all as-yet unobserved rows at the first validation issue time.
    mutated[:, target:] = mutated[:, target:] + (scaler["maximum"] - scaler["minimum"] + 1) * 100
    observed_scaler = train_scaler(mutated, splits)
    original_x = CausalBatches(raw, splits["val"], scaler).arrays(0)[0][:1]
    mutated_x = CausalBatches(mutated, splits["val"], observed_scaler).arrays(0)[0][:1]
    if observed_scaler != scaler or not np.array_equal(original_x, mutated_x):
        raise ValueError("Future raw observations changed training scaler or first validation input")
    return {"passed": True, "future_mutation_start_target": target,
        "training_scaler_unchanged_by_future_mutation": True, "first_validation_features_unchanged": True,
        "target_sets_disjoint": True, "source_second_channel_used": False,
        "raw_lookback_steps": 12, "effective_maximum_context_steps": 12,
        "evaluation_mode": "one-step rolling observed-history forecasts; no online parameter updates",
        "time_coordinates": "ordered row indices; absolute timestamps and service-gap semantics unverified"}


def patched_forward(source):
    """Minimal audited AST edit: only two implicit fusion softmax dimensions."""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrendEnhancementModule")
    fn = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward"))
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "F" and node.func.attr == "softmax"]
    if len(calls) != 2 or {node.args[0].id for node in calls if len(node.args) == 1 and isinstance(node.args[0], ast.Name)} != {"x1", "x2"} or any(node.keywords for node in calls):
        raise ValueError("Pinned fusion softmax source differs from the declared two-call patch")
    for node in calls:
        node.keywords.append(ast.keyword(arg="dim", value=ast.Constant(value=-1)))
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    rendered = ast.unparse(module) + "\n"
    return module, {"changed_calls": 2, "softmax_dimension": -1, "axis_meaning": "within-window features, not forecast batch",
        "patched_forward_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "source": rendered}


def install_fusion(model, original_module, source):
    syntax, description = patched_forward(source)
    namespace = dict(vars(original_module))
    exec(compile(syntax, "causal_reference_fusion", "exec"), namespace)
    original_keys = tuple(model.state_dict())
    original_parameters = [(name, value.requires_grad) for name, value in model.named_parameters()]
    cls = type("CausalTrendEnhancementModule", (original_module.TrendEnhancementModule,), {"forward": namespace["forward"]})
    model.trend_enhancement_module = cls()
    if tuple(model.state_dict()) != original_keys or [(name, value.requires_grad) for name, value in model.named_parameters()] != original_parameters:
        raise ValueError("Causal fusion adaptation changed model parameters or freeze policy")
    return description


def model_gates(model, inputs, atol=1e-6, rtol=1e-5):
    """Test deterministic evaluation invariance using training inputs only."""
    import torch
    if inputs.shape[0:2] != (1, 16):
        raise ValueError("Causality gate expects one outer batch of 16 origins")
    was_training = model.training
    rows = []
    model.eval()
    try:
        with torch.no_grad():
            baseline = model(inputs).detach()
            # CUDA GRU layout initialization may happen on the first forward.
            # Compare versions across the subsequent controlled forwards.
            versions = [parameter._version for parameter in model.parameters()]
            variants = []
            for fill in (0., 1.):
                changed = inputs.clone()
                changed[:, 1:] = fill
                variants.append(("companion_fill_" + str(fill), baseline[:1], model(changed)[:1]))
            variants += [("single_origin", baseline[:1], model(inputs[:, :1])),
                         ("batch_permutation", baseline, model(inputs.flip(1)).flip(0))]
            for name, expected, observed in variants:
                if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():
                    raise ValueError("Nonfinite causal-probe prediction")
                maximum = float((expected - observed).abs().max())
                passed = torch.allclose(expected, observed, atol=atol, rtol=rtol)
                rows.append({"case": name, "max_absolute_normalized_difference": maximum, "passed": passed})
                if not passed:
                    raise ValueError("Cross-window independence check failed: " + name + " max difference=" + str(maximum))
        if [parameter._version for parameter in model.parameters()] != versions:
            raise ValueError("Causal probe mutated model parameters")
    finally:
        model.train(was_training)
    return {"passed": True, "mode": "eval", "cases": rows, "atol": atol, "rtol": rtol,
        "probe_input_scope": "first training minibatch; no test accuracy used", "optimizer_updates": 0,
        "parameter_versions_unchanged_during_controls": True,
        "limitation": "Finite functional probes plus audited source; not a proof for every possible input/device"}


def save_forecast(path, targets, truth, normalized, scaler):
    atomic_npz(path, target_indices=np.asarray(targets, dtype="int64"), truth=np.asarray(truth, dtype="float32"),
        normalized_predictions=np.asarray(normalized, dtype="float32"),
        predictions=inverse_minmax(normalized, scaler["minimum"], scaler["maximum"]))
    atomic_json(path.with_suffix(".json"), {"sha256": file_sha(path), "scaler": scaler, "schema": "CAUSAL_FORECAST_V1"})


def load_forecast(path, targets, truth, scaler):
    if not path.exists() or not path.with_suffix(".json").exists():
        return None
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("scaler") != scaler or meta.get("schema") != "CAUSAL_FORECAST_V1" or file_sha(path) != meta["sha256"]:
        raise ValueError("Causal forecast shard hash/scaler mismatch")
    with np.load(path, allow_pickle=False) as stream:
        arrays = {name: stream[name] for name in stream.files}
    if not np.array_equal(arrays["target_indices"], targets) or not np.array_equal(arrays["truth"], np.asarray(truth, dtype="float32")):
        raise ValueError("Causal forecast shard truth/origins mismatch")
    if arrays["normalized_predictions"].shape != np.asarray(truth).shape or not np.array_equal(arrays["predictions"], inverse_minmax(arrays["normalized_predictions"], scaler["minimum"], scaler["maximum"])):
        raise ValueError("Causal forecast prediction shape/inverse mismatch")
    return arrays
