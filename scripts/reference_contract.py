"""Numerical checks used independently of the author's metric implementation."""
import hashlib
import json
from pathlib import Path

import numpy as np


ANCHORS = {"S1": {"MAE": 11.87, "RMSE": 18.53, "WMAPE_percent": 5.16},
           "S2": {"MAE": 14.78, "RMSE": 22.96, "WMAPE_percent": 6.43},
           "S3": {"MAE": 13.25, "RMSE": 20.39, "WMAPE_percent": 5.76}}


def independent_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype="float64"), np.asarray(prediction, dtype="float64")
    if truth.shape != prediction.shape or truth.ndim != 2 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be equally shaped finite time-by-node arrays")
    absolute = np.abs(truth - prediction)
    denominator = truth.sum()
    if denominator <= 0:
        raise ValueError("Reference WMAPE denominator is nonpositive")
    return {"MAE": float(absolute.mean()), "RMSE": float(np.sqrt(np.square(truth - prediction).mean())),
            "WMAPE_percent": float(100 * absolute[truth > 0].sum() / denominator)}


def inverse_author(normalized, scale):
    result = np.asarray(normalized, dtype="float32") * np.float32(scale)
    return np.maximum(result, np.float32(0))


def anchor_comparison(measured):
    return {strategy: {"published": values,
        "absolute_differences": {name: float(abs(measured[name] - value)) for name, value in values.items()},
        "same_two_decimal_rounding": {name: format(measured[name], ".2f") == format(value, ".2f") for name, value in values.items()},
        "interpretation": "Diagnostic only; checkpoint training strategy and published test-origin provenance remain unresolved"}
        for strategy, values in ANCHORS.items()}


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_batch(directory, index, targets, truth, normalized, scale):
    path = directory / ("batch-%04d.npz" % index)
    atomic_npz(path, target_indices=np.asarray(targets, dtype="int64"), truth=np.asarray(truth, dtype="float32"),
               normalized_predictions=np.asarray(normalized, dtype="float32"), predictions=inverse_author(normalized, scale))
    metadata = {"sha256": file_sha(path), "batch_index": index}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def load_batch(directory, index, targets, truth, scale):
    path = directory / ("batch-%04d.npz" % index)
    record = path.with_suffix(".json")
    if not path.exists() or not record.exists():
        return None
    if json.loads(record.read_text())["sha256"] != file_sha(path):
        raise ValueError("Saved prediction shard fails integrity verification: " + str(path))
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    expected_truth = np.asarray(truth, dtype="float32")
    if not np.array_equal(arrays["target_indices"], targets) or not np.array_equal(arrays["truth"], expected_truth):
        raise ValueError("Saved prediction shard targets differ from this evaluation contract")
    if arrays["normalized_predictions"].shape != expected_truth.shape or not np.isfinite(arrays["normalized_predictions"]).all():
        raise ValueError("Saved predictions are invalid")
    if not np.array_equal(arrays["predictions"], inverse_author(arrays["normalized_predictions"], scale)):
        raise ValueError("Saved inverse-scaled predictions differ from the author transformation")
    return arrays
