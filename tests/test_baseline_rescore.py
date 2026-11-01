"""Synthetic physical-unit regression, complete CPU rescore and integrity checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_rescore as rescore
from acquire import atomic_json
from pems08_audit import derive_splits
from pems08_infer import module_from
from reference_contract import file_sha, inverse_author, inverse_minmax, save_batch
from test_pems08_audit import SOURCE

METRIC_SOURCE = '''
import numpy as np
class Metrics:
    def __init__(self, truth, prediction):
        self.truth, self.prediction = truth, prediction
    def evaluate_performance(self):
        error = self.truth - self.prediction
        total = np.sum(self.truth)
        weighted = []
        for i in range(len(self.truth)):
            for j in range(self.truth.shape[1]):
                if self.truth[i,j] > 0:
                    weighted.append(self.truth[i,j] / total * abs(error[i,j] / self.truth[i,j]))
        return float(np.sqrt(np.mean(error**2))), 0.0, float(np.mean(abs(error))), float(np.sum(weighted))
'''


class InverseScalingTests(unittest.TestCase):
    def test_negative_minimum_roundtrip_and_zero_minimum_legacy_control(self):
        truth = np.array([[0, 10, 100, 1000, 4744]], dtype="float64")
        minimum, maximum = -53.2963, 4744
        z = ((truth - minimum) / (maximum - minimum)).astype("float32")
        np.testing.assert_allclose(inverse_minmax(z, minimum, maximum), truth, atol=0.0005, rtol=1e-6)
        self.assertGreater(float(inverse_author(z, maximum)[0, 0]), 50)
        z = np.array([[-.02, 0, .01, .3, 1]], dtype="float32")
        self.assertTrue(np.array_equal(inverse_minmax(z, 0, 1147), inverse_author(z, 1147)))

    def test_positive_minimum_clipping_and_invalid_scalers(self):
        np.testing.assert_array_equal(inverse_minmax(np.array([[-1, 0, .5, 1]]), 10, 110), [[0, 10, 60, 110]])
        for lower, upper in ((1, 1), (2, 1), (float("nan"), 1), (0, float("inf"))):
            with self.assertRaises(ValueError):
                inverse_minmax([.5], lower, upper)
        with self.assertRaises(ValueError):
            inverse_minmax([float("nan")], 0, 1)


class CompleteRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name).resolve()
        self.code = self.project / "code"
        self.repo = self.project / "work/reference/synthetic"
        self.original = self.project / "work/code_versions/original"
        for path in (self.code / "scripts", self.code / "config", self.repo / "data", self.repo / "utils", self.original / "scripts"):
            path.mkdir(parents=True, exist_ok=True)
        # All source fixtures, arrays, weights and forecasts are synthetic.
        # The checkpoint bytes are hashed only, never unpickled.
        self.loader_source = SOURCE
        (self.repo / "data/unified_dataset.py").write_text(self.loader_source)
        metric_source = METRIC_SOURCE
        (self.repo / "utils/metrics.py").write_text(metric_source)
        self.metrics = module_from(self.repo / "utils/metrics.py", "_synthetic_rescore_metrics")
        for name in ("reference_train.py", "reference_contract.py"):
            (self.original / "scripts" / name).write_text("# synthetic original source " + name)
        (self.code / "scripts/reference_contract.py").write_bytes((ROOT / "scripts/reference_contract.py").read_bytes())
        common = {"mode": "ratio_tail", "train_ratio": .8, "val_ratio": .1, "test_ratio": .1, "train_subtract_pre_len": False}
        target = {"mode": "ratio_window", "train_start": "time_lag", "train_len_ratio": .1,
            "val_start_ratio": .1, "val_len_ratio": .1, "test_start_ratio": .9, "test_len_ratio": .1}
        self.profiles, registry, assets = {}, {}, []
        for key, minimum, maximum in (("pems08", 0., 1147.), ("bj", -53.2963, 4744.)):
            shape = [2, 320, 2]
            profile = {"shape": shape, "stage1": "data/" + key + ".npy", "slug": key, "title": key}
            self.profiles[key] = profile
            values = np.stack([np.tile(np.arange(320) % 100, (2, 1))] * 2, axis=-1).astype("float64")
            values[0, 0, 0], values[0, 1, 1] = maximum, minimum
            path = self.repo / profile["stage1"]
            np.save(path, values)
            assets.append({"path": profile["stage1"], "size": path.stat().st_size, "sha256": file_sha(path)})
            registry[key] = {"pretrain": {"split_config": common, "tg_in_one_day": 288},
                "finetune": {"split_config": target, "tg_in_one_day": 288, "max_inflow": int(maximum)}}
        (self.repo / "data/dataset_registry.py").write_text("DATASET_REGISTRY = " + repr(registry))
        metrics_path = self.repo / "utils/metrics.py"
        assets.append({"path": "utils/metrics.py", "size": metrics_path.stat().st_size, "sha256": file_sha(metrics_path)})
        lock = {"reference_commit": "synthetic-reference", "lfs_files": assets}
        atomic_json(self.code / "config/sources.lock.json", lock)
        tasks = [{"dataset": key, "seed": seed} for key in ("pems08", "bj") for seed in (11, 22, 33)]
        group = self.project / "results/slurm_jobs/70000"
        atomic_json(group / "submission.json", {"project": str(self.project), "job_id": "70000", "stage": "reference-train3", "array_tasks": tasks})
        for index, task in enumerate(tasks):
            key = task["dataset"]
            values = np.load(self.repo / self.profiles[key]["stage1"])
            lower, upper = float(values.min()), float(values.max())
            splits = derive_splits(self.loader_source, registry[key], values, "pretrain")
            contract = {**task, "config": {"array": "stage1", "route": "pretrain"}, "reference_commit": "synthetic-reference",
                "source_sha256": file_sha(self.repo / self.profiles[key]["stage1"]), "min": lower, "max": upper, "splits": splits,
                "trainer_sha256": file_sha(self.original / "scripts/reference_train.py"),
                "helper_sha256": {"reference_contract.py": file_sha(self.original / "scripts/reference_contract.py")}}
            store = self.project / "outputs/synthetic_training" / (key + "-" + str(task["seed"]))
            shards = store / "test_predictions"
            shards.mkdir(parents=True)
            best = store / "best-0002.pt"
            best.write_bytes(b"synthetic checkpoint for hash-only validation")
            truth_rows, prediction_rows = [], []
            for batch in range(splits["test"]["inner_batches"]):
                start = splits["test"]["start_target_index"] + batch * 16
                targets = np.arange(start, start + 16)
                truth = np.array([values[:, i, 0] for i in targets], dtype="float32")
                normalized = ((truth.astype("float64") - lower) / (upper - lower) + .001).astype("float32")
                save_batch(shards, batch, targets, truth, normalized, upper)
                truth_rows.append(truth)
                prediction_rows.append(inverse_author(normalized, upper))
            measured = rescore.metrics_pair(self.metrics, np.concatenate(truth_rows), np.concatenate(prediction_rows))["author"]
            result = {**task, "output_store": str(store), "best": {"file": best.name, "sha256": file_sha(best), "epoch": 2},
                "epochs": 3, "scored_origins": splits["test"]["scored_windows"], "metrics_author": measured,
                "source_array": "stage1", "split_route": "pretrain"}
            source_report = self.project / "results/slurm_jobs" / ("70000_" + str(index)) / "report"
            atomic_json(source_report / "training_contract.json", contract)
            atomic_json(source_report / "training_result.json", result)
            atomic_json(source_report.parent / "submission.json", {**task, "project": str(self.project),
                "job_id": "70000_" + str(index), "stage": "reference-train3", "code": str(self.original)})
            atomic_json(source_report.parent / "job_status.json", {"execution_state": "COMPLETED_WITH_REVIEW", "shell_exit_code": 0})
            atomic_json(store / "identity.json", contract)
            atomic_json(store / "training_result.json", result)
            atomic_json(store / "COMPLETED.json", {"result_sha256": file_sha(store / "training_result.json")})
        self.source_hashes = self.snapshot_sources()

    def snapshot_sources(self):
        roots = [self.project / "outputs/synthetic_training", self.project / "results/slurm_jobs/70000"] + [self.project / "results/slurm_jobs" / ("70000_" + str(i)) for i in range(6)]
        return {str(path): file_sha(path) for root in roots for path in root.rglob("*") if path.is_file()}

    def execute(self, name):
        report = self.project / "results/slurm_jobs" / name / "report"
        with mock.patch.object(rescore, "prepare_reference", return_value=self.repo), \
             mock.patch.object(rescore, "PROFILES", self.profiles), contextlib.redirect_stdout(io.StringIO()):
            status = rescore.run(self.project, self.code, report, "70000")
        return status, json.loads((report / "rescore_summary.json").read_text())

    def test_all_six_rescore_without_source_mutation_and_recover_derived_shards(self):
        status, summary = self.execute("80000")
        self.assertEqual(status, 3, summary.get("failures"))
        self.assertEqual(summary["completed_tasks"], 6)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["model_inference_performed"])
        for row in summary["tasks"]:
            if row["dataset"] == "pems08":
                self.assertEqual(row["legacy_metrics"], row["corrected_metrics"])
            else:
                self.assertLess(row["corrected_metrics"]["author"]["MAE"], row["legacy_metrics"]["author"]["MAE"])
        self.assertEqual(self.source_hashes, self.snapshot_sources())
        status, repeat = self.execute("80001")
        self.assertEqual(status, 3)
        self.assertTrue(all(row["reused_batches"] == row["completed_batches"] for row in repeat["tasks"]))
        self.assertEqual(self.source_hashes, self.snapshot_sources())

    def test_corrupt_saved_normalized_forecast_is_rejected_and_other_cases_survive(self):
        path = self.project / "outputs/synthetic_training/bj-11/test_predictions/batch-0000.npz"
        path.write_bytes(b"corrupt")
        status, summary = self.execute("80000")
        self.assertEqual(status, 2)
        self.assertEqual(summary["completed_tasks"], 5)
        self.assertEqual(summary["failed_tasks"], 1)
        self.assertIn("integrity", summary["failures"][0]["error"])

    def test_wrong_truth_is_rejected_even_when_the_shard_checksum_is_updated(self):
        from reference_contract import atomic_npz
        path = self.project / "outputs/synthetic_training/bj-11/test_predictions/batch-0000.npz"
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        arrays["truth"] = arrays["truth"] + 10
        atomic_npz(path, **arrays)
        atomic_json(path.with_suffix(".json"), {"batch_index": 0, "sha256": file_sha(path)})
        status, summary = self.execute("80000")
        self.assertEqual(status, 2)
        self.assertIn("targets differ", summary["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
