"""Causal data boundaries, inverse recovery and optional pinned-model checks."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from causal_protocol import (CausalBatches, data_gates, features, install_fusion,
    load_forecast, model_gates, patched_forward, save_forecast, train_scaler)
from pems08_audit import derive_splits, literal_assignment
from reference_contract import independent_metrics, inverse_minmax
from test_pems08_audit import SOURCE, REGISTRY


def synthetic_data(nodes=3):
    raw = (np.arange(nodes * 320).reshape(nodes, 320) % 97).astype("float64")
    splits = derive_splits(SOURCE, REGISTRY, np.empty((nodes, 320, 2)), "pretrain")
    return raw, splits


class CausalDataTests(unittest.TestCase):
    def test_heldout_extremes_cannot_change_scaler_or_available_history(self):
        raw, splits = synthetic_data()
        scaler = train_scaler(raw, splits)
        changed = raw.copy()
        changed[:, splits["val"]["start_target_index"]:] = 100000
        self.assertEqual(scaler, train_scaler(changed, splits))
        self.assertEqual(scaler["fit_end_index_exclusive"], splits["train"]["scored_end_target_index_exclusive"])
        self.assertLess(scaler["maximum"], changed.max())
        self.assertTrue(data_gates(changed, splits, scaler)["passed"])
        original = CausalBatches(raw, splits["val"], scaler).arrays(0)[0][0]
        mutated = CausalBatches(changed, splits["val"], scaler).arrays(0)[0][0]
        np.testing.assert_array_equal(original, mutated)

    def test_prefix_trend_uses_only_the_supplied_twelve_observations(self):
        history = np.tile(np.arange(1, 13, dtype="float64"), (2, 3, 1))
        scaler = {"minimum": 0., "maximum": 12.}
        encoded = features(history, scaler)
        np.testing.assert_allclose(encoded[..., 0] * 12, history)
        np.testing.assert_allclose(encoded[..., 1] * 12, np.broadcast_to((np.arange(1, 13) + 1) / 2, history.shape))
        changed = history.copy()
        changed[..., 5:] = 9999
        np.testing.assert_array_equal(features(changed, scaler)[..., :5, :], encoded[..., :5, :])
        self.assertGreater(float(features(changed, scaler).max()), 1)  # no normalized clipping

    def test_training_and_validation_target_overlap_is_rejected(self):
        raw, splits = synthetic_data()
        splits["val"]["start_target_index"] = splits["train"]["scored_end_target_index_exclusive"] - 1
        splits["val"]["scored_end_target_index_exclusive"] = splits["val"]["start_target_index"] + splits["val"]["inner_batches"] * 16
        with self.assertRaisesRegex(ValueError, "chronological and disjoint"):
            train_scaler(raw, splits)

    def test_forecast_shards_verify_scaler_truth_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch-0000.npz"
            scaler = {"minimum": 10., "maximum": 110.}
            targets = np.array([10, 11])
            truth = np.array([[0, 20], [60, 130]], dtype="float32")
            z = np.array([[-.2, .1], [.5, 1.2]], dtype="float32")
            save_forecast(path, targets, truth, z, scaler)
            stored = load_forecast(path, targets, truth, scaler)
            np.testing.assert_array_equal(stored["predictions"], inverse_minmax(z, 10, 110))
            with self.assertRaisesRegex(ValueError, "truth/origins"):
                load_forecast(path, targets + 1, truth, scaler)
            with self.assertRaisesRegex(ValueError, "hash/scaler"):
                load_forecast(path, targets, truth, {"minimum": 0., "maximum": 110.})
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash/scaler"):
                load_forecast(path, targets, truth, scaler)

    def test_patch_only_adds_dimensions_and_rejects_unknown_fusion_source(self):
        source = '''class TrendEnhancementModule:
    def forward(self, x1, x2):
        p1 = F.softmax(x1)
        p2 = F.softmax(x2)
        return p1 * x1 + p2 * x2
'''
        changed, description = patched_forward(source)
        self.assertEqual(description["changed_calls"], 2)
        for node in ast.walk(changed):
            if isinstance(node, ast.Call):
                self.assertEqual(node.keywords[0].arg, "dim")
                self.assertEqual(node.keywords[0].value.value, -1)
                node.keywords = []
        original_fn = ast.parse(source).body[0].body[0]
        self.assertEqual(ast.dump(changed.body[0], include_attributes=False), ast.dump(original_fn, include_attributes=False))
        with self.assertRaisesRegex(ValueError, "two-call patch"):
            patched_forward(source.replace("F.softmax(x1)", "F.softmax(x1, dim=0)"))

    def test_mismatched_causal_seed_contracts_cannot_be_aggregated(self):
        from training_check import summarize
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for index in (0, 1):
                report = project / "results/slurm_jobs" / ("70000_" + str(index)) / "report"
                report.mkdir(parents=True)
                (report / "training_result.json").write_text(json.dumps({"protocol_status": "CAUSAL_INDEX_GATES_PASSED",
                    "data_contract_sha256": "different_contract_" + str(index),
                    "metrics_author": {"MAE": 1, "RMSE": 2, "WMAPE_percent": 3}}))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "mismatched data contracts"):
                summarize(project, "70000", {"stage": "causal-train3", "job_name": "fixture",
                    "array_tasks": [{"dataset": "pems08", "seed": seed} for seed in (11, 22)]},
                    lambda arguments: {"stdout": "", "stderr": ""})


@unittest.skipUnless(importlib.util.find_spec("torch") and os.environ.get("TRAFFIC_TEST_GPU_SMOKE_CPU") and os.environ.get("TRAFFIC_TEST_REFERENCE_SOURCE"), "Needs pinned Torch and provided original source")
class CausalModelTests(unittest.TestCase):
    def test_complete_causal_runner_and_completed_result_reuse_on_cpu_fixture(self):
        import torch
        from transformers import GPT2Config, GPT2Model
        import reference_train as training
        from acquire import atomic_json
        from reference_contract import file_sha
        from test_baseline_rescore import METRIC_SOURCE
        torch.set_num_threads(2)
        raw, _ = synthetic_data(8)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code, repo = project / "code", project / "work/reference/fixture"
            for path in (code / "config", code / "scripts", repo / "data", repo / "model", repo / "utils"):
                path.mkdir(parents=True)
            for name in ("reference_contract.py", "pems08_audit.py", "pems08_infer.py", "reference_datasets.py", "causal_protocol.py"):
                (code / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
            source = Path(os.environ["TRAFFIC_TEST_REFERENCE_SOURCE"]) / "model/LLM_UTP.py"
            (repo / "model/LLM_UTP.py").write_bytes(source.read_bytes())
            (repo / "data/unified_dataset.py").write_text(SOURCE)
            (repo / "data/dataset_registry.py").write_text("DATASET_REGISTRY = " + repr({"pems08": REGISTRY}))
            (repo / "utils/metrics.py").write_text(METRIC_SOURCE)
            # Supplied auxiliary channels are deliberately extreme and differ;
            # only equal raw-flow channel zero is eligible for causal inputs.
            np.save(repo / "data/one.npy", np.stack((raw, raw - 1000), axis=-1))
            np.save(repo / "data/two.npy", np.stack((raw, raw + 100000), axis=-1))
            profile = {"shape": [8, 320, 2], "stage1": "data/one.npy", "stage2": "data/two.npy", "slug": "pems08", "interval_minutes": 5}
            config = json.loads((ROOT / "config/causal_reference.json").read_text())
            config.update(datasets=["pems08"], seeds=[11], max_epochs=3)
            atomic_json(code / "config/causal_reference.json", config)
            base = project / "work/models/gpt2/synthetic"
            GPT2Model(GPT2Config(vocab_size=32, n_positions=512, n_ctx=512, n_embd=768,
                n_layer=6, n_head=12, n_inner=32)).save_pretrained(base, safe_serialization=True)
            asset = lambda path, root: {"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": file_sha(path)}
            atomic_json(code / "config/sources.lock.json", {"reference_commit": "synthetic",
                "lfs_files": [asset(repo / name, repo) for name in ("data/one.npy", "data/two.npy", "utils/metrics.py")],
                "gpt2": {"revision": "synthetic", "files": [asset(path, base) for path in base.iterdir()]}})
            reports = [project / "results" / name for name in ("first", "repeat")]
            for report in reports:
                report.mkdir(parents=True)
                (report / "reference_environment_freeze.txt").write_text("synthetic fixed CPU test environment\n")
            original_get = CausalBatches.get
            # The *test only* mocks CUDA admission and maps data/model to CPU.
            # Production retains its mandatory CUDA checks with no fallback.
            with mock.patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "0", "SLURM_CPUS_PER_TASK": "2"}), \
                 mock.patch.object(training, "PROFILES", {"pems08": profile}), \
                 mock.patch.object(training, "prepare_reference", return_value=repo), \
                 mock.patch.object(torch, "__version__", "2.0.0+cu118"), \
                 mock.patch.object(torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(torch.cuda, "device_count", return_value=1), \
                 mock.patch.object(torch.cuda, "get_device_name", return_value="SIMULATED_CPU_TEST"), \
                 mock.patch.object(torch.cuda, "get_device_capability", return_value=(0, 0)), \
                 mock.patch.object(torch.cuda, "manual_seed_all"), \
                 mock.patch.object(torch.nn.Module, "cuda", lambda self: self), \
                 mock.patch.object(CausalBatches, "get", lambda self, index, device: original_get(self, index, "cpu")), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = training.run(project, code, reports[0], causal=True)
                self.assertEqual(first, 3, (reports[0] / "failure_traceback.txt").read_text() if first != 3 else "")
                self.assertEqual(training.run(project, code, reports[1], causal=True), 3)
            result = json.loads((reports[0] / "training_result.json").read_text())
            contract = json.loads((reports[0] / "data_contract.json").read_text())
            self.assertEqual(result["protocol_status"], "CAUSAL_INDEX_GATES_PASSED")
            self.assertLess(contract["scaler"]["maximum"], 100)
            self.assertGreaterEqual(contract["scaler"]["minimum"], 0)
            self.assertNotIn("paper_comparisons_diagnostic_only", result)
            self.assertFalse(result["paper_reproduction_claim"])
            self.assertTrue(json.loads((reports[1] / "training_progress.json").read_text())["reused_completed_result"])

    def test_bad_cross_batch_model_is_rejected(self):
        import torch
        class Bad(torch.nn.Module):
            def forward(self, x):
                return x.squeeze(0)[..., 0].mean(-1).unsqueeze(-1) + x.mean()
        with self.assertRaisesRegex(ValueError, "Cross-window independence"):
            model_gates(Bad(), torch.rand(1, 16, 3, 12, 2))

    def test_original_architecture_causal_gates_epoch_fit_and_resume(self):
        import torch
        from transformers import GPT2Config, GPT2Model
        from pems08_infer import module_from
        from reference_train import fit
        from bootstrap_hpc import blob_sha
        torch.set_num_threads(2)
        source_path = Path(os.environ["TRAFFIC_TEST_REFERENCE_SOURCE"]) / "model/LLM_UTP.py"
        lock = json.loads((ROOT / "config/sources.lock.json").read_text())
        self.assertEqual(blob_sha(source_path.read_bytes()), next(row["sha"] for row in lock["reference_files"] if row["path"] == "model/LLM_UTP.py"))
        original = module_from(source_path, "_causal_synthetic_original")
        raw, splits = synthetic_data(8)
        scaler = train_scaler(raw, splits)
        train, val, test = [CausalBatches(raw, splits[name], scaler) for name in ("train", "val", "test")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "synthetic_base"
            torch.manual_seed(11)
            GPT2Model(GPT2Config(vocab_size=32, n_positions=512, n_ctx=512, n_embd=768,
                n_layer=6, n_head=12, n_inner=32)).save_pretrained(base, safe_serialization=True)
            def make_model():
                torch.manual_seed(11)
                return original.TrafficPredictionModel(8, 12, 64, [1, 2, 4], 12, 0, str(base), 768)
            complete = make_model()
            inputs, _ = train.get(0, "cpu")
            with self.assertRaisesRegex(ValueError, "Cross-window independence"):
                model_gates(complete, inputs)
            install_fusion(complete, original, source_path.read_text())
            self.assertTrue(model_gates(complete, inputs)["passed"])
            config = {"learning_rate": .0001, "max_epochs": 3, "patience": 100,
                "checkpoint_batches": 50, "checkpoint_seconds": 300, "progress_batches": 10}
            identity = {"seed": 11, "config": config, "fixture": "synthetic causal"}
            with contextlib.redirect_stdout(io.StringIO()):
                expected = fit(complete, train, val, root / "complete", identity, config, lambda **kw: None)
                partial = make_model()
                install_fusion(partial, original, source_path.read_text())
                fit(partial, train, val, root / "resume", identity, config, lambda **kw: None, stop_after_updates=2)
                del partial
                resumed = make_model()
                install_fusion(resumed, original, source_path.read_text())
                observed = fit(resumed, train, val, root / "resume", identity, config, lambda **kw: None)
            self.assertEqual(expected["best"]["validation_mse"], observed["best"]["validation_mse"])
            for name, value in complete.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
            complete.load_state_dict(torch.load(root / "complete" / expected["best"]["file"], weights_only=True), strict=True)
            self.assertTrue(model_gates(complete, inputs)["passed"])
            complete.eval()
            x, _, targets = test.arrays(0)
            with torch.no_grad():
                z = complete(torch.from_numpy(x).unsqueeze(0)).squeeze(-1).numpy()
            truth = np.array([raw[:, i] for i in targets], dtype="float32")
            measured = independent_metrics(truth, inverse_minmax(z, scaler["minimum"], scaler["maximum"]))
            self.assertTrue(all(np.isfinite(list(measured.values()))))
