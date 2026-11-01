"""Synthetic evidence chains, not real traffic measurements or Slurm jobs."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduction_review as review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.code = self.root / "work/code_versions/synthetic"
        self.report = self.root / "results/slurm_jobs/80000/report"
        self.write(self.code / "scripts/reference_datasets.py", "# synthetic source\n")
        self.write(self.code / "scripts/pems08_infer.py", "# synthetic inference source\n")
        self.write(self.code / "scripts/reference_contract.py", "# synthetic contract source\n")
        # The worker's own source belongs in the temporary project for containment.
        self.worker = self.code / "scripts/reproduction_review.py"
        self.write(self.worker, "# synthetic worker identity\n")
        self.config = json.loads((ROOT / "config/reproduction_review.json").read_text())
        self.save(self.code / "config/reproduction_review.json", self.config)
        files = []
        for name in ("main_run.py", "fine-tuning.py", "prediction.py", "model/LLM_UTP.py", "data/unified_dataset.py", "data/dataset_registry.py"):
            data = ("# synthetic " + name + "\n").encode()
            self.write(self.root / "work/reference/snapshots/fixture" / name, data)
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            files.append({"path": name, "sha": blob})
        assets = {name: hashlib.sha256(name.encode()).hexdigest() for profile in review.PROFILES.values()
            for name in (profile["stage2"], profile["checkpoint"], "utils/metrics.py")}
        self.save(self.code / "config/sources.lock.json", {"reference_commit": "fixture", "reference_files": files,
            "gpt2": {"revision": "base-fixture"}, "lfs_files": [{"path": name, "sha256": checksum} for name, checksum in assets.items()]})
        for name, item in self.config["required_jobs"].items():
            self.job(item["job_id"], item["stage"], terminal=name != "training")
        checkpoint = []
        for key in review.PUBLIC_ORDER:
            profile = review.PROFILES[key]
            measured = dict(profile["anchors"]["S1"])
            if key == "pems04":
                measured["MAE"] += 1
            store = self.root / "outputs" / key
            freeze = self.source("checkpoints") / "datasets" / profile["slug"] / "reference_environment_freeze.txt"
            self.write(freeze, "synthetic fixture environment\n")
            identity = {"config": {"dataset": key, "reference_commit": "fixture", "gpt2_base_revision": "base-fixture"},
                "assets": {name: assets[name] for name in (profile["stage2"], profile["checkpoint"], "utils/metrics.py")},
                "inference_source": review.digest(self.code / "scripts/pems08_infer.py"),
                "contract_source": review.digest(self.code / "scripts/reference_contract.py"), "environment_freeze": review.digest(freeze)}
            self.save(store / "identity.json", identity)
            self.write(store / "forecasts.npz", b"synthetic, not a numpy payload")
            result = {"dataset": key, "execution_status": "COMPLETED_WITH_REVIEW", "strict_load_performed": True,
                "training_performed": False, "evaluation_scope": "FAITHFUL_RELEASE_DEFAULT", "input_stage": "stage2", "split_route": "finetune",
                "metrics_author": measured, "metrics_independent": measured, "output_store": str(store), "scored_origins": 16,
                "run_identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "forecast_file_sha256": review.digest(store / "forecasts.npz")}
            self.save(self.source("checkpoints") / "datasets" / profile["slug"] / "inference_result.json", result)
            self.save(store / "completed.json", result)
            checkpoint.append(result)
        self.save(self.source("checkpoints") / "public8_result.json", {"execution_status": "COMPLETED_WITH_REVIEW", "completed_datasets": 8, "failed_datasets": 0, "datasets": checkpoint})
        self.save(self.source("diagnostics") / "diagnostic_result.json", {"execution_status": "COMPLETED_WITH_REVIEW",
            "cases": {str(i): {"state": "COMPLETED_WITH_REVIEW"} for i in range(10)}, "paired_comparisons": [], "aliases": {}, "selection_policy": "synthetic fixed cases"})
        self.tasks = [{"dataset": dataset, "seed": seed} for dataset in ("pems08", "bj") for seed in (11, 22, 33)]
        parent = self.source("training").parent / "submission.json"
        value = json.loads(parent.read_text())
        value["array_tasks"] = self.tasks
        self.save(parent, value)
        corrections = []
        for i, task in enumerate(self.tasks):
            key = "609006_" + str(i)
            directory = self.job(key, "reference-train3", **task)
            contract = {**task, "config": {"array": "stage1", "route": "pretrain"}, "min": -5 if task["dataset"] == "bj" else 0, "max": 100,
                "trainer_sha256": review.digest(self.worker), "helper_sha256": {}, "initialization": "synthetic fresh base",
                "reference_commit": "fixture", "gpt2_revision": "base-fixture", "splits": {}, "runtime": {"test": "CPU synthetic evidence"}}
            self.save(directory / "training_contract.json", contract)
            # Keep trainer identity different from the review worker.
            self.write(self.code / "scripts/reference_train.py", self.worker.read_bytes())
            store = self.root / "outputs/training" / key
            self.write(store / "best.pt", b"synthetic weights")
            result = {**task, "output_store": str(store), "best": {"file": "best.pt", "epoch": 2}, "epochs": 3, "optimizer_steps": 10}
            self.save(directory / "training_result.json", result)
            self.save(store / "training_result.json", result)
            self.save(store / "identity.json", contract)
            self.save(store / "COMPLETED.json", {"result_sha256": review.digest(store / "training_result.json")})
            metric = {"MAE": 2 + i, "RMSE": 4 + i, "WMAPE_percent": 3 + i}
            correction = {**task, "source_task": key, "execution_status": "COMPLETED_WITH_REVIEW", "inverse_transform": "TRAINING_MINMAX_AFFINE_V1",
                "minimum": contract["min"], "maximum": contract["max"], "corrected_metrics": {"author": metric, "independent": metric},
                "legacy_metrics": {"author": metric}, "output_store": str(self.root / "outputs/correction" / key),
                "source_inputs": {"source_contract_sha256": review.digest(directory / "training_contract.json"),
                    "source_result_sha256": review.digest(directory / "training_result.json"), "selected_model_sha256": review.digest(store / "best.pt")}}
            corrections.append(correction)
            self.save(self.source("correction") / "tasks" / key / "rescore_result.json", correction)
            self.save(Path(correction["output_store"]) / "rescore_result.json", correction)
        self.save(self.source("correction") / "rescore_summary.json", {"tasks": corrections, "source_array": "609006", "execution_status": "COMPLETED_WITH_REVIEW", "completed_tasks": 6, "failed_tasks": 0})

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)

    def save(self, path, value):
        self.write(path, json.dumps(value))

    def source(self, name):
        return self.root / "results/slurm_jobs" / self.config["required_jobs"][name]["job_id"] / "report"

    def job(self, key, stage, terminal=True, **fields):
        root = self.root / "results/slurm_jobs" / key
        self.save(root / "submission.json", {"project": str(self.root), "job_id": key, "stage": stage, "code": str(self.code), **fields})
        if terminal:
            self.save(root / "job_status.json", {"job_id": key, "stage": stage, "shell_exit_code": 0, "execution_state": "COMPLETED_WITH_REVIEW"})
        return root / "report"

    def execute(self):
        from unittest import mock
        with mock.patch.object(review, "__file__", str(self.worker)), contextlib.redirect_stdout(io.StringIO()):
            return review.run(self.root, self.code, self.report)

    def test_complete_review_preserves_inputs_recomputes_aggregates_and_never_certifies_S3(self):
        before = {str(p): review.digest(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.execute(), 3)
        summary = json.loads((self.report / "reproduction_review.json").read_text())
        self.assertEqual(summary["checkpoint_all_three_rounding_matches"], 7)
        self.assertEqual(summary["corrected_aggregates"]["pems08"]["MAE"], {"mean": 3, "sample_std": 1.0})
        self.assertFalse(summary["full_paper_reproduction"])
        self.assertTrue(summary["causal_adaptation_on_hold"])
        self.assertTrue(all(not row["exact_S3_reproduction"] for row in summary["training_experiments"]))
        self.assertTrue(all(row["state"] == "NOT_VERIFIED" for row in summary["optional_evidence"].values()))
        self.assertEqual(before, {name: review.digest(Path(name)) for name in before})
        self.assertEqual(self.execute(), 3)

    def test_corrupt_selected_weights_stop_review_preserving_partial_evidence(self):
        self.write(self.root / "outputs/training/609006_2/best.pt", b"tampered")
        self.assertEqual(self.execute(), 2)
        self.assertIn("hash mismatch", (self.report / "SUMMARY.md").read_text())
        self.assertTrue((self.report / "evidence_manifest.json").exists())

    def test_missing_required_job_does_not_become_a_scientific_review_success(self):
        (self.source("diagnostics").parent / "job_status.json").unlink()
        self.assertEqual(self.execute(), 2)

    def test_diagnostic_variant_cannot_replace_faithful_inference(self):
        path = self.source("checkpoints") / "datasets/pems08/inference_result.json"
        value = json.loads(path.read_text())
        value["evaluation_scope"] = "RELEASE_VARIANT_DIAGNOSTIC"
        self.save(path, value)
        self.assertEqual(self.execute(), 2)
        self.assertIn("diagnostic variant", (self.report / "SUMMARY.md").read_text())

    def test_escape_path_and_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            review.contained(self.root, self.root.parent / "other-project")
        with self.assertRaises(ValueError):
            review.metrics({"MAE": float("nan"), "RMSE": 1, "WMAPE_percent": 1})

    def test_legacy_format_requires_the_exact_reviewed_worker(self):
        path = self.source("checkpoints") / "datasets/pems08/inference_result.json"
        value = json.loads(path.read_text())
        for key in ("evaluation_scope", "input_stage", "split_route"):
            value.pop(key)
        self.save(path, value)
        self.save(Path(value["output_store"]) / "completed.json", value)
        self.assertEqual(self.execute(), 2)
        self.assertIn("Unknown historical", (self.report / "SUMMARY.md").read_text())
        self.config["checkpoint_legacy_schema"].update(source_commit=self.code.name,
            inference_source_sha256=review.digest(self.code / "scripts/pems08_infer.py"))
        self.save(self.code / "config/reproduction_review.json", self.config)
        self.assertEqual(self.execute(), 3)


if __name__ == "__main__":
    unittest.main()
