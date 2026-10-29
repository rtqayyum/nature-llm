"""Synthetic fixed-plan, matched-cohort and failure-isolation checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import release_diagnose as diagnostic
from reference_contract import file_sha, independent_metrics
from test_pems08_audit import SOURCE


class DiagnosticTests(unittest.TestCase):
    def test_archive_inventory_does_not_extract_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, report = root / "code", root / "report"
            (code / "config").mkdir(parents=True)
            archive = root / "work/reference/snapshots/synthetic/Source Data/Source Data.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../../must_not_extract.txt", "synthetic")
                stream.writestr("Table1.xlsx", "synthetic source table")
            asset = {"path": "Source Data/Source Data.zip", "size": archive.stat().st_size, "sha256": file_sha(archive)}
            (code / "config/sources.lock.json").write_text(json.dumps({"reference_commit": "synthetic", "lfs_files": [asset]}))
            inventory = diagnostic.inventory_source_data(root, code, report)
            self.assertEqual(inventory["member_count"], 2)
            self.assertFalse((root / "must_not_extract.txt").exists())
            self.assertFalse((root / "Table1.xlsx").exists())

    def test_fixed_plan_deduplicates_equal_ranges_but_not_input_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, repo = root / "code", root / "repo"
            (code / "config").mkdir(parents=True)
            (repo / "data").mkdir(parents=True)
            registry, profiles, files = {}, {}, []
            for key in diagnostic.TARGETS:
                length = 1001 if key == "pems04" else 1000
                registry[key] = {
                    "pretrain": {"split_config": {"mode": "ratio_tail", "train_ratio": .8, "val_ratio": .1, "test_ratio": .1}},
                    "finetune": {"split_config": {"mode": "ratio_window", "train_len_ratio": .1,
                        "val_start_ratio": .1, "val_len_ratio": .1, "test_start_ratio": .9,
                        "test_len_ratio": .08 if key == "sh" else .1}}}
                profile = {"shape": [1, length, 2]}
                for stage in ("stage1", "stage2"):
                    name = key + "_" + stage + ".npy"
                    values = np.arange(length * 2, dtype="float32").reshape(1, length, 2)
                    if stage == "stage1":
                        values[..., 1] += .25
                    np.save(repo / name, values)
                    profile[stage] = name
                profile["checkpoint"] = key + ".pth"
                (repo / profile["checkpoint"]).write_bytes(b"synthetic checkpoint")
                for name in (profile["stage1"], profile["stage2"], profile["checkpoint"]):
                    files.append({"path": name, "size": (repo / name).stat().st_size, "sha256": file_sha(repo / name)})
                profiles[key] = profile
            (repo / "data/dataset_registry.py").write_text("DATASET_REGISTRY = " + repr(registry))
            (repo / "data/unified_dataset.py").write_text(SOURCE)
            (code / "config/sources.lock.json").write_text(json.dumps({"reference_commit": "synthetic", "lfs_files": files}))
            with mock.patch.object(diagnostic, "PROFILES", profiles), mock.patch.object(diagnostic, "prepare_reference", return_value=repo):
                plan = diagnostic.make_plan(root, code)
            self.assertEqual(len(plan["cases"]), 10)
            self.assertEqual(plan["aliases"], {"bj__stage2-pretrain": "bj__stage2-finetune", "bj__stage1-pretrain": "bj__stage1-finetune"})
            self.assertNotIn("best", plan)
            self.assertIn("no metric-based selection", plan["selection_policy"])

    def test_pair_uses_common_target_ids_not_row_position(self):
        left = {"target_indices": np.array([1, 2, 3]), "truth": np.array([[10.], [20.], [30.]]),
                "predictions": np.array([[9.], [18.], [27.]])}
        right = {"target_indices": np.array([2, 3, 4]), "truth": np.array([[20.], [30.], [40.]]),
                 "predictions": np.array([[19.], [28.], [37.]])}
        result = diagnostic.paired_comparison(left, right)
        self.assertEqual(result["common_origins"], 2)
        self.assertEqual(result["mean_absolute_prediction_difference"], 1)
        self.assertEqual(result["left_common_metrics"]["MAE"], 2.5)
        right["truth"][0] = 999
        with self.assertRaisesRegex(ValueError, "truths disagree"):
            diagnostic.paired_comparison(left, right)

    def test_changed_forecast_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "forecasts.npz", target_indices=[1], truth=[[1]], predictions=[[1]])
            result = {"output_store": str(root), "forecast_file_sha256": "wrong"}
            with self.assertRaisesRegex(ValueError, "digest"):
                diagnostic.read_forecasts(root, result)

    def run_batch(self, failing=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        code, report = root / "code", root / "report"
        code.mkdir()
        report.mkdir()
        (report / "reference_environment_freeze.txt").write_text("synthetic environment\n")
        cases, aliases = [], {}
        for key in diagnostic.TARGETS:
            for stage in ("stage2", "stage1"):
                case_id = key + "__" + stage + "-finetune"
                cases.append({"dataset": key, "variant": stage + "-finetune", "case_id": case_id,
                    "split": {"start_target_index": 10, "scored_end_target_index_exclusive": 26}})
                aliases[key + "__" + stage + "-pretrain"] = case_id
        plan = {"cases": cases, "aliases": aliases, "selection_policy": "retain all fixed cases"}
        def execute(command, **kwargs):
            destination = Path(command[command.index("--report") + 1])
            key = command[command.index("--dataset") + 1]
            variant = command[command.index("--release-variant") + 1]
            if key + "__" + variant == failing:
                (destination / "inference_result.json").write_text(json.dumps({"error": "synthetic strict failure"}))
                return subprocess.CompletedProcess(command, 2)
            store = root / "outputs" / (key + "__" + variant)
            store.mkdir(parents=True)
            truth = np.arange(1, 17, dtype="float32")[:, None]
            np.savez(store / "forecasts.npz", target_indices=np.arange(10, 26), truth=truth, predictions=truth)
            metric = independent_metrics(truth, truth)
            result = {"dataset": key, "release_variant": variant, "strict_load_performed": True,
                "output_store": str(store), "forecast_file_sha256": file_sha(store / "forecasts.npz"),
                "metrics_author": metric, "scored_origins": 16}
            (destination / "inference_result.json").write_text(json.dumps(result))
            (destination / "evaluation_config.json").write_text(json.dumps({"normalization_minimum": 0, "normalization_maximum": 16}))
            self.assertEqual(kwargs["env"]["SLURM_CPUS_PER_TASK"], "4")
            return subprocess.CompletedProcess(command, 3)
        with mock.patch.object(diagnostic, "make_plan", return_value=plan), \
             mock.patch.object(diagnostic, "inventory_source_data", return_value={"sha256": "synthetic"}), \
             mock.patch.object(diagnostic.subprocess, "run", side_effect=execute), \
             mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "12"}):
            status = diagnostic.run(root, code, report)
        return status, json.loads((report / "diagnostic_result.json").read_text()), report

    def test_controller_retains_every_case_and_paired_comparison(self):
        status, result, report = self.run_batch()
        self.assertEqual(status, 3)
        self.assertEqual(len(result["cases"]), 6)
        self.assertEqual(len(result["paired_comparisons"]), 12)
        self.assertTrue(all(row["mean_absolute_prediction_difference"] == 0 for row in result["paired_comparisons"]))
        self.assertTrue((report / "diagnostic_plan.json").exists())
        self.assertEqual(result["training_experiments_completed"], 0)

    def test_required_failure_preserves_other_completed_cases(self):
        status, result, _ = self.run_batch("pems04__stage1-finetune")
        self.assertEqual(status, 2)
        self.assertEqual(sum(row["state"] == "COMPLETED_WITH_REVIEW" for row in result["cases"].values()), 5)
        self.assertTrue(any("error" in row for row in result["paired_comparisons"]))


if __name__ == "__main__":
    unittest.main()
