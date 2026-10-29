"""Synthetic orchestration: no Slurm, traffic weights or external downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public8_infer
from reference_datasets import PROFILES, PUBLIC_ORDER


class PublicBatchTests(unittest.TestCase):
    def run_fixture(self, failing=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = Path(temporary.name)
        code = project / "code"
        code.mkdir()
        report = project / "results/slurm_jobs/1/report"
        report.mkdir(parents=True)
        (report / "reference_environment_freeze.txt").write_text("synthetic packages\n")
        calls = []
        def execute(command, **kwargs):
            key = command[command.index("--dataset") + 1]
            calls.append((key, command, kwargs["env"]))
            destination = Path(command[command.index("--report") + 1])
            if key == failing:
                (destination / "inference_result.json").write_text(json.dumps({"execution_status": "FAILED", "error": "synthetic shape mismatch"}))
                kwargs["stderr"].write("synthetic shape mismatch\n")
                return subprocess.CompletedProcess(command, 2)
            anchor = PROFILES[key]["anchors"]["S1"]
            (destination / "inference_result.json").write_text(json.dumps({"execution_status": "COMPLETED_WITH_REVIEW",
                "dataset": key, "strict_load_performed": True, "metrics_author": anchor, "metrics_independent": anchor,
                "scored_origins": 16, "output_store": str(project / "outputs" / key),
                "paper_anchor_comparisons": {"S1": {"same_two_decimal_rounding": {name: True for name in anchor}}}}))
            return subprocess.CompletedProcess(command, 3)
        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "16", "SLURM_JOB_ID": "1", "SLURM_JOB_NAME": "synthetic-public8"}), \
             mock.patch.object(public8_infer.subprocess, "run", side_effect=execute):
            result = public8_infer.run(project, code, report, python="/synthetic/python")
        return project, report, result, calls

    def test_all_eight_have_isolated_paths_and_bounded_threads(self):
        project, report, status, calls = self.run_fixture()
        self.assertEqual(status, 3)
        summary = json.loads((report / "public8_result.json").read_text())
        self.assertEqual((summary["completed_datasets"], summary["failed_datasets"]), (8, 0))
        self.assertEqual({key for key, _, _ in calls}, set(PUBLIC_ORDER))
        self.assertTrue(all(env["SLURM_CPUS_PER_TASK"] == "4" and "--preflight" in command for _, command, env in calls))
        self.assertEqual(len({command[command.index("--report") + 1] for _, command, _ in calls}), 8)
        self.assertEqual(len(list((project / "logs").rglob("*.out"))), 8)
        self.assertEqual(summary["training_experiments_completed"], 0)

    def test_one_required_failure_does_not_hide_seven_successes(self):
        _, report, status, calls = self.run_fixture(failing="taxi")
        self.assertEqual(status, 2)
        summary = json.loads((report / "public8_result.json").read_text())
        self.assertEqual((len(calls), summary["completed_datasets"], summary["failed_datasets"]), (8, 7, 1))
        self.assertEqual(summary["datasets"][-1]["reason"], "synthetic shape mismatch")
        self.assertIn("synthetic shape mismatch", (report / "SUMMARY.md").read_text())

    def test_profiles_match_pinned_asset_paths_without_restricted_examples(self):
        lock = json.loads((ROOT / "config/sources.lock.json").read_text())
        paths = {item["path"] for item in lock["lfs_files"]}
        self.assertEqual(len(PROFILES), 8)
        for key in PUBLIC_ORDER:
            self.assertTrue({PROFILES[key][name] for name in ("stage1", "stage2", "checkpoint")} <= paths)
        self.assertTrue(set(PUBLIC_ORDER).isdisjoint({"nn", "bus", "hsr"}))


if __name__ == "__main__":
    unittest.main()
