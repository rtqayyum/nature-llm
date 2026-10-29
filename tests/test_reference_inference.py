"""Numerical protocol and recovery tests; no real traffic weights are used."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_contract import independent_metrics, inverse_author, save_batch, load_batch, anchor_comparison
from pems08_infer import matching_preflight, strict_diagnostics
import reference_env


class ReferenceTests(unittest.TestCase):
    def test_original_wmape_excludes_zero_truth_but_mae_does_not(self):
        result = independent_metrics([[0, 2], [4, 0]], [[10, 1], [6, 7]])
        self.assertEqual(result["MAE"], 5)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(38.5))
        self.assertEqual(result["WMAPE_percent"], 50)

    def test_inverse_clips_only_negative_predictions(self):
        result = inverse_author([[-1, 0.5, 2]], 10)
        np.testing.assert_array_equal(result, [[0, 5, 20]])
        self.assertEqual(result.dtype, np.dtype("float32"))

    def test_saved_batch_resume_enforces_targets_truth_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = np.array([10, 11])
            truth = np.array([[0, 2], [3, 4]], dtype="float32")
            path = save_batch(root, 0, targets, truth, truth / 10, 10)
            self.assertIsNotNone(load_batch(root, 0, targets, truth, 10))
            with self.assertRaisesRegex(ValueError, "targets"):
                load_batch(root, 0, targets + 1, truth, 10)
            with self.assertRaisesRegex(ValueError, "transformation"):
                load_batch(root, 0, targets, truth, 20)
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_batch(root, 0, targets, truth, 10)

    def test_anchor_agreement_is_diagnostic_not_reproduction(self):
        result = anchor_comparison({"MAE": 11.87, "RMSE": 18.53, "WMAPE_percent": 5.16})
        self.assertTrue(all(result["S1"]["same_two_decimal_rounding"].values()))
        self.assertIn("unresolved", result["S1"]["interpretation"])
        self.assertTrue(all(value >= 0 for value in result["S2"]["absolute_differences"].values()))

    def test_strict_diagnostics_never_strip_prefix_or_accept_wrong_shapes(self):
        class Tensor:
            def __init__(self, shape):
                self.shape = shape
        class Model:
            def state_dict(self):
                return {"f1.weight": Tensor((2, 3))}
        result = strict_diagnostics(Model(), {"module.f1.weight": Tensor((2, 3))})
        self.assertFalse(result["compatible"])
        self.assertEqual(result["missing_keys"], ["f1.weight"])
        result = strict_diagnostics(Model(), {"f1.weight": Tensor((3, 2))})
        self.assertFalse(result["compatible"])
        self.assertEqual(len(result["shape_mismatches"]), 1)

    def test_preflight_must_match_actual_asset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for job_id, digest in [(1, "correct"), (2, "different")]:
                report = root / "results/slurm_jobs" / str(job_id) / "report"
                report.mkdir(parents=True)
                (report / "pems08_audit.json").write_text(json.dumps({"execution_status": "COMPLETED_WITH_REVIEW", "assets": {"checkpoint": {"sha256": digest}}}))
            path, _ = matching_preflight(root, {"checkpoint": {"sha256": "correct"}})
            self.assertIn("/1/", str(path))
            with self.assertRaisesRegex(ValueError, "No completed"):
                matching_preflight(root, {"checkpoint": {"sha256": "unknown"}})

    def test_environment_install_uses_only_project_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, report = root / "code", root / "report"
            (code / "config").mkdir(parents=True)
            (code / "config/reference_cpu_requirements.txt").write_text("numpy==1.24.0\n")
            def execute(args, **kwargs):
                if kwargs.get("stdout") is not None:
                    kwargs["stdout"].write("numpy==1.24.0\n")
                return subprocess.CompletedProcess(args, 1 if kwargs.get("capture_output") else 0, "", "")
            with mock.patch.object(reference_env.sys, "version_info", (3, 9, 0)), \
                 mock.patch.object(reference_env.subprocess, "run", side_effect=execute) as run:
                python = reference_env.ensure(root, code, report)
            self.assertEqual(python, root / ".venv-reference-cpu/bin/python")
            installs = [call.args[0] for call in run.call_args_list if "install" in call.args[0]]
            self.assertEqual(len(installs), 2)
            self.assertTrue(all(str(command[0]) == str(python) and "--no-user" in command for command in installs))
            self.assertTrue(all(str(command[command.index("--cache-dir") + 1]) == str(root / ".cache/pip") for command in installs))
            self.assertFalse((root / ".venv").exists())

    def test_newer_python_is_rejected_before_install(self):
        with mock.patch.object(reference_env.sys, "version_info", (3, 12, 0)), mock.patch.object(reference_env.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "Python 3.8"):
                reference_env.ensure(Path("/tmp/fixture"), Path("/tmp/fixture/code"), Path("/tmp/fixture/report"))
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
