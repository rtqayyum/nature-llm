"""Fixed-history counterfactual and zero-update gradient audit checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import causality_audit as audit


class CausalityUnitTests(unittest.TestCase):
    def test_future_mask_preserves_all_first_history_and_available_past(self):
        mask = audit.future_mask()
        self.assertEqual(mask.shape, (16, 12))
        self.assertFalse(mask[0].any())
        self.assertTrue(mask[1, 11])
        self.assertFalse(mask[1, :11].any())
        self.assertTrue(mask[12:].all())
        for companion, position in np.ndindex(mask.shape):
            self.assertEqual(bool(mask[companion, position]), companion - 12 + position >= 0)

    def run_fixture(self, failure=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = Path(temporary.name)
        code, report = project / "code", project / "report"
        code.mkdir()
        calls = []
        def execute(command, **kwargs):
            key = command[command.index("--dataset") + 1]
            destination = Path(command[command.index("--report") + 1])
            calls.append((key, kwargs["env"]["SLURM_CPUS_PER_TASK"]))
            row = {"execution_status": "COMPLETED_WITH_REVIEW", "dataset": key, "strict_load_performed": True,
                "optimizer_steps": 0, "parameter_versions_unchanged": True,
                "first_target_future_input_dependence_observed": True,
                "declared_trainable_elements": 100, "elements_receiving_gradients": 20}
            if key == failure:
                row = {"execution_status": "FAILED", "dataset": key, "error": "synthetic checkpoint mismatch"}
            (destination / "causality_result.json").write_text(json.dumps(row))
            return subprocess.CompletedProcess(command, 2 if key == failure else 3)
        with mock.patch.object(audit.subprocess, "run", side_effect=execute), mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4"}):
            status = audit.run(project, code, report)
        return status, json.loads((report / "causality_summary.json").read_text()), calls

    def test_all_eight_are_audited_with_bounded_cpu_and_zero_updates(self):
        status, result, calls = self.run_fixture()
        self.assertEqual(status, 3)
        self.assertEqual(len(result["datasets"]), 8)
        self.assertTrue(all(threads == "2" for _, threads in calls))
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["training_experiments_completed"], 0)

    def test_required_failure_retains_other_seven_results(self):
        status, result, calls = self.run_fixture("pems04")
        self.assertEqual((status, len(calls)), (2, 8))
        self.assertEqual(sum(row["execution_status"] == "COMPLETED_WITH_REVIEW" for row in result["datasets"].values()), 7)
        self.assertEqual(result["datasets"]["pems04"]["error"], "synthetic checkpoint mismatch")


@unittest.skipUnless(importlib.util.find_spec("torch") and os.environ.get("TRAFFIC_TEST_CAUSALITY_RUNTIME"), "Needs pinned probe runtime")
class TorchCausalityTests(unittest.TestCase):
    def check_model(self, coupled):
        import torch
        import torch.nn.functional as functional
        torch.set_num_threads(2)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, x):
                feature = x[0, :, :, -1, 0].unsqueeze(-1)
                return feature * self.weight * (functional.softmax(feature) if coupled else 1)
        model = Model()
        values = torch.linspace(.1, .9, 16 * 3 * 12 * 2).reshape(1, 16, 3, 12, 2)
        before = model.weight.detach().clone()
        result, _ = audit.probe(model, values, values, torch.zeros(16, 3, 1), 10)
        self.assertEqual(result["first_target_future_input_dependence_observed"], coupled)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(torch.equal(before, model.weight.detach()))
        self.assertTrue(result["parameter_contract"][0]["received_gradient"])
    def test_detects_companion_dependent_model(self):
        self.check_model(True)
    def test_independent_model_is_not_flagged(self):
        self.check_model(False)


if __name__ == "__main__":
    unittest.main()
