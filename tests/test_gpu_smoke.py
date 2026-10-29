"""CPU synthetic checks for CUDA smoke setup and checkpoint recovery."""
import importlib.util
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
import reference_env


class GpuEnvironmentTests(unittest.TestCase):
    def test_cuda_install_has_a_separate_project_prefix_and_official_index(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code, report = project / "code", project / "report"
            (code / "config").mkdir(parents=True)
            (code / "config/reference_cpu_requirements.txt").write_text("numpy==1.24.0\n")
            def execute(args, **kwargs):
                if kwargs.get("stdout"):
                    kwargs["stdout"].write("synthetic CUDA freeze\n")
                return subprocess.CompletedProcess(args, 1 if kwargs.get("capture_output") else 0)
            with mock.patch.object(reference_env.sys, "version_info", (3, 9, 0)), \
                 mock.patch.object(reference_env.subprocess, "run", side_effect=execute) as run:
                python = reference_env.ensure(project, code, report, flavor="gpu")
            self.assertEqual(python, project / ".venv-reference-gpu/bin/python")
            installs = [call.args[0] for call in run.call_args_list if "install" in call.args[0]]
            self.assertTrue(all(str(args[0]) == str(python) and "--no-user" in args for args in installs))
            cuda = next(args for args in installs if "torch==2.0.0+cu118" in args)
            self.assertIn("https://download.pytorch.org/whl/cu118", cuda)
            self.assertFalse((project / ".venv-reference-cpu").exists())


@unittest.skipUnless(importlib.util.find_spec("torch") and os.environ.get("TRAFFIC_TEST_GPU_SMOKE_CPU"), "Needs pinned Torch CPU smoke runtime")
class SmokeResumeTests(unittest.TestCase):
    def test_optimizer_and_dropout_rng_resume_match_uninterrupted_cpu_run(self):
        import torch
        from gpu_smoke import fit_batch
        torch.set_num_threads(2)
        def model():
            torch.manual_seed(11)
            return torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
        x = torch.linspace(0, 1, 32).reshape(8, 4)
        y = torch.zeros(8, 1)
        identity = {"dataset": "synthetic", "budget": 4}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = model()
            expected = fit_batch(uninterrupted, x, y, root / "complete", identity, lambda **kw: None, steps=4)
            partial = model()
            fit_batch(partial, x, y, root / "resume", identity, lambda **kw: None, steps=2)
            resumed = model()
            result = fit_batch(resumed, x, y, root / "resume", identity, lambda **kw: None, steps=4)
            self.assertEqual(result["resumed_from_step"], 2)
            self.assertEqual(result["new_optimizer_steps"], 2)
            self.assertEqual(result["final_eval_training_batch_mse"], expected["final_eval_training_batch_mse"])
            for name, value in uninterrupted.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]))
            with self.assertRaisesRegex(ValueError, "different experiment"):
                fit_batch(model(), x, y, root / "resume", {"different": True}, lambda **kw: None, steps=4)
            pointer = json.loads((root / "resume/LATEST.json").read_text())
            (root / "resume" / pointer["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest"):
                fit_batch(model(), x, y, root / "resume", identity, lambda **kw: None, steps=4)


if __name__ == "__main__":
    unittest.main()
