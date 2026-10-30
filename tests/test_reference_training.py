"""Meaningful synthetic checks: mid-epoch resume, batch identity and mail fallback."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class NotificationTests(unittest.TestCase):
    def test_group_summary_reports_partial_seeds_and_scheduler_timeout(self):
        from training_check import summarize
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            tasks = [{"dataset": d, "seed": seed} for d in ("pems08", "bj") for seed in (11, 22, 33)]
            report = project / "results/slurm_jobs/123_0/report"
            report.mkdir(parents=True)
            (report / "training_result.json").write_text(json.dumps({"metrics_author": {"MAE": 1, "RMSE": 2, "WMAPE_percent": 3}}))
            def execute(arguments):
                return {"stdout": "123_1|fixture|TIMEOUT|0:0|01:00:00|5|00:02:00||TimeLimit\n" if arguments[0] == "sacct" else "", "stderr": ""}
            with contextlib.redirect_stdout(io.StringIO()):
                result = summarize(project, "123", {"array_tasks": tasks, "job_name": "fixture"}, execute)
            self.assertEqual(result["completed_experiments"], 1)
            self.assertEqual(result["reported_failures"], 1)
            self.assertEqual(result["aggregates"]["pems08"]["completed_seeds"], 1)
            self.assertIsNone(result["aggregates"]["pems08"]["metrics"]["MAE"]["sample_std"])
            self.assertTrue((project / "results/slurm_jobs/123/report/SUMMARY.md").exists())

    def test_missing_local_mail_is_reported_without_claiming_delivery(self):
        from job_notifications import queued_receipt
        with mock.patch("job_notifications.shutil.which", return_value=None):
            result = queued_receipt(Path("/synthetic"), {"job_id": "123", "job_name": "fixture", "cpus": 5,
                "gpus": 1, "time_limit": "01:00:00"}, {"mail_user": "researcher@example.org"})
        self.assertEqual(result["state"], "UNAVAILABLE")

    def test_receipt_uses_exact_recipient_and_records_acceptance_only(self):
        from job_notifications import queued_receipt
        with mock.patch("job_notifications.shutil.which", return_value="/fake/sendmail"), \
             mock.patch("job_notifications.subprocess.run") as send:
            result = queued_receipt(Path("/synthetic"), {"job_id": "123", "job_name": "fixture", "cpus": 5,
                "gpus": 1, "time_limit": "01:00:00"}, {"mail_user": "researcher@example.org"})
        self.assertEqual(result["state"], "ACCEPTED_BY_LOCAL_MAIL_SERVICE")
        self.assertFalse(result["delivery_verified"])
        self.assertIn("To: researcher@example.org", send.call_args.kwargs["input"])


@unittest.skipUnless(importlib.util.find_spec("torch") and os.environ.get("TRAFFIC_TEST_GPU_SMOKE_CPU"), "Needs pinned Torch CPU test runtime")
class TrainingTests(unittest.TestCase):
    def test_patience_stops_on_validation_without_using_test_scores(self):
        import torch
        from reference_train import fit
        class Batches:
            def __init__(self, worsening=False):
                self.calls, self.worsening = 0, worsening
            def __len__(self):
                return 1
            def get(self, index, device):
                self.calls += 1
                return torch.ones(2, 1), torch.full((2, 1), 100.0 + self.calls if self.worsening else 0.0)
        with tempfile.TemporaryDirectory() as directory:
            result = fit(torch.nn.Linear(1, 1), Batches(), Batches(True), Path(directory), {"seed": 11},
                {"learning_rate": 0.0, "max_epochs": 10, "patience": 2, "checkpoint_batches": 50, "progress_batches": 1}, lambda **kw: None)
            self.assertEqual(result["epoch"], 4)
            self.assertEqual(result["best"]["epoch"], 2)
            self.assertEqual(result["bad_epochs"], 2)

    def test_mid_epoch_resume_matches_uninterrupted_training_and_selects_validation(self):
        import torch
        from reference_train import fit
        torch.set_num_threads(2)
        class Batches:
            def __len__(self):
                return 3
            def get(self, index, device):
                x = (torch.arange(32).reshape(8, 4).float() / 32 + index).to(device)
                return x, x.sum(-1, keepdim=True) * .1
        def model():
            torch.manual_seed(11)
            return torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        config = {"learning_rate": .001, "max_epochs": 4, "patience": 100, "checkpoint_batches": 2, "progress_batches": 1}
        identity = {"seed": 11, "config": config}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = model()
            expected = fit(complete, Batches(), Batches(), root / "complete", identity, config, lambda **kw: None)
            partial = model()
            paused = fit(partial, Batches(), Batches(), root / "resume", identity, config, lambda **kw: None, stop_after_updates=2)
            self.assertEqual((paused["epoch"], paused["cursor"]), (0, 2))
            resumed = model()
            observed = fit(resumed, Batches(), Batches(), root / "resume", identity, config, lambda **kw: None)
            self.assertEqual(observed["updates"], 12)
            self.assertEqual(observed["best"]["validation_mse"], expected["best"]["validation_mse"])
            self.assertGreaterEqual(observed["best"]["epoch"], 2)
            for name, value in complete.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]))
            self.assertLessEqual(len(list((root / "resume").glob("state-*.pt"))), 2)
            self.assertLessEqual(len(list((root / "resume").glob("best-*.pt"))), 2)
            from reference_train import load_checkpoint
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(root / "resume", {"seed": 22})
            pointer = json.loads((root / "resume/LATEST.json").read_text())
            (root / "resume" / pointer["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_checkpoint(root / "resume", identity)

    @unittest.skipUnless(os.environ.get("TRAFFIC_TEST_REFERENCE_SOURCE"), "Needs pinned original source fixture")
    def test_lazy_batches_equal_unchanged_released_loader(self):
        import numpy as np
        import torch
        from reference_train import WindowBatches
        from pems08_infer import module_from
        from pems08_audit import derive_splits, literal_assignment
        source = Path(os.environ["TRAFFIC_TEST_REFERENCE_SOURCE"])
        loader = module_from(source / "data/unified_dataset.py", "_training_test_original_loader")
        registry = literal_assignment((source / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")["pems08"]
        values = np.arange(3 * 17856 * 2, dtype="float64").reshape(3, 17856, 2) % 1147
        normalized = (values - values.min()) / (values.max() - values.min())
        splits = derive_splits((source / "data/unified_dataset.py").read_text(), registry, values, "pretrain")
        for name in ("train", "val", "test"):
            original = loader.UnifiedTrafficDataset("pems08", "pretrain", name, 5, registry["pretrain"]["tg_in_one_day"], 12, 1,
                values, registry["pretrain"]["split_config"], 16)
            lazy = WindowBatches(normalized, values, splits[name])
            self.assertEqual(len(original), len(lazy))
            for index in (0, len(lazy) // 2, len(lazy) - 1):
                x, y = lazy.get(index, "cpu")
                expected = original[index]
                self.assertTrue(torch.equal(x.squeeze(0), expected[0].float()))
                self.assertTrue(torch.equal(y, expected[1].float()))


if __name__ == "__main__":
    unittest.main()
