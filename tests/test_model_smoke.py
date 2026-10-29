"""Optional bounded forward test: real pinned framework, synthetic weights only.

Set TRAFFIC_TEST_REFERENCE_SOURCE to a separately acquired pinned source tree.
No author dataset/checkpoint/base-model payload is downloaded by this test.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@unittest.skipUnless(importlib.util.find_spec("torch") and os.environ.get("TRAFFIC_TEST_REFERENCE_SOURCE"),
                     "Needs pinned reference runtime and separately provided source")
class OriginalModelSmoke(unittest.TestCase):
    def test_unchanged_author_model_strict_load_and_two_block_forward(self):
        import torch
        import transformers
        from transformers import GPT2Config, GPT2Model
        from pems08_infer import module_from, strict_diagnostics
        self.assertEqual((torch.__version__, transformers.__version__), ("2.0.0+cpu", "4.35.0"))
        source = Path(os.environ["TRAFFIC_TEST_REFERENCE_SOURCE"]) / "model/LLM_UTP.py"
        data = source.read_bytes()
        digest = hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest()
        lock = json.loads((ROOT / "config/sources.lock.json").read_text())
        expected = next(item for item in lock["reference_files"] if item["path"] == "model/LLM_UTP.py")
        self.assertEqual(digest, expected["sha"])
        torch.set_num_threads(2)
        torch.manual_seed(11)
        module = module_from(source, "_synthetic_smoke_author_model")
        temporary_root = ROOT / "tmp/synthetic_smoke"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            base = root / "synthetic_gpt2"
            # Small synthetic vocabulary/MLP reduce test cost; no real traffic
            # metrics or pretrained-language-model claim follows from this test.
            config = GPT2Config(vocab_size=32, n_positions=256, n_ctx=256, n_embd=768, n_layer=6, n_head=12, n_inner=32)
            GPT2Model(config).save_pretrained(base, safe_serialization=True)
            model = module.TrafficPredictionModel(170, 12, 64, [1, 2, 4], 12, 0, str(base), 768)
            checkpoint = root / "synthetic_state.pth"
            torch.save(model.state_dict(), checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue(strict_diagnostics(model, state)["compatible"])
            model.load_state_dict(state, strict=True)
            model.eval()
            executed = []
            hooks = [block.register_forward_hook(lambda m, i, o, index=index: executed.append(index)) for index, block in enumerate(model.gpt2.h)]
            inputs = torch.randn(1, 16, 170, 12, 2)
            with torch.no_grad():
                output = model(inputs)
            for hook in hooks:
                hook.remove()
            self.assertEqual(executed, [0, 1])
            self.assertEqual(tuple(output.shape), (16, 170, 1))
            self.assertTrue(torch.isfinite(output).all())
            with torch.no_grad():
                self.assertTrue(torch.allclose(output, model(inputs), rtol=1e-5, atol=1e-6))

    @unittest.skipUnless(os.environ.get("TRAFFIC_TEST_REFERENCE_METRICS"), "Needs separately verified original metric source")
    def test_full_synthetic_inference_and_resume(self):
        import torch
        import numpy as np
        from transformers import GPT2Config, GPT2Model
        from pems08_infer import module_from
        from pems08_audit import derive_splits, literal_assignment
        from bootstrap_hpc import blob_sha
        from reference_contract import file_sha
        original = Path(os.environ["TRAFFIC_TEST_REFERENCE_SOURCE"])
        metrics = Path(os.environ["TRAFFIC_TEST_REFERENCE_METRICS"])
        published = json.loads((ROOT / "config/sources.lock.json").read_text())
        expected = next(item for item in published["lfs_files"] if item["path"] == "utils/metrics.py")
        self.assertEqual(file_sha(metrics), expected["sha256"])
        temporary_root = ROOT / "tmp/synthetic_smoke"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            project = Path(directory).resolve()
            code = project / "work/code_versions/synthetic"
            (code / "config").mkdir(parents=True)
            (code / "scripts").mkdir()
            (code / "scripts/reference_contract.py").write_bytes((ROOT / "scripts/reference_contract.py").read_bytes())
            commit = "d" * 40
            repo = project / "work/reference/snapshots" / commit
            for name in ("model", "data/stage2", "utils", "checkpoint/08"):
                (repo / name).mkdir(parents=True, exist_ok=True)
            for name in ("model/LLM_UTP.py", "data/unified_dataset.py", "data/dataset_registry.py"):
                (repo / name).write_bytes((original / name).read_bytes())
            (repo / "utils/metrics.py").write_bytes(metrics.read_bytes())
            base = project / "work/models/gpt2/synthetic"
            config = GPT2Config(vocab_size=32, n_positions=256, n_ctx=256, n_embd=768, n_layer=6, n_head=12, n_inner=32)
            torch.set_num_threads(2)
            torch.manual_seed(11)
            GPT2Model(config).save_pretrained(base, safe_serialization=True)
            module = module_from(repo / "model/LLM_UTP.py", "_full_synthetic_author")
            model = module.TrafficPredictionModel(170, 12, 64, [1, 2, 4], 12, 0, str(base), 768)
            torch.save(model.state_dict(), repo / "checkpoint/08/best_model_dict.pth")
            del model
            values = (np.arange(170 * 17856 * 2, dtype="float64") % 1148).reshape(170, 17856, 2)
            np.save(repo / "data/stage2/08fine-tuning.npy", values)
            mutable = ["data/stage2/08fine-tuning.npy", "checkpoint/08/best_model_dict.pth", "utils/metrics.py"]
            lfs, files = [], []
            for path in sorted(repo.rglob("*")):
                if not path.is_file():
                    continue
                name, content = str(path.relative_to(repo)), path.read_bytes()
                if name in mutable:
                    sha = hashlib.sha256(content).hexdigest()
                    lfs.append({"path": name, "sha256": sha, "size": len(content)})
                    content = ("version https://git-lfs.github.com/spec/v1\noid sha256:" + sha + "\nsize " + str(len(content)) + "\n").encode()
                files.append({"path": name, "mode": "100644", "size": len(content), "sha": blob_sha(content)})
            (repo / ".source_snapshot.json").write_text(json.dumps({"repository": "LLM-UTP/LLM-UTP", "commit": commit, "files": files}))
            basefiles = [{"path": path.name, "size": path.stat().st_size, "sha256": file_sha(path)} for path in base.iterdir() if path.is_file()]
            lock = {"reference_commit": commit, "reference_files": files, "lfs_files": lfs,
                    "gpt2": {"revision": "synthetic", "files": basefiles}}
            (code / "config/sources.lock.json").write_text(json.dumps(lock))
            preflight = project / "results/slurm_jobs/1/report"
            preflight.mkdir(parents=True)
            registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")["pems08"]
            splits = derive_splits((repo / "data/unified_dataset.py").read_text(), registry, values, "finetune")
            (preflight / "pems08_audit.json").write_text(json.dumps({"execution_status": "COMPLETED_WITH_REVIEW",
                "assets": {item["path"]: item for item in lfs}, "splits": {"finetune": splits}}))
            identity = None
            for number in (2, 3):
                report = project / "results/slurm_jobs" / str(number) / "report"
                report.mkdir(parents=True)
                (report / "reference_environment_freeze.txt").write_text("Synthetic integration fixture; torch=2.0.0+cpu transformers=4.35.0\n")
                with (report / "run.log").open("w") as log:
                    result = subprocess.run([sys.executable, str(ROOT / "scripts/pems08_infer.py"), "--project", str(project),
                        "--code", str(code), "--report", str(report)], env={**os.environ, "SLURM_CPUS_PER_TASK": "2"}, stdout=log, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 3, (report / "run.log").read_text()[-4000:])
                output = json.loads((report / "inference_result.json").read_text())
                self.assertTrue(output["strict_load_performed"])
                self.assertEqual((output["scored_origins"], output["nodes"]), (1776, 170))
                progress = json.loads((report / "inference_progress.json").read_text())
                if identity is not None:
                    self.assertEqual(output["run_identity"], identity)
                    self.assertEqual(progress["resumed_batches"], 111)
                identity = output["run_identity"]
            self.assertTrue(all(file_sha(repo / item["path"]) == item["sha256"] for item in lfs))


if __name__ == "__main__":
    unittest.main()
