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
            config = GPT2Config(vocab_size=32, n_positions=512, n_ctx=512, n_embd=768, n_layer=6, n_head=12, n_inner=32)
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
        self.synthetic_roundtrip("pems08")

    def test_all_public_original_loader_ranges_and_targets(self):
        import gc
        import numpy as np
        from pems08_infer import module_from
        from pems08_audit import derive_splits, literal_assignment
        from bootstrap_hpc import blob_sha
        from reference_datasets import PROFILES, PUBLIC_ORDER
        original = Path(os.environ["TRAFFIC_TEST_REFERENCE_SOURCE"])
        lock = json.loads((ROOT / "config/sources.lock.json").read_text())
        for name in ("data/unified_dataset.py", "data/dataset_registry.py"):
            expected = next(row["sha"] for row in lock["reference_files"] if row["path"] == name)
            self.assertEqual(blob_sha((original / name).read_bytes()), expected)
        module = module_from(original / "data/unified_dataset.py", "_all_public_loader_test")
        registry = literal_assignment((original / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")
        # Independently calculated from released lengths and registry routes.
        expected_ranges = {"pems03": (23587, 26207, 2608), "pems04": (15292, 16991, 1696),
            "pems08": (16070, 17855, 1776), "sh": (4686, 5214, 528), "bj": (1512, 1800, 288),
            "hz": (1334, 1650, 304), "bike": (7884, 8760, 864), "taxi": (7884, 8760, 864)}
        total = 0
        for key in PUBLIC_ORDER:
            with self.subTest(dataset=key):
                shape = PROFILES[key]["shape"]
                values = np.broadcast_to(np.arange(shape[1], dtype="float32")[None, :, None], shape)
                route = registry[key]["finetune"]
                audited = derive_splits((original / "data/unified_dataset.py").read_text(), registry[key], values, "finetune")["test"]
                start, end, count = expected_ranges[key]
                self.assertEqual((audited["start_target_index"], audited["end_target_index_exclusive"], audited["scored_windows"]), (start, end, count))
                dataset = module.UnifiedTrafficDataset(key, "predict", "test", 15, route["tg_in_one_day"], 12, 1, values, route["split_config"], 16)
                self.assertEqual((dataset.start_index, dataset.end_index, len(dataset) * 16), (start, end, count))
                self.assertEqual(float(dataset[0][2][0, 0, 0]), start)
                self.assertEqual(float(dataset[len(dataset) - 1][2][-1, 0, 0]), start + count - 1)
                total += count
                del dataset, values
                gc.collect()
        self.assertEqual(total, 8928)

    @unittest.skipUnless(os.environ.get("TRAFFIC_TEST_REFERENCE_METRICS"), "Needs separately verified original metric source")
    def test_beijing_preflight_inference_and_resume(self):
        self.synthetic_roundtrip("bj")

    def synthetic_roundtrip(self, dataset):
        import torch
        import numpy as np
        from transformers import GPT2Config, GPT2Model
        from pems08_infer import module_from
        from pems08_audit import derive_splits, literal_assignment
        from bootstrap_hpc import blob_sha
        from reference_contract import file_sha
        from reference_datasets import PROFILES
        profile = PROFILES[dataset]
        nodes, length, channels = profile["shape"]
        scale = 1147 if dataset == "pems08" else 4744
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
            for name in ("model", "data/stage1", "data/stage2", "utils", str(Path(profile["checkpoint"]).parent)):
                (repo / name).mkdir(parents=True, exist_ok=True)
            for name in ("model/LLM_UTP.py", "data/unified_dataset.py", "data/dataset_registry.py"):
                (repo / name).write_bytes((original / name).read_bytes())
            (repo / "utils/metrics.py").write_bytes(metrics.read_bytes())
            base = project / "work/models/gpt2/synthetic"
            config = GPT2Config(vocab_size=32, n_positions=512, n_ctx=512, n_embd=768, n_layer=6, n_head=12, n_inner=32)
            torch.set_num_threads(2)
            torch.manual_seed(11)
            GPT2Model(config).save_pretrained(base, safe_serialization=True)
            module = module_from(repo / "model/LLM_UTP.py", "_full_synthetic_author")
            model = module.TrafficPredictionModel(nodes, 12, 64, [1, 2, 4], 12, 0, str(base), 768)
            torch.save(model.state_dict(), repo / profile["checkpoint"])
            del model
            values = (np.arange(nodes * length * channels, dtype="float64") % (scale + 1)).reshape(nodes, length, channels)
            np.save(repo / profile["stage2"], values)
            np.save(repo / profile["stage1"], values)
            mutable = [profile["stage1"], profile["stage2"], profile["checkpoint"], "utils/metrics.py"]
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
            registry = literal_assignment((repo / "data/dataset_registry.py").read_text(), "DATASET_REGISTRY")[dataset]
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
                        "--code", str(code), "--report", str(report)] + (["--dataset", "bj", "--preflight"] if dataset == "bj" else []), env={**os.environ, "SLURM_CPUS_PER_TASK": "2"}, stdout=log, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 3, (report / "run.log").read_text()[-4000:])
                output = json.loads((report / "inference_result.json").read_text())
                self.assertTrue(output["strict_load_performed"])
                expected_origins = 1776 if dataset == "pems08" else 288
                self.assertEqual((output["scored_origins"], output["nodes"]), (expected_origins, nodes))
                if dataset == "bj":
                    audited = json.loads((report / "beijing_audit.json").read_text())
                    row = audited["splits"]["finetune"]["test"]
                    self.assertEqual((row["start_target_index"], row["end_target_index_exclusive"], row["dropped_tail_windows"]), (1512, 1800, 0))
                    train = audited["splits"]["finetune"]["train"]
                    self.assertEqual((train["start_target_index"], train["end_target_index_exclusive"], train["scored_windows"]), (503, 1007, 496))
                    self.assertEqual(audited["overlaps"]["finetune_train_val_scored_target_origins"], 0)
                    self.assertTrue((report / "preflight_SUMMARY.md").exists())
                progress = json.loads((report / "inference_progress.json").read_text())
                if identity is not None:
                    self.assertEqual(output["run_identity"], identity)
                    self.assertEqual(progress["resumed_batches"], expected_origins // 16)
                identity = output["run_identity"]
            self.assertTrue(all(file_sha(repo / item["path"]) == item["sha256"] for item in lfs))


if __name__ == "__main__":
    unittest.main()
