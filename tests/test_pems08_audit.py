"""Synthetic arrays/checkpoints and independently checked range expectations."""
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pems08_audit as audit
from bootstrap_hpc import blob_sha


# A small loader contract fixture, including deliberately unexecutable imports
# and a constructor. The extractor must evaluate only the split methods.
SOURCE = '''
import library_that_is_not_installed
raise RuntimeError("do not execute the original module")
class UnifiedTrafficDataset:
    def __init__(self):
        raise RuntimeError("do not run the original constructor")
    def _total_len(self):
        return self.inflow_data.shape[1]
    def _build_start_end(self):
        if self.split_config["mode"] == "ratio_tail":
            return self._ratio_tail(self._total_len())
        return self._ratio_window(self._total_len())
    def _ratio_tail(self, total_len):
        if self.split == "train":
            return self.time_lag, int(total_len * self.split_config["train_ratio"])
        if self.split == "val":
            start = int(total_len * self.split_config["train_ratio"])
            return start, start + int(total_len * self.split_config["val_ratio"])
        return total_len - int(total_len * self.split_config["test_ratio"]), total_len
    def _ratio_window(self, total_len):
        if self.split == "train":
            return self.time_lag, self.time_lag + int(total_len * self.split_config["train_len_ratio"])
        if self.split == "val":
            start = int(total_len * self.split_config["val_start_ratio"])
            return start, start + int(total_len * self.split_config["val_len_ratio"])
        start = int(total_len * self.split_config["test_start_ratio"])
        return start, start + int(total_len * self.split_config["test_len_ratio"])
    def __len__(self):
        return len(self.X) // self.inner_batch_size
'''
REGISTRY = {"pretrain": {"split_config": {"mode": "ratio_tail", "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}},
    "finetune": {"max_inflow": 1147, "split_config": {"mode": "ratio_window", "train_len_ratio": 0.1,
        "val_start_ratio": 0.1, "val_len_ratio": 0.1, "test_start_ratio": 0.9, "test_len_ratio": 0.1}}}


def write_checkpoint(path, payload):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))
        archive.writestr("archive/data/0", b"\0" * 16)
        archive.writestr("archive/version", b"3\n")


class PemsAuditTests(unittest.TestCase):
    def test_original_module_is_not_imported_and_tail_coordinates_are_exact(self):
        values = np.empty((1, 17856, 2), dtype="uint8")
        rows = audit.derive_splits(SOURCE, REGISTRY, values, "finetune")
        self.assertEqual((rows["test"]["start_target_index"], rows["test"]["end_target_index_exclusive"]), (16070, 17855))
        self.assertEqual((rows["test"]["constructed_windows"], rows["test"]["scored_windows"], rows["test"]["dropped_tail_windows"]), (1785, 1776, 9))
        self.assertEqual(audit.intersection_count(rows["train"], rows["val"], False), 12)
        self.assertEqual(audit.intersection_count(rows["train"], rows["val"]), 3)
        pretrain = audit.derive_splits(SOURCE, REGISTRY, values, "pretrain")
        self.assertEqual(pretrain["test"]["start_target_index"], 16071)
        self.assertEqual(audit.intersection_count(pretrain["train"], rows["test"]), 0)

    def test_origin_ledger_marks_discarded_tail_and_causal_input_boundary(self):
        row = audit.derive_splits(SOURCE, REGISTRY, np.empty((1, 17856, 2)), "finetune")["test"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "origins.csv"
            audit.write_origins(path, row)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual((rows[0]["forecast_origin_index"], rows[0]["input_start"], rows[0]["target_index"]), ("16069", "16058", "16070"))
            self.assertEqual(sum(int(item["included_by_author_inner_batches"]) for item in rows), 1776)
            self.assertEqual(rows[-1]["target_index"], "17854")

    def test_checkpoint_inspection_never_executes_pickle_reduce(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must_not_exist"
            class Payload:
                def __reduce__(self):
                    return os.system, ("touch " + str(marker),)
            path = Path(directory) / "checkpoint.pth"
            write_checkpoint(path, {"gpt2.h.0.attn.c_attn.weight": Payload(), "gpt2.h.5.ln_1.weight": None})
            result = audit.checkpoint_metadata(path)
            self.assertFalse(marker.exists())
            self.assertFalse(result["pickle_executed"])
            self.assertFalse(result["strict_model_load_performed"])
            self.assertEqual(result["gpt_block_indices_in_serialized_names"], [0, 5])

    def test_config_is_literal_only(self):
        with self.assertRaises(ValueError):
            audit.literal_assignment("DATASET_REGISTRY = dict()", "DATASET_REGISTRY")

    def test_full_preflight_is_read_only_and_missing_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code = project / "work/code_versions/test"
            (code / "config").mkdir(parents=True)
            commit = "c" * 40
            repo = project / "work/reference/snapshots" / commit
            (repo / "data/stage1").mkdir(parents=True)
            (repo / "data/stage2").mkdir(parents=True)
            (repo / "checkpoint/08").mkdir(parents=True)
            (repo / "utils").mkdir()
            data = np.arange(170 * 1000 * 2, dtype="float64").reshape(170, 1000, 2) % 1148
            np.save(repo / "data/stage2/08fine-tuning.npy", data)
            first = data.copy()
            first[..., 1] += 0.25
            np.save(repo / "data/stage1/08.npy", first)
            write_checkpoint(repo / "checkpoint/08/best_model_dict.pth", {"gpt2.h.0.ln_1.weight": None})
            (repo / "utils/metrics.py").write_text("# synthetic metric source\n")
            (repo / "data/dataset_registry.py").write_text("DATASET_REGISTRY = " + repr({"pems08": REGISTRY}))
            (repo / "data/unified_dataset.py").write_text(SOURCE)
            mutable = ["data/stage1/08.npy", "data/stage2/08fine-tuning.npy", "checkpoint/08/best_model_dict.pth", "utils/metrics.py"]
            assets, files = [], []
            for path in sorted(repo.rglob("*")):
                if not path.is_file():
                    continue
                name = str(path.relative_to(repo))
                content = path.read_bytes()
                if name in mutable:
                    sha = hashlib.sha256(content).hexdigest()
                    assets.append({"path": name, "size": len(content), "sha256": sha})
                    content = ("version https://git-lfs.github.com/spec/v1\noid sha256:" + sha + "\nsize " + str(len(content)) + "\n").encode()
                files.append({"path": name, "size": len(content), "sha": blob_sha(content), "mode": "100644"})
            (repo / ".source_snapshot.json").write_text(json.dumps({"repository": "LLM-UTP/LLM-UTP", "commit": commit, "files": files}))
            (code / "config/sources.lock.json").write_text(json.dumps({"reference_commit": commit, "reference_files": files, "lfs_files": assets}))
            before = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in mutable}
            report = project / "results/fixture"
            self.assertEqual(audit.run(project, code, report), 3)
            result = json.loads((report / "pems08_audit.json").read_text())
            self.assertFalse(result["inference_performed"])
            self.assertTrue(result["arrays"]["stage_comparison"]["channel_0_identical"])
            self.assertEqual(json.loads((report / "preflight_progress.json").read_text())["completed_checks"], 5)
            self.assertTrue(all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == sha for name, sha in before.items()))
            (repo / mutable[0]).write_bytes(b"bad payload")
            self.assertEqual(audit.run(project, code, project / "results/failed"), 2)


if __name__ == "__main__":
    unittest.main()
