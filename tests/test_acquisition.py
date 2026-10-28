"""Scientific failure modes and HTTP behavior, using no real traffic data."""
import functools
import hashlib
import http.server
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("acquire", ROOT / "scripts" / "acquire.py")
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


class Handler(http.server.BaseHTTPRequestHandler):
    payload = b"traffic-test-object-" * 200
    observed_ranges = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        data = self.payload
        range_header = self.headers.get("Range")
        self.observed_ranges.append(range_header)
        if self.path == "/corrupt":
            data = b"x" * len(data)
        if self.path == "/range" and range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
            data = data[start:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "object.bin"
        self.asset = {"path": "object.bin", "size": len(Handler.payload),
                      "sha256": hashlib.sha256(Handler.payload).hexdigest()}

    def locations(self, route):
        return lambda: [(self.base + route, {})]

    def test_resumes_partial_at_exact_byte(self):
        partial = self.path.with_suffix(".bin.part")
        partial.write_bytes(Handler.payload[:117])
        acquire.download_verified(self.asset, self.path, self.locations("/range"), retries=1)
        self.assertEqual(self.path.read_bytes(), Handler.payload)
        self.assertEqual(Handler.observed_ranges[-1], "bytes=117-")

    def test_range_ignored_restarts_without_appending(self):
        self.path.with_suffix(".bin.part").write_bytes(Handler.payload[:99])
        acquire.download_verified(self.asset, self.path, self.locations("/ignores-range"), retries=1)
        self.assertEqual(self.path.read_bytes(), Handler.payload)

    def test_wrong_digest_cannot_replace_previous_file(self):
        self.path.write_bytes(b"preserve existing bytes")
        with self.assertRaises(RuntimeError):
            acquire.download_verified(self.asset, self.path, self.locations("/corrupt"), retries=1)
        self.assertEqual(self.path.read_bytes(), b"preserve existing bytes")
        self.assertFalse(self.path.with_suffix(".bin.part").exists())

    def test_verified_existing_avoids_network(self):
        self.path.write_bytes(Handler.payload)
        def no_network():
            raise AssertionError("should not contact network")
        result = acquire.download_verified(self.asset, self.path, no_network)
        self.assertEqual(result["status"], "VERIFIED_EXISTING")

    def test_git_blob_digest_checks_small_hf_files(self):
        self.path.write_bytes(Handler.payload)
        asset = {"path": "config.json", "size": len(Handler.payload),
                 "git_blob_sha1": hashlib.sha1(("blob %d\0" % len(Handler.payload)).encode() + Handler.payload).hexdigest()}
        self.assertIsNotNone(acquire.verify(self.path, asset))
        self.path.write_bytes(b"q" * len(Handler.payload))
        self.assertIsNone(acquire.verify(self.path, asset))


class ArrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "values.npy"

    def test_zero_negative_and_missing_are_separate(self):
        import numpy as np
        data = np.ones((2, 16, 2), dtype="float32")
        data[0, 0, 0], data[0, 1, 0], data[0, 2, 0] = 0, -2, np.nan
        np.save(self.path, data)
        row = acquire.audit_array(self.path, 2)
        self.assertEqual((row["zero_elements"], row["negative_elements"], row["nonfinite_elements"]), (1, 1, 1))
        self.assertIn("NOT_VERIFIED", row["timestamp_status"])

    def test_pointer_is_not_a_dataset(self):
        self.path.write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "Not a complete"):
            acquire.audit_array(self.path, 170)

    def test_time_node_transposition_is_not_silently_fixed(self):
        import numpy as np
        np.save(self.path, np.ones((16, 2, 2)))
        with self.assertRaisesRegex(ValueError, "layout"):
            acquire.audit_array(self.path, 2)


class FullAuditTests(unittest.TestCase):
    def test_offline_full_inventory_reports_duplicate_without_training(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, work, report = root / "project", root / "work", root / "report"
            commit = "1" * 40
            repo = work / "reference" / "snapshots" / commit
            (project / "config").mkdir(parents=True)
            repo.mkdir(parents=True)
            report.mkdir()
            lfs, bodies = [], {}
            for index, (_, (first, second, nodes, _)) in enumerate(acquire.PUBLIC.items()):
                for stage, filename in (("stage1", first), ("stage2", second)):
                    path = repo / "data" / stage / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value = 12 if stage == "stage1" and filename in ("bike.npy", "taxi.npy") else index + (20 if stage == "stage2" else 1)
                    np.save(path, np.full((nodes, 16, 2), value, dtype="float32"))
                    content = path.read_bytes()
                    relative = str(path.relative_to(repo))
                    asset = {"path": relative, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                    lfs.append(asset)
                    bodies[relative] = content
                    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:%s\nsize %d\n" % (asset["sha256"], asset["size"]))
            files = [{"path": str(path.relative_to(repo)), "size": path.stat().st_size,
                      "sha": acquire.blob_sha(path.read_bytes()), "mode": "100644"}
                     for path in repo.rglob("*") if path.is_file()]
            marker = {"repository": "LLM-UTP/LLM-UTP", "commit": commit, "files": files}
            (repo / ".source_snapshot.json").write_text(json.dumps(marker))
            origin = "https://github.com/LLM-UTP/LLM-UTP.git"
            for relative, content in bodies.items():
                (repo / relative).write_bytes(content)
            modeldir = work / "models" / "gpt2" / "fixture-revision"
            modeldir.mkdir(parents=True)
            (modeldir / "config.json").write_text("{}")
            fp = acquire.fingerprint(modeldir / "config.json")
            lock = {"reference_repository": origin, "reference_commit": commit, "reference_files": files, "lfs_files": lfs,
                    "gpt2": {"revision": "fixture-revision", "files": [{"path": "config.json", **fp}]}}
            (project / "config" / "sources.lock.json").write_text(json.dumps(lock))
            argv = ["acquire.py", "--project", str(project), "--work", str(work), "--report", str(report), "--audit-only"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(acquire, "request_json", side_effect=AssertionError("offline")):
                result = acquire.main()
            audit = json.loads((report / "acquisition_audit.json").read_text())
            self.assertEqual(result, 3)
            self.assertEqual(len(audit["datasets"]), 16)
            self.assertEqual(audit["download_status"], "ALL_REQUESTED_ASSETS_VERIFIED")
            self.assertEqual(audit["duplicate_domains"][0]["datasets"], ["nyc_bike", "nyc_taxi"])
            self.assertFalse(audit["training_started"])
            self.assertEqual(audit["scientific_status"], "REPRODUCTION_NOT_ESTABLISHED")
            self.assertTrue(all(not x["eligible_as_full_training_domain"] for x in audit["restricted_datasets"]))


if __name__ == "__main__":
    unittest.main()
