"""Source updates, corruption, archive boundaries and locks without Git."""
import fcntl
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bootstrap_hpc as bootstrap


def source_archive(entries):
    output = io.BytesIO()
    files = []
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries.items():
            archive.writestr("source-fixture/" + name, data)
            files.append({"path": name, "size": len(data), "sha": bootstrap.blob_sha(data), "mode": "100644"})
    return output.getvalue(), files


class HPCSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = "a" * 40
        self.archive, self.files = source_archive({"run.sh": b"#!/bin/bash\nexit 0\n", "config/test.json": b"{}\n"})

    def install(self, destination=None, archive=None):
        return bootstrap.install_snapshot("rtqayyum/nature-llm", self.commit, self.files,
                                          destination or self.root / "version", archive_bytes=archive or self.archive)

    def test_install_verifies_archive_and_reuses_same_source_without_network(self):
        directory = self.install()
        with mock.patch.object(bootstrap.urllib.request, "urlopen", side_effect=AssertionError("network not needed")):
            self.install(directory)
        marker = json.loads((directory / bootstrap.MARKER).read_text())
        self.assertEqual(marker["commit"], self.commit)
        self.assertEqual((directory / "config/test.json").read_bytes(), b"{}\n")

    def test_corrupt_archive_cannot_install_code(self):
        bad, _ = source_archive({"run.sh": b"#!/bin/bash\nexit 1\n", "config/test.json": b"{}\n"})
        with self.assertRaisesRegex(ValueError, "digest"):
            self.install(archive=bad)
        self.assertFalse((self.root / "version").exists())

    def test_traversal_archive_cannot_write_outside_snapshot(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("source/../../escaped", b"x")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.install(archive=output.getvalue())
        self.assertFalse((self.root / "escaped").exists())

    def test_archive_symlink_is_rejected(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            member = zipfile.ZipInfo("source/run.sh")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, "/etc/passwd")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.install(archive=output.getvalue())

    def test_source_edits_are_preserved_and_update_stops(self):
        directory = self.install()
        (directory / "run.sh").write_text("local edits\n")
        with self.assertRaisesRegex(ValueError, "source file"):
            self.install(directory)
        self.assertEqual((directory / "run.sh").read_text(), "local edits\n")

    def test_bootstrap_runs_without_git_and_keeps_stable_work_root(self):
        project = self.root / "nature-llm"
        project.mkdir()
        (project / "old_checkout.txt").write_text("preserve")
        with mock.patch.dict(os.environ, {"PATH": ""}, clear=True), \
             mock.patch.object(bootstrap, "resolve_source", return_value=(self.commit, self.files)), \
             mock.patch.object(bootstrap.urllib.request, "urlopen", return_value=io.BytesIO(self.archive)), \
             mock.patch.object(bootstrap.subprocess, "call", return_value=0) as launch:
            result = bootstrap.main(["--project", str(project)])
        self.assertEqual(result, 0)
        self.assertEqual(launch.call_args.kwargs["env"]["TRAFFIC_WORK_ROOT"], str(project / "work"))
        self.assertEqual(launch.call_args.args[0][0], "bash")
        self.assertEqual((project / "old_checkout.txt").read_text(), "preserve")
        self.assertTrue((project / "work/CURRENT_CODE.json").exists())

    def test_running_acquisition_blocks_code_update_before_network(self):
        project = self.root / "nature-llm"
        work = project / "work"
        work.mkdir(parents=True)
        with (work / "acquisition.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(bootstrap, "resolve_source") as fetch:
                with self.assertRaisesRegex(SystemExit, "Acquisition is running"):
                    bootstrap.main(["--project", str(project)])
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
