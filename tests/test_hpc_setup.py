"""Exercise the missing-Git HPC setup without depending on local modules."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class HPCSetupTests(unittest.TestCase):
    def shell(self, body):
        return subprocess.run(["bash", "-c", body], capture_output=True, text=True)

    def test_existing_git_does_not_load_modules(self):
        result = self.shell('source "' + str(ROOT / "scripts/hpc_bootstrap.sh") + '"; '
                            'module() { echo UNEXPECTED_MODULE; return 1; }; traffic_require_git')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("UNEXPECTED_MODULE", result.stdout)

    def fixture_script(self, directory):
        # Hide the test host's installed Git/module-init files to represent gn01.
        source = (ROOT / "scripts/hpc_bootstrap.sh").read_text()
        source = source.replace("for git_path in /usr/bin/git /usr/local/bin/git /bin/git;", "for git_path in /nonexistent/test/git;")
        source = source.replace("for init_file in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash;",
                                "for init_file in /nonexistent/test/modules;")
        path = Path(directory) / "bootstrap.sh"
        path.write_text(source)
        return path

    def test_git_module_can_restore_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = self.fixture_script(directory)
            marker = Path(directory) / "git_available"
            result = self.shell('source "' + str(helper) + '"; '
                                'command() { if [[ "$1" == -v && "$2" == git ]]; then '
                                '[[ -e "' + str(marker) + '" ]]; else builtin command "$@"; fi; }; '
                                'module() { if [[ "$1" == load && "$2" == git ]]; then touch "' + str(marker) + '"; fi; }; '
                                'traffic_require_git')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Trying HPC module: git", result.stdout)
            self.assertTrue(marker.exists())

    def test_unavailable_git_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = self.fixture_script(directory)
            result = self.shell('source "' + str(helper) + '"; '
                                'command() { if [[ "$1" == -v && "$2" == git ]]; then return 1; '
                                'else builtin command "$@"; fi; }; traffic_require_git')
            self.assertEqual(result.returncode, 127)
            self.assertIn("Git is missing", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_direct_python_updater_fails_before_subprocess_when_git_missing(self):
        spec = importlib.util.spec_from_file_location("update", ROOT / "scripts/update.py")
        updater = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(updater)
        with mock.patch.object(updater.shutil, "which", return_value=None), mock.patch.object(updater.subprocess, "check_output") as execute:
            with self.assertRaisesRegex(SystemExit, "Git is not on PATH"):
                updater.main()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
