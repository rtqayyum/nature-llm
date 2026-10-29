"""Synthetic Slurm commands and process metadata; no cluster jobs are submitted."""
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
import slurm_submit as submit
import slurm_job as job
from slurm_check import cpu_utilization


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "nature-llm"
        self.code = self.project / "work/code_versions/fixture"
        self.code.mkdir(parents=True)
        (self.code / "Batch").write_bytes((ROOT / "Batch").read_bytes())

    def fake_slurm(self, args, **kwargs):
        if args[0] == "scontrol":
            return subprocess.CompletedProcess(args, 0, "PartitionName=" + args[3] + " MaxTime=" + ("01:00:00" if args[3] == "shortq" else "12:00:00"), "")
        if args[0] == "sbatch":
            return subprocess.CompletedProcess(args, 0, "73001\n", "")
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, "PENDING\n", "")
        raise AssertionError(args)

    def submit_fixture(self, stage="acquire"):
        with mock.patch.object(submit, "ROOT", self.project), mock.patch.object(submit.shutil, "which", return_value="/fake/sbatch"), \
             mock.patch.object(submit.subprocess, "run", side_effect=self.fake_slurm) as execute:
            result = submit.main(["--project", str(self.project), "--code", str(self.code), "--stage", stage])
        return result, execute

    def test_submit_caps_walltime_and_records_synthetic_job(self):
        result, execute = self.submit_fixture()
        self.assertEqual(result, 0)
        record = json.loads((self.project / "results/slurm_jobs/73001/submission.json").read_text())
        self.assertEqual((record["time_limit"], record["cpus"], record["gpus"]), ("12:00:00", 4, 0))
        sbatch = next(call for call in execute.call_args_list if call.args[0][0] == "sbatch")
        self.assertIn("--time=12:00:00", sbatch.args[0])
        self.assertEqual(sbatch.kwargs["env"]["TRAFFIC_CODE_DIR"], str(self.code))
        for name in ("logs", "results", "outputs", "figures", "tmp", ".cache"):
            self.assertTrue((self.project / name).is_dir())

    def test_pending_project_job_is_not_submitted_again(self):
        self.submit_fixture()
        result, execute = self.submit_fixture()
        self.assertEqual(result, 0)
        self.assertFalse(any(call.args[0][0] == "sbatch" for call in execute.call_args_list))

    def test_pems08_uses_observed_short_queue_and_one_cpu(self):
        result, execute = self.submit_fixture("pems08-audit")
        record = json.loads((self.project / "results/slurm_jobs/73001/submission.json").read_text())
        self.assertEqual((record["stage"], record["partition"], record["cpus"], record["time_limit"]), ("pems08-audit", "shortq", 1, "00:30:00"))
        sbatch = next(call for call in execute.call_args_list if call.args[0][0] == "sbatch")
        self.assertIn("--job-name=nutp-p08-preflight", sbatch.args[0])
        self.assertEqual(sbatch.kwargs["env"]["TRAFFIC_STAGE"], "pems08-audit")

    def test_first_inference_is_cpu_only_with_eight_threads(self):
        _, execute = self.submit_fixture("pems08-infer")
        record = json.loads((self.project / "results/slurm_jobs/73001/submission.json").read_text())
        self.assertEqual((record["stage"], record["partition"], record["cpus"], record["gpus"]), ("pems08-infer", "longq", 8, 0))
        call = next(call for call in execute.call_args_list if call.args[0][0] == "sbatch")
        self.assertIn("--job-name=nutp-p08-infer-cpu", call.args[0])
        self.assertFalse(any("gres" in part for part in call.args[0]))

    def test_metro_inference_uses_short_cpu_job(self):
        _, execute = self.submit_fixture("beijing-infer")
        record = json.loads((self.project / "results/slurm_jobs/73001/submission.json").read_text())
        self.assertEqual((record["partition"], record["cpus"], record["gpus"], record["time_limit"]), ("shortq", 4, 0, "00:30:00"))
        call = next(call for call in execute.call_args_list if call.args[0][0] == "sbatch")
        self.assertIn("--job-name=nutp-bj-infer-cpu", call.args[0])
        self.assertFalse(any("gres" in part for part in call.args[0]))

    def test_public8_allocates_sixteen_cpus_without_gpu(self):
        _, execute = self.submit_fixture("public8-infer")
        record = json.loads((self.project / "results/slurm_jobs/73001/submission.json").read_text())
        self.assertEqual((record["partition"], record["cpus"], record["gpus"]), ("longq", 16, 0))
        call = next(call for call in execute.call_args_list if call.args[0][0] == "sbatch")
        self.assertIn("--mem=32G", call.args[0])
        self.assertIn("--job-name=nutp-public8-infer", call.args[0])
        self.assertEqual(call.kwargs["env"]["TRAFFIC_STAGE"], "public8-infer")

    def test_custom_entrypoint_is_never_overwritten(self):
        (self.project / "Batch").write_text("user content\n")
        with mock.patch.object(submit, "ROOT", self.project), self.assertRaisesRegex(ValueError, "preserved"):
            submit.prepare(self.project, self.code)
        self.assertEqual((self.project / "Batch").read_text(), "user content\n")

    def test_process_validation_rejects_other_project_and_slurm(self):
        process = Path(self.temp.name) / "proc/123"
        process.mkdir(parents=True)
        command = ["/python", str(self.code / "scripts/pipeline.py"), "--project", str(self.code), "--work", str(self.project / "work")]
        (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in command) + b"\0")
        (process / "environ").write_bytes(b"PATH=/bin\0")
        self.assertTrue(submit.validate_process(123, self.project, command, process.parent))
        self.assertFalse(submit.validate_process(123, self.project, command + ["other"], process.parent))
        (process / "environ").write_bytes(b"SLURM_JOB_ID=999\0")
        self.assertFalse(submit.validate_process(123, self.project, command, process.parent))

    def test_handoff_never_signals_unvalidated_process(self):
        directory = self.project / "work/reports/old"
        directory.mkdir(parents=True)
        (directory / "launch.json").write_text(json.dumps({"pid": 54321, "command": ["other-project"]}))
        with mock.patch.object(submit, "validate_process", return_value=False), mock.patch.object(submit.os, "killpg") as kill:
            self.assertEqual(submit.handoff(self.project), [])
            kill.assert_not_called()

    def test_review_and_genuine_failure_remain_distinct(self):
        with mock.patch.dict(os.environ, {"TRAFFIC_CODE_DIR": str(self.code), "SLURM_JOB_ID": "73001"}):
            job.main(["finalize", "--project", str(self.project), "--raw-exit", "3", "--shell-exit", "0"])
            status_path = self.project / "results/slurm_jobs/73001/job_status.json"
            self.assertEqual(json.loads(status_path.read_text())["execution_state"], "COMPLETED_WITH_REVIEW")
            job.main(["finalize", "--project", str(self.project), "--raw-exit", "2", "--shell-exit", "2"])
            self.assertEqual(json.loads(status_path.read_text())["execution_state"], "FAILED")

    def test_review_completes_srun_step_but_preserves_pipeline_code(self):
        with mock.patch.dict(os.environ, {"TRAFFIC_CODE_DIR": str(self.code), "SLURM_JOB_ID": "73001",
                "TRAFFIC_WORK_ROOT": str(self.project / "work"), "TRAFFIC_ENV_DIR": str(self.project / ".venv")}), \
             mock.patch("pipeline.main", return_value=3), mock.patch.object(sys, "argv", ["fixture"]):
            result = job.main(["run", "--project", str(self.project)])
        self.assertEqual(result, 0)
        path = self.project / "results/slurm_jobs/73001/pipeline_result.json"
        self.assertEqual(json.loads(path.read_text())["pipeline_exit_code"], 3)

    def test_cpu_utilization_accounts_for_allocated_cpus(self):
        row = "73001|job|COMPLETED|0:0|00:01:40|100|4|00:03:20|1G|None\n"
        self.assertEqual(cpu_utilization(row, "73001"), 50.0)
        completed = "605458|job|COMPLETED|0:0|00:00:52|52|4|00:18.812||None\n"
        self.assertEqual(cpu_utilization(completed, "605458"), 9.04)

    def test_cpu_batch_requests_one_process_no_gpu_and_local_environment(self):
        batch = (ROOT / "Batch").read_text()
        self.assertIn("#SBATCH --ntasks=1", batch)
        self.assertIn("#SBATCH --cpus-per-task=4", batch)
        self.assertNotIn("#SBATCH --gres", batch)
        self.assertIn('TRAFFIC_ENV_DIR="$ROOT/.venv"', batch)
        self.assertIn("module load anaconda3", batch)
        self.assertNotIn("/cm/shared/apps", batch)
        subprocess.run(["bash", "-n", str(ROOT / "Batch")], check=True)


if __name__ == "__main__":
    unittest.main()
