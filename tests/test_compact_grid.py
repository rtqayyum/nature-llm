"""Fixed compact matrix, one-shot array submission and artifact collection."""
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import compact_baselines as pilot
import compact_grid as grid
import compact_grid_submit as submitter
import shared_baselines as base
from ectt_common import read, write, digest, finish, completed, artifact_manifest


class MatrixTests(unittest.TestCase):
    def test_exact_96_record_64_gpu_matrix_and_unchanged_pilot(self):
        matrix = grid.plan(ROOT)
        self.assertEqual((len(matrix['records']), len(matrix['tasks'])), (96, 64))
        self.assertEqual(matrix['counts'], {'GPU_FIT': 64, 'PILOT_REUSE': 2, 'DETERMINISTIC_FALLBACK': 30})
        self.assertEqual(len({r['key'] for r in matrix['records']}), 96)
        self.assertEqual({(r['dataset'], r['budget'], r['seed']) for r in matrix['records'] if r['kind'] == 'PILOT_REUSE'},
                         {('pems08', '0.05', 11), ('bj', '0.05', 11)})
        self.assertTrue(all(float(t['budget']) > 0 for t in matrix['tasks']))
        self.assertFalse(any(t['dataset'] in ('bj', 'hz') and t['budget'] == '0.01' for t in matrix['tasks']))
        self.assertEqual(pilot.identity(ROOT), grid.config(ROOT)[0]['pilot_identity'])


class GridTests(unittest.TestCase):
    def setUp(self):
        from test_compact_baselines import CompactIntegrationTests
        self.owner = CompactIntegrationTests(); self.owner.setUp()
        self.addCleanup(self.owner.doCleanups)
        self.project, self.code = self.owner.project, self.owner.code
        self.report = self.project/'results/slurm_jobs/777/report'
        self.tiny = self.owner.synthetic_recipe()
        pilot_report = self.project/'results/slurm_jobs/620634/report'
        with mock.patch.object(pilot, 'config', return_value=self.tiny), redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.run(self.project, self.code, pilot_report, device='cpu'), 0)
        write(pilot_report.parent/'job_status.json', {'stage': pilot.STAGE, 'execution_state': 'COMPLETED', 'shell_exit_code': 0})
        runtime = read(pilot_report/'compact_runtime.json')
        write(pilot_report/'compact_runtime.json', dict(runtime, allocated_cpus=5, visible_gpus=1, torch='2.4.1+cu118', synthetic_fixture=True))
        self.recipe = read(self.code/'config/compact_grid_v1.json')
        self.recipe.update(pilot_identity=pilot.identity(self.code), pilot_recipe_sha256=digest(self.code/'config/compact_local_v1.json'))
        write(self.code/'config/compact_grid_v1.json', self.recipe)
        self.matrix = grid.plan(self.code)
        self.addCleanup(mock.patch.stopall)

    def fake_scheduler(self, arguments, **kwargs):
        if arguments[0] == 'sbatch':
            report = '--job-name=ectt-compact-report' in arguments
            return subprocess.CompletedProcess(arguments, 0, '88002\n' if report else '88001\n', '')
        if arguments[:3] == ['squeue', '--noheader', '--format=%A']:
            return subprocess.CompletedProcess(arguments, 0, '', '')
        return self.owner.fixture_owner.fake_slurm(arguments, **kwargs)

    def submit(self, retry=False, scheduler=None):
        import slurm_submit
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(submitter.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(submitter.subprocess, 'run', side_effect=scheduler or self.fake_scheduler) as calls, redirect_stdout(io.StringIO()):
            result = submitter.submit(self.project, self.code, retry=retry)
        return result, calls

    def test_foundation_requires_verified_completed_gpu_pilot(self):
        self.assertTrue(grid.foundation(self.project, self.code, full=True)['full_artifact_verification'])
        case = Path(grid.foundation(self.project, self.code)['pilot_store'])/'cases/bj/0.05/11/result.json'
        case.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            grid.foundation(self.project, self.code, full=True)

    def test_one_command_submits_64_task_array_and_afterany_cpu_collector(self):
        result, calls = self.submit()
        self.assertEqual(result, 0)
        sbatch = [c for c in calls.call_args_list if c.args[0][0] == 'sbatch']
        self.assertEqual(len(sbatch), 2)
        gpu, cpu = (c.args[0] for c in sbatch)
        self.assertIn('--array=0-63%4', gpu); self.assertIn('--gres=gpu:1', gpu)
        self.assertIn('--cpus-per-task=5', gpu); self.assertIn('--partition=gpuq', gpu)
        self.assertIn('--dependency=afterany:88001', cpu)
        self.assertFalse(any(x.startswith('--gres') for x in cpu))
        self.assertIn('--cpus-per-task=5', cpu)
        self.assertTrue(any(x.startswith('--mail-type=') and 'ARRAY_TASKS' in x for x in gpu))
        for index, task in enumerate(self.matrix['tasks']):
            child = read(self.project/'results/slurm_jobs'/('88001_'+str(index))/'submission.json')
            self.assertEqual(child['key'], task['key']); self.assertEqual(child['array_job_id'], '88001')
        state = read(self.project/'work/LAST_COMPACT_GRID.json')
        self.assertEqual((state['array_job_id'], state['report_job_id']), ('88001', '88002'))
        self.assertEqual((self.project/'work/LAST_JOB_ID').read_text().strip(), '88001')

    def test_repeated_command_does_not_duplicate_array_or_collector(self):
        self.submit(); _, calls = self.submit()
        self.assertFalse(any(c.args[0][0] == 'sbatch' for c in calls.call_args_list))
        def next_batch(arguments, **kwargs):
            if arguments[0] == 'sbatch':
                number = '88102' if '--job-name=ectt-compact-report' in arguments else '88101'
                return subprocess.CompletedProcess(arguments, 0, number+'\n', '')
            return self.fake_scheduler(arguments, **kwargs)
        _, calls = self.submit(retry=True, scheduler=next_batch)
        self.assertEqual(sum(c.args[0][0] == 'sbatch' for c in calls.call_args_list), 2)
        journal = read(self.project/'work/LAST_COMPACT_GRID.json')
        self.assertEqual((journal['array_job_id'], journal['report_job_id']), ('88101', '88102'))
        self.assertEqual(journal['earlier_submissions'][0]['array_job_id'], '88001')

    def test_collector_submission_failure_retries_only_collector(self):
        def fail_report(arguments, **kwargs):
            if arguments[0] == 'sbatch' and '--job-name=ectt-compact-report' in arguments:
                raise subprocess.CalledProcessError(1, arguments, stderr='synthetic scheduler rejection')
            return self.fake_scheduler(arguments, **kwargs)
        with self.assertRaises(subprocess.CalledProcessError):
            self.submit(scheduler=fail_report)
        _, calls = self.submit()
        commands = [c.args[0] for c in calls.call_args_list if c.args[0][0] == 'sbatch']
        self.assertEqual(len(commands), 1); self.assertIn('--dependency=afterany:88001', commands[0])

    def test_missing_mail_and_existing_study_block_submission(self):
        with mock.patch.dict(os.environ, {'TRAFFIC_MAIL_USER': ''}):
            with self.assertRaisesRegex(ValueError, 'require saved email'):
                self.submit()
        write(self.project/'results/slurm_jobs/555/submission.json', {'stage': base.STAGE})
        def active(arguments, **kwargs):
            if arguments[:3] == ['squeue', '--noheader', '--format=%A']:
                return subprocess.CompletedProcess(arguments, 0, '555\n', '')
            return self.fake_scheduler(arguments, **kwargs)
        with self.assertRaisesRegex(ValueError, 'still active'):
            self.submit(scheduler=active)

    def test_real_task_entrypoint_preserves_pilot_and_baseline_and_reuses_case(self):
        evidence = grid.foundation(self.project, self.code, full=True)
        before = {key: artifact_manifest(Path(evidence[key])) for key in ('pilot_store',)}
        baseline = Path(evidence['baseline']['store']); baseline_before = artifact_manifest(baseline)
        with mock.patch.object(grid, 'config', return_value=(self.recipe, self.tiny)), redirect_stdout(io.StringIO()):
            self.assertEqual(grid.run_task(self.project, self.code, self.report, 0, device='cpu'), 0)
        output = read(self.report/'compact_result.json')
        self.assertEqual(output['task'], self.matrix['tasks'][0]); self.assertEqual(output['new_optimizer_steps'], 20)
        self.assertFalse(output['case_result']['development_only']); self.assertFalse(output['development_only'])
        self.assertEqual(before['pilot_store'], artifact_manifest(Path(evidence['pilot_store'])))
        self.assertEqual(baseline_before, artifact_manifest(baseline))
        second = self.project/'results/slurm_jobs/778/report'
        with mock.patch.object(grid, 'config', return_value=(self.recipe, self.tiny)), redirect_stdout(io.StringIO()):
            self.assertEqual(grid.run_task(self.project, self.code, second, 0, device='cpu'), 0)
        self.assertEqual(read(second/'compact_result.json')['new_optimizer_steps'], 0)

    def make_completed_synthetic_tasks(self, array_id='99000'):
        """Collector-only fixtures: no model training claims from these arrays."""
        evidence = grid.foundation(self.project, self.code, full=True)
        write(self.project/'results/slurm_jobs'/array_id/'submission.json', {'stage': grid.STAGE,
              'grid_identity': self.matrix['identity'], 'array_tasks': self.matrix['tasks']})
        for task in self.matrix['tasks']:
            store = grid.case_path(self.project, self.code, task); store.mkdir(parents=True)
            prior = Path(evidence['baseline']['store'])/'cases'/task['dataset']/task['budget']
            with np.load(prior/'forecasts.npz', allow_pickle=False) as old:
                p = old['persistence']
                np.savez_compressed(store/'forecasts.npz', origins=old['origins'], truth=old['truth'],
                    persistence=p, ridge=p, compact=p, support_selected_local=p)
            m = read(prior/'result.json')['metrics']['persistence']
            record = {**task, 'metrics': {name: m for name in ('persistence', 'ridge', 'compact', 'support_selected_local')},
                      'selection': {'selected_family': 'persistence'}, 'recovery_probe_passed': True,
                      'posttraining_invariance_passed': 4, 'synthetic_collector_fixture': True}
            shutil.copyfile(prior/'access_ledger.json', store/'access_ledger.json')
            write(store/'result.json', record); finish(store, self.matrix['identity'])
            child = self.project/'results/slurm_jobs'/(array_id+'_'+str(task['task_index']))
            write(child/'job_status.json', {'execution_state': 'COMPLETED', 'shell_exit_code': 0})
            write(child/'report/compact_result.json', {'identity': self.matrix['identity'], 'case_result': record})

    def test_collector_verifies_all_96_records_and_explicit_failed_cases(self):
        self.make_completed_synthetic_tasks()
        self.assertEqual(grid.collect(self.project, self.code, self.report, '99000'), 0)
        result = read(self.report/'compact_grid_result.json')
        self.assertEqual(result['completed_cases'], 96); self.assertFalse(result['failed_cases'])
        self.assertEqual(sum(r['kind'] == 'PILOT_REUSE' for r in result['records'].values()), 2)
        zero = result['records']['pems03/0.0/11']
        self.assertFalse(zero['independent_seed_replication']); self.assertEqual(zero['selected_family'], 'persistence')
        self.assertEqual(zero['forecast_columns']['compact'], 'persistence')
        rows = (self.report/'compact_grid_results.csv').read_text().splitlines()
        self.assertEqual(len(rows), 385)
        child = self.project/'results/slurm_jobs/99000_0/job_status.json'
        write(child, {'execution_state': 'FAILED', 'shell_exit_code': 2})
        self.assertEqual(grid.collect(self.project, self.code, self.project/'results/slurm_jobs/779/report', '99000'), 2)
        failed = read(self.project/'results/slurm_jobs/779/report/compact_grid_result.json')
        self.assertEqual(failed['completed_cases'], 95)
        self.assertIn(self.matrix['tasks'][0]['key'], failed['failed_cases'])

    def test_bootstrap_array_check_and_final_report_status(self):
        import bootstrap_hpc
        import slurm_check
        import slurm_job
        with mock.patch.object(bootstrap_hpc, 'resolve_source', return_value=('a'*40, [])), mock.patch.object(bootstrap_hpc, 'install_snapshot', return_value=self.code), mock.patch.object(bootstrap_hpc.subprocess, 'call', return_value=0) as call, redirect_stdout(io.StringIO()):
            bootstrap_hpc.main(['--project', str(self.project), '--stage', grid.STAGE, '--submit', '--retry-grid'])
        self.assertEqual(call.call_args.args[0][-3:], ['--stage', grid.STAGE, '--retry-grid'])
        self.submit()
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['Check', '--project', str(self.project), '88001']), mock.patch.object(slurm_check, 'command', return_value={'stdout': '', 'stderr': '', 'exit_code': 0}), redirect_stdout(output):
            slurm_check.main()
        self.assertIn('tasks=64', output.getvalue()); self.assertIn('REPORT_JOB_ID=88002', output.getvalue())
        self.assertNotIn('ASSET WORKERS', output.getvalue())
        write(self.report/'compact_grid_result.json', {'completed_cases': 96, 'scientific_status': 'COMPACT_LOCAL_GRID_COMPLETE', 'training_performed': False})
        with mock.patch.dict(os.environ, {'TRAFFIC_CODE_DIR': str(self.code), 'SLURM_JOB_ID': '777', 'TRAFFIC_STAGE': grid.REPORT_STAGE}), redirect_stdout(io.StringIO()):
            slurm_job.main(['finalize', '--project', str(self.project), '--shell-exit', '0', '--raw-exit', '0'])
        status = read(self.report.parent/'job_status.json')
        self.assertEqual(status['benchmark_experiments_completed'], 96); self.assertFalse(status['training_performed'])
        self.assertEqual(status['S01_independent_review'], 'PENDING')


if __name__ == '__main__':
    unittest.main()
