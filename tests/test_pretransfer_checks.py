"""Development-only artifact audit and fresh-process recovery on synthetic CPU."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import compact_baselines as pilot
import compact_grid as grid
import pretransfer_checks as checks
import qwen_recovery as recovery
from ectt_common import read, write, digest, finish, artifact_manifest


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_production_recipe_preserves_completed_grid_and_qwen_inputs(self):
        self.assertEqual(checks.config(ROOT)['grid_identity'], grid.identity(ROOT))
        self.assertEqual(len(checks.identity(ROOT)), 64)
        self.assertEqual(checks.config(ROOT)['resources']['gpus'], 1)

    def test_cached_only_weight_verification_preserves_marker_and_rejects_corruption(self):
        contract = {'model': {'backbone': 'synthetic/test', 'revision': 'a'*40}}
        directory = self.root/'work/models/qwen2_5_1_5b'/('a'*40); directory.mkdir(parents=True)
        files = []
        for name, data in [('config.json', b'{}'), ('model.safetensors', b'synthetic-not-model-weights')]:
            (directory/name).write_bytes(data)
            files.append({'name': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha1': None})
        write(directory/'PINNED_FILES.json', {'repository': 'synthetic/test', 'revision': 'a'*40, 'files': files})
        before = artifact_manifest(directory)
        self.assertEqual(recovery.cached_weights(self.root, contract)[0], directory)
        self.assertEqual(before, artifact_manifest(directory))
        (directory/'model.safetensors').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'wrong-size'):
            recovery.cached_weights(self.root, contract)

    def launch(self, phase):
        program = '''import sys
from pathlib import Path
import torch
sys.path.insert(0,sys.argv[1])
from qwen_recovery import segment
torch.set_num_threads(1);torch.manual_seed(19)
model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
x,y=torch.rand(2,3),torch.rand(2,1)
phase=sys.argv[3]
segment(model,x,y,Path(sys.argv[2]),'SYNTHETIC_RECOVERY',6,.0001,phase,3,lambda **v:None)
sys.exit(75 if phase=='interrupt' else 0)
'''
        arm = self.root/('full' if phase == 'full' else 'resume')
        return subprocess.run([sys.executable, '-c', program, str(ROOT/'scripts'), str(arm), phase],
                              capture_output=True, text=True)

    def test_fresh_process_model_adam_rng_losses_cursor_and_predictions_match(self):
        a = self.launch('full'); b = self.launch('interrupt'); c = self.launch('resume')
        self.assertEqual((a.returncode, b.returncode, c.returncode), (0, 75, 0), a.stderr+b.stderr+c.stderr)
        result = recovery.compare(self.root/'full', self.root/'resume', 'SYNTHETIC_RECOVERY', 6)
        self.assertTrue(all(result.values()))
        first = recovery.verify_receipt(self.root/'resume', 'interrupt', 'SYNTHETIC_RECOVERY')
        last = recovery.verify_receipt(self.root/'resume', 'resume', 'SYNTHETIC_RECOVERY')
        self.assertEqual((first['optimizer_steps'], last['resumed_from_step'], last['new_optimizer_steps']), (3, 3, 3))
        old_receipt = (self.root/'resume/resume_receipt.json').read_bytes()
        # Reconstructing a finished segment must preserve its original restart cursor.
        replay = self.launch('resume'); self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(old_receipt, (self.root/'resume/resume_receipt.json').read_bytes())

    def test_corrupt_checkpoint_and_changed_optimizer_are_rejected(self):
        for phase in ('full', 'interrupt', 'resume'):
            result = self.launch(phase); self.assertIn(result.returncode, (0, 75), result.stderr)
        payload, path = recovery.load_latest(self.root/'resume', 'SYNTHETIC_RECOVERY')
        next(iter(payload['optimizer']['state'].values()))['exp_avg'].add_(1)
        torch.save(payload, path)
        pointer = read(self.root/'resume/LATEST.json'); pointer['sha256'] = digest(path)
        write(self.root/'resume/LATEST.json', pointer)
        receipt = read(self.root/'resume/resume_receipt.json'); receipt['checkpoint_sha256'] = digest(path)
        write(self.root/'resume/resume_receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'exact recovery failed'):
            recovery.compare(self.root/'full', self.root/'resume', 'SYNTHETIC_RECOVERY', 6)
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            recovery.load_latest(self.root/'resume', 'SYNTHETIC_RECOVERY')

    def test_parent_drives_three_fresh_processes_and_reuses_completed_probe(self):
        store = self.root/'coordinator'; runtime = {'study': 'SYNTHETIC', 'device': 'SYNTHETIC_CPU'}
        ident = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
        program = '''import sys
from pathlib import Path
import torch
sys.path.insert(0,sys.argv[1])
from qwen_recovery import segment
from ectt_common import write
torch.set_num_threads(1);torch.manual_seed(19)
model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
x,y=torch.rand(2,3),torch.rand(2,1);phase=sys.argv[3];arm=Path(sys.argv[2])
segment(model,x,y,arm,sys.argv[4],6,.0001,phase,3,lambda **v:None)
rows=[{'case':n,'passed':True,'maximum_absolute_difference':0.} for n in ['single_origin','origin_permutation','later_companion_mutation','node_permutation']]
write(arm/(phase+'_admission.json'),{'pretraining':rows,'posttraining':rows,'test_labels_read_by_learner':0,'verification_labels_read_by_learner':0,'peak_gpu_allocated_bytes':0,'synthetic_fixture':True})
sys.exit(75 if phase=='interrupt' else 0)
'''
        original = subprocess.run
        def child(command, **kwargs):
            phase = command[command.index('--segment')+1]
            write(store/'runtime_contract.json', runtime)
            arm = store/('uninterrupted' if phase == 'full' else 'restarted')
            return original([sys.executable, '-c', program, str(ROOT/'scripts'), str(arm), phase, ident],
                            capture_output=True, text=True)
        with mock.patch.object(recovery, 'cached_weights', return_value=(self.root, 'SYNTHETIC_HASH')), mock.patch.object(recovery.subprocess, 'run', side_effect=child) as calls:
            result = recovery.run(self.root, ROOT, self.root, store, 'SYNTHETIC', lambda **v: None)
            self.assertEqual(calls.call_count, 3)
            again = recovery.run(self.root, ROOT, self.root, store, 'SYNTHETIC', lambda **v: None)
            self.assertEqual(calls.call_count, 3)
        self.assertTrue(result['passed']); self.assertEqual(result['new_optimizer_steps'], 12)
        self.assertEqual(again['new_optimizer_steps'], 0); self.assertTrue(again['reused'])


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        from test_compact_grid import GridTests
        self.owner = GridTests(); self.owner.setUp(); self.addCleanup(self.owner.doCleanups)
        self.project, self.code = self.owner.project, self.owner.code
        self.report = self.project/'results/slurm_jobs/91000/report'
        self.recipe = read(self.code/'config/pretransfer_checks_v1.json')
        self.recipe.update(grid_identity=grid.identity(self.code), expected_training_curves=45, loss_summary_window=2)
        write(self.code/'config/pretransfer_checks_v1.json', self.recipe)
        matrix = grid.plan(self.code)
        write(self.project/'results/slurm_jobs/620653/submission.json',
              {'stage': grid.STAGE, 'grid_identity': grid.identity(self.code), 'array_tasks': matrix['tasks']})
        write(self.project/'results/slurm_jobs/620656/job_status.json', {'stage': grid.REPORT_STAGE,
              'execution_state': 'COMPLETED', 'shell_exit_code': 0, 'scientific_status': 'COMPACT_LOCAL_GRID_COMPLETE'})
        write(self.project/'results/slurm_jobs/620656/report/compact_progress.json', {'completed_cases': 96, 'failed_cases': 0})
        self.tiny = self.owner.tiny
        self.addCleanup(mock.patch.stopall)

    def prepare_development(self):
        torch.set_num_threads(1)
        evidence = grid.foundation(self.project, self.code, full=True)
        c = self.owner.owner.c
        for task in grid.plan(self.code)['records']:
            if task['dataset'] not in ('pems08', 'bj') or task['kind'] != 'GPU_FIT':
                continue
            outer = grid.case_path(self.project, self.code, task); outer.mkdir(parents=True)
            raw = self.owner.owner.fixture_owner.raw(c, task['dataset'])
            recipe = dict(self.tiny, budget=task['budget'], seed=task['seed'])
            with redirect_stdout(io.StringIO()):
                result = pilot.save_case(raw, c['datasets'][task['dataset']], task['dataset'], recipe,
                    Path(evidence['baseline']['store'])/'cases'/task['dataset']/task['budget'],
                    outer/'kernel', grid.identity(self.code), 'cpu', lambda **v: None)
            write(outer/'result.json', dict(result, development_only=True, forecast_file='kernel/forecasts.npz',
                                           access_ledger_file='kernel/access_ledger.json'))
            finish(outer, grid.identity(self.code))

    def test_review_all_existing_development_curves_without_refitting_or_mutation(self):
        self.prepare_development()
        old = artifact_manifest(self.project/'outputs/ectt/compact_grid')
        output = self.project/'outputs/ectt/pretransfer_checks/synthetic/review'
        with mock.patch.object(grid, 'config', return_value=(self.owner.recipe, self.tiny)):
            result = checks.review_development(self.project, self.code, output, 'SYNTHETIC', lambda **v: None)
            again = checks.review_development(self.project, self.code, output, 'SYNTHETIC', lambda **v: None)
        self.assertEqual(result, again); self.assertEqual(result['reviewed_trained_cases'], 15)
        self.assertEqual(result['training_curves_reviewed'], 45); self.assertEqual(result['reviewed_fallback_records'], 9)
        self.assertEqual(result['new_optimizer_steps'], 0); self.assertFalse(result['new_test_evaluation_performed'])
        self.assertEqual(old, artifact_manifest(self.project/'outputs/ectt/compact_grid'))

    def test_incomplete_collector_and_support_selection_timing_stop_checks(self):
        status = self.project/'results/slurm_jobs/620656/job_status.json'
        original = read(status); write(status, dict(original, execution_state='FAILED'))
        with self.assertRaisesRegex(ValueError, 'incomplete/failed'):
            checks.foundation(self.project, self.code)
        write(status, original); self.prepare_development()
        task = next(r for r in grid.plan(self.code)['tasks'] if r['dataset'] == 'pems08')
        outer = grid.case_path(self.project, self.code, task); path = outer/'kernel/selection.json'
        data = read(path); data['inner_split'] += 1; write(path, data); finish(outer, grid.identity(self.code))
        with mock.patch.object(grid, 'config', return_value=(self.owner.recipe, self.tiny)):
            with self.assertRaisesRegex(ValueError, 'timing differs'):
                checks.review_development(self.project, self.code, self.project/'outputs/audit', 'SYNTHETIC', lambda **v: None)

    def test_one_job_coordinator_runs_review_and_propagates_qwen_failure(self):
        self.prepare_development()
        def partial_failure(*args):
            write(self.report/'qwen_segment_progress.json', {'cumulative_new_optimizer_steps': 3, 'synthetic_fixture': True})
            raise ValueError('synthetic Qwen mismatch')
        with mock.patch.object(grid, 'config', return_value=(self.owner.recipe, self.tiny)), mock.patch.object(recovery, 'run', side_effect=partial_failure), redirect_stdout(io.StringIO()):
            self.assertEqual(checks.run(self.project, self.code, self.report), 2)
        self.assertEqual(read(self.report/'transfer_check_progress.json')['phase'], 'FAILED')
        self.assertEqual(read(self.report/'transfer_check_progress.json')['new_optimizer_steps'], 3)
        self.assertTrue((self.report/'failure_traceback.txt').is_file())
        outcome = {'passed': True, 'engineering_optimizer_steps': 12, 'new_optimizer_steps': 12, 'synthetic_fixture': True}
        with mock.patch.object(grid, 'config', return_value=(self.owner.recipe, self.tiny)), mock.patch.object(recovery, 'run', return_value=outcome), redirect_stdout(io.StringIO()):
            self.assertEqual(checks.run(self.project, self.code, self.report), 0)
        result = read(self.report/'transfer_check_result.json')
        self.assertEqual(result['benchmark_experiments_completed'], 0); self.assertEqual(result['new_optimizer_steps'], 12)
        self.assertFalse(result['method_evaluated']); self.assertEqual(result['S01_independent_review'], 'PENDING')

    def test_real_worker_refuses_cpu_fallback(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'one visible GPU'):
                recovery.worker(self.project, self.code, self.report, self.project/'outputs/check', 'SYNTHETIC', 'full', 'x')

    def test_submission_bootstrap_monitor_and_final_status(self):
        import slurm_submit
        import bootstrap_hpc
        import slurm_check
        import slurm_job
        fixture = self.owner.owner.fixture_owner
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=fixture.fake_slurm), redirect_stdout(io.StringIO()):
            self.assertEqual(slurm_submit.main(['--project', str(self.project), '--code', str(self.code), '--stage', checks.STAGE]), 0)
        submission = read(self.project/'results/slurm_jobs/90001/submission.json')
        self.assertEqual((submission['cpus'], submission['gpus'], submission['partition'], submission['time_limit']), (5, 1, 'gpuq', '02:00:00'))
        self.assertFalse(any(x.startswith('--array') for x in submission['command']))
        self.assertIn('--gres=gpu:1', submission['command'])
        with mock.patch.object(bootstrap_hpc, 'resolve_source', return_value=('a'*40, [])), mock.patch.object(bootstrap_hpc, 'install_snapshot', return_value=self.code), mock.patch.object(bootstrap_hpc.subprocess, 'call', return_value=0) as call, redirect_stdout(io.StringIO()):
            bootstrap_hpc.main(['--project', str(self.project), '--stage', checks.STAGE, '--submit'])
        self.assertEqual(call.call_args.args[0][-2:], ['--stage', checks.STAGE])
        write(self.report/'transfer_check_result.json', {'scientific_status': 'PRETRANSFER_ENGINEERING_CHECKS_PASSED',
              'qwen_recovery_passed': True, 'engineering_optimizer_steps': 12, 'new_optimizer_steps': 12,
              'training_performed': True, 'development_trained_cases_reviewed': 15})
        with mock.patch.dict(os.environ, {'TRAFFIC_CODE_DIR': str(self.code), 'SLURM_JOB_ID': '91000', 'TRAFFIC_STAGE': checks.STAGE}), redirect_stdout(io.StringIO()):
            slurm_job.main(['finalize', '--project', str(self.project), '--shell-exit', '0', '--raw-exit', '0'])
        self.assertEqual(read(self.report.parent/'job_status.json')['new_optimizer_steps'], 12)
        write(self.report.parent/'submission.json', dict(submission, job_id='91000'))
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['Check', '--project', str(self.project), '91000']), mock.patch.object(slurm_check, 'command', return_value={'stdout': '', 'stderr': '', 'exit_code': 0}), redirect_stdout(output):
            slurm_check.main()
        self.assertIn('TRANSFER', output.getvalue()); self.assertNotIn('ASSET WORKERS', output.getvalue())


if __name__ == '__main__':
    unittest.main()
