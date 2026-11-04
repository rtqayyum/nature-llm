"""Compact recovery, information boundaries and synthetic Slurm integration."""
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import compact_baselines as compact
import compact_training as training
import shared_baselines as base
from compact_model import CompactTrafficModel
from ectt_common import read, write, digest, completed, artifact_manifest
from ectt_data import fit_scaler
from ectt_model import invariance_checks


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.raw = np.stack([np.arange(130.) % 17, np.arange(130.) % 23])
        self.scaler = fit_scaler(self.raw, [0, 70])
        self.architecture = {'width': 8, 'layers': 1, 'heads': 2}
        self.recipe = read(ROOT/'config/compact_local_v1.json')
        self.recipe.update(updates=6, origin_batch=2, checkpoint_every_steps=2,
                           recovery_probe_updates=6, recovery_probe_split=3)
        torch.set_num_threads(1)

    def fit(self, store, raw=None, stop_after=None):
        return training.fit(self.raw if raw is None else raw, [12, 70], self.scaler,
            self.architecture, self.recipe, 11, self.store/store, 'synthetic', 'cpu', lambda **v: None,
            stop_after=stop_after)

    def test_trained_model_is_origin_independent_and_node_equivariant(self):
        model, _ = self.fit('fit')
        x, _ = training.data_batch(self.raw, [70, 71], self.scaler, 'cpu')
        self.assertEqual(len(invariance_checks(model, x)), 4)
        with self.assertRaisesRegex(ValueError, '12 x 3'):
            model(torch.zeros(2, 2, 11, 3))

    def test_fit_ignores_all_values_after_fitting_support(self):
        a, first = self.fit('first')
        changed = self.raw.copy(); changed[:, 70:] += 100000
        b, second = self.fit('second', changed)
        self.assertTrue(training.equal_state(a.state_dict(), b.state_dict()))
        self.assertEqual(first['losses'], second['losses'])

    def test_forecast_never_reads_its_current_or_future_truth(self):
        model, _ = self.fit('fit')
        raw = self.raw
        class Guard:
            def __getitem__(self, key):
                node, rows = key
                if not isinstance(rows, slice) or rows.stop > 80:
                    raise AssertionError('Prediction accessed current/future truth')
                return raw[node, rows]
        observed = training.predict(model, Guard(), [80, 81], self.scaler, 8, 'cpu')
        changed = raw.copy(); changed[:, 80:] = 999999
        np.testing.assert_array_equal(observed, training.predict(model, changed, [80, 81], self.scaler, 1, 'cpu'))

    def test_exact_serialized_optimizer_rng_cursor_recovery(self):
        record = training.recovery_probe(self.raw, [12, 70], self.scaler, self.architecture,
            self.recipe, 11, self.store/'probe', 'synthetic', 'cpu', lambda **v: None)
        self.assertTrue(record['passed']); self.assertTrue(all(record['checks'].values()))
        original = artifact_manifest(self.store/'probe')
        again = training.recovery_probe(self.raw, [12, 70], self.scaler, self.architecture,
            self.recipe, 11, self.store/'probe', 'synthetic', 'cpu', lambda **v: None)
        self.assertEqual(again, record); self.assertEqual(original, artifact_manifest(self.store/'probe'))

    def test_fresh_process_resume_matches_uninterrupted_model_and_adam(self):
        self.fit('whole'); self.fit('split', stop_after=3)
        program = '''import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, sys.argv[1])
from ectt_common import read
from ectt_data import fit_scaler
from compact_training import fit
torch.set_num_threads(1)
raw=np.stack([np.arange(130.)%17,np.arange(130.)%23])
fit(raw,[12,70],fit_scaler(raw,[0,70]),{'width':8,'layers':1,'heads':2},read(Path(sys.argv[2])),11,Path(sys.argv[3]),'synthetic','cpu',lambda **v:None)
'''
        write(self.store/'recipe.json', self.recipe)
        subprocess.run([sys.executable, '-c', program, str(ROOT/'scripts'), str(self.store/'recipe.json'),
                        str(self.store/'split')], check=True, capture_output=True, text=True)
        a = training.load_checkpoint(self.store/'whole', training.fingerprint_from_pointer(self.store/'whole'))
        b = training.load_checkpoint(self.store/'split', training.fingerprint_from_pointer(self.store/'split'))
        for key in ('model', 'optimizer', 'rng', 'step', 'losses'):
            self.assertTrue(training.equal_state(a[key], b[key]), key)

    def test_completed_fit_reuses_without_updates_and_corrupt_checkpoint_fails(self):
        self.fit('fit'); before = artifact_manifest(self.store/'fit')
        _, info = self.fit('fit')
        self.assertEqual(info['new_optimizer_steps'], 0); self.assertEqual(before, artifact_manifest(self.store/'fit'))
        pointer = read(self.store/'fit/latest.json'); path = self.store/'fit'/pointer['file']
        path.write_bytes(b'corrupted checkpoint')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.fit('fit')

    def test_changed_fit_contract_cannot_resume(self):
        self.fit('fit', stop_after=3)
        self.recipe['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            self.fit('fit')

    def test_recipe_and_dependency_identity_preserve_old_baseline(self):
        recipe = compact.config(ROOT)
        self.assertEqual(base.identity(ROOT), recipe['baseline_identity'])
        self.assertEqual((recipe['resources']['cpus'], recipe['resources']['gpus']), (5, 1))
        self.assertEqual(recipe['datasets'], ['pems08', 'bj'])
        self.assertEqual(len(compact.identity(ROOT)), 64)

    def test_largest_architecture_full_node_count_invariance(self):
        torch.manual_seed(11)
        model = CompactTrafficModel(width=128, layers=4, heads=4)
        # Activate the output so the check exercises the whole encoder.
        torch.nn.init.normal_(model.head[-1].weight, std=.01)
        inputs = torch.randn(2, 358, 12, 3); inputs[..., 2] = 1
        self.assertEqual(len(invariance_checks(model, inputs, atol=1e-5, rtol=1e-4)), 4)


class CompactIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_shared_baselines import BaselineTests
        self.fixture_owner = BaselineTests(); self.fixture_owner.setUp()
        self.addCleanup(self.fixture_owner.doCleanups)
        self.c, self.r = self.fixture_owner.fixture()
        self.project, self.code = self.fixture_owner.project, self.fixture_owner.code
        self.report = self.project/'results/slurm_jobs/90002/report'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.project/'results/slurm_jobs/620562/report'), 0)
        write(self.project/'results/slurm_jobs/620562/job_status.json',
              {'stage': base.STAGE, 'execution_state': 'COMPLETED', 'shell_exit_code': 0})
        self.recipe = read(self.code/'config/compact_local_v1.json')
        self.recipe.update(scientific_contract_sha256=digest(self.code/'config/ectt_protocol_v1.json'),
                           baseline_identity=base.identity(self.code))
        write(self.code/'config/compact_local_v1.json', self.recipe)
        self.addCleanup(mock.patch.stopall)

    def synthetic_recipe(self):
        recipe = copy.deepcopy(self.recipe)
        recipe.update(updates=4, origin_batch=2, recovery_probe_updates=4, recovery_probe_split=2,
                      checkpoint_every_steps=2, configurations=[{'width': 8, 'layers': 1, 'heads': 2},
                                                               {'width': 16, 'layers': 2, 'heads': 2}])
        return recipe

    def test_real_entrypoint_common_truth_metrics_case_reuse_and_preserved_baselines(self):
        evidence = compact.foundation(self.project, self.code, full=True)
        old_store = Path(evidence['store']); before = artifact_manifest(old_store)
        with mock.patch.object(compact, 'config', return_value=self.synthetic_recipe()), redirect_stdout(io.StringIO()):
            self.assertEqual(compact.run(self.project, self.code, self.report, device='cpu'), 0)
        self.assertEqual(before, artifact_manifest(old_store))
        result = read(self.report/'compact_result.json')
        self.assertEqual(result['completed_cases'], 2); self.assertEqual(result['new_optimizer_steps'], 40)
        self.assertFalse(result['method_evaluated']); self.assertTrue(result['development_only'])
        for slug, row in result['cases'].items():
            case = self.project/'outputs/ectt/compact_local'/compact.identity(self.code)/'cases'/slug/'0.05/11'
            self.assertTrue(completed(case, compact.identity(self.code)))
            with np.load(case/'forecasts.npz', allow_pickle=False) as saved, np.load(old_store/'cases'/slug/'0.05/forecasts.npz', allow_pickle=False) as old:
                np.testing.assert_array_equal(saved['truth'], old['truth']); np.testing.assert_array_equal(saved['ridge'], old['ridge'])
                np.testing.assert_array_equal(saved['origins'], np.arange(*self.c['datasets'][slug]['test_targets']))
                self.assertAlmostEqual(row['metrics']['compact']['MAE'], base.independent_scores(saved['compact'], saved['truth'])['MAE'])
            self.assertEqual(read(case/'access_ledger.json')['verification_labels_used_for_fitting_or_selection'], 0)
        first_case = self.project/'outputs/ectt/compact_local'/compact.identity(self.code)/'cases/pems08/0.05/11'
        saved_case = artifact_manifest(first_case)
        second_report = self.project/'results/slurm_jobs/90003/report'
        with mock.patch.object(compact, 'config', return_value=self.synthetic_recipe()), redirect_stdout(io.StringIO()):
            self.assertEqual(compact.run(self.project, self.code, second_report, device='cpu'), 0)
        self.assertEqual(read(second_report/'compact_result.json')['new_optimizer_steps'], 0)
        self.assertEqual(saved_case, artifact_manifest(first_case))

    def test_foundation_fails_on_corrupt_case_or_incomplete_previous_job(self):
        path = self.project/'results/slurm_jobs/620562/job_status.json'; state = read(path)
        write(path, dict(state, execution_state='FAILED'))
        with self.assertRaisesRegex(ValueError, 'not completed'):
            compact.foundation(self.project, self.code)
        write(path, state)
        case = Path(compact.foundation(self.project, self.code)['store'])/'cases/pems08/0.05/fit.json'
        case.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            compact.foundation(self.project, self.code, full=True)

    def test_worker_requires_real_gpu_and_preserves_failed_run_evidence(self):
        with mock.patch.object(compact, 'config', return_value=self.synthetic_recipe()), mock.patch.object(torch.cuda, 'is_available', return_value=False), redirect_stdout(io.StringIO()):
            self.assertEqual(compact.run(self.project, self.code, self.report), 2)
        self.assertEqual(read(self.report/'compact_progress.json')['phase'], 'FAILED')
        self.assertTrue((self.report/'failure_traceback.txt').is_file())
        self.assertFalse((self.report/'compact_result.json').exists())

    def test_missing_email_active_study_and_partition_limit(self):
        import slurm_submit
        args = ['--project', str(self.project), '--code', str(self.code), '--stage', compact.STAGE]
        with mock.patch.dict(os.environ, {'TRAFFIC_MAIL_USER': ''}), mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=self.fixture_owner.fake_slurm) as calls, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'require saved email'):
                slurm_submit.main(args)
            self.assertFalse(any(call.args[0][0] == 'sbatch' for call in calls.call_args_list))
        write(self.project/'results/slurm_jobs/555/submission.json', {'stage': base.STAGE})
        def active(arguments, **kwargs):
            if arguments[:3] == ['squeue', '--noheader', '--format=%A']:
                return subprocess.CompletedProcess(arguments, 0, '555\n', '')
            return self.fixture_owner.fake_slurm(arguments, **kwargs)
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=active) as calls, redirect_stdout(io.StringIO()):
            self.assertEqual(slurm_submit.main(args), 0)
            self.assertFalse(any(call.args[0][0] == 'sbatch' for call in calls.call_args_list))
        with mock.patch.object(slurm_submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'PartitionName=gpuq MaxTime=01:00:00', '')):
            partition, limit, _ = slurm_submit.allocation(compact.STAGE)
        self.assertEqual((partition, limit), ('gpuq', 3600))

    def test_gpu_submission_and_bootstrap_monitor_final_status(self):
        import bootstrap_hpc
        import slurm_submit
        import slurm_check
        import slurm_job
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=self.fixture_owner.fake_slurm), redirect_stdout(io.StringIO()):
            self.assertEqual(slurm_submit.main(['--project', str(self.project), '--code', str(self.code), '--stage', compact.STAGE]), 0)
        submission = read(self.project/'results/slurm_jobs/90001/submission.json')
        self.assertEqual((submission['cpus'], submission['gpus'], submission['partition'], submission['time_limit']), (5, 1, 'gpuq', '02:00:00'))
        self.assertIn('--gres=gpu:1', submission['command'])
        with mock.patch.object(bootstrap_hpc, 'resolve_source', return_value=('a'*40, [])), mock.patch.object(bootstrap_hpc, 'install_snapshot', return_value=self.code), mock.patch.object(bootstrap_hpc.subprocess, 'call', return_value=0) as call, redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap_hpc.main(['--project', str(self.project), '--stage', compact.STAGE, '--submit']), 0)
        self.assertEqual(call.call_args.args[0][-2:], ['--stage', compact.STAGE])
        write(self.report/'compact_result.json', {'completed_cases': 2, 'new_optimizer_steps': 2024, 'optimizer_steps': 2024,
              'training_performed': True, 'scientific_status': 'COMPACT_DEVELOPMENT_PILOT_ONLY'})
        with mock.patch.dict(os.environ, {'TRAFFIC_CODE_DIR': str(self.code), 'SLURM_JOB_ID': '90002', 'TRAFFIC_STAGE': compact.STAGE}), redirect_stdout(io.StringIO()):
            slurm_job.main(['finalize', '--project', str(self.project), '--shell-exit', '0', '--raw-exit', '0'])
        status = read(self.report.parent/'job_status.json')
        self.assertEqual(status['benchmark_experiments_completed'], 2); self.assertEqual(status['new_optimizer_steps'], 2024)
        write(self.report.parent/'submission.json', dict(submission, job_id='90002'))
        write(self.report/'compact_progress.json', {'completed_cases': 2, 'total_cases': 2, 'phase': 'COMPLETED'})
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['Check', '--project', str(self.project), '90002']), mock.patch.object(slurm_check, 'command', return_value={'stdout': '', 'stderr': '', 'exit_code': 0}), redirect_stdout(output):
            slurm_check.main()
        self.assertIn('COMPACT', output.getvalue()); self.assertNotIn('ASSET WORKERS', output.getvalue())


if __name__ == '__main__':
    unittest.main()
