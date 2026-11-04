"""Synthetic common-protocol execution; no real HPC evidence or email."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import shared_baselines as base
from ectt_common import read, write, digest, finish
from ectt_statistics import confidence_sequence, decision, paired_difference
import slurm_submit


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)/'nature-llm'
        self.code = self.project/'work/code_versions/synthetic-fixture'
        self.code.mkdir(parents=True)
        for name in ('config', 'scripts', 'research'):
            shutil.copytree(ROOT/name, self.code/name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(ROOT/'Batch', self.code/'Batch')
        self.report = self.project/'results/slurm_jobs/90000/report'
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ, {'TRAFFIC_MAIL_USER': 'synthetic@example.invalid'}).start()
        mock.patch('job_notifications.queued_receipt', return_value={'state': 'SYNTHETIC_NO_EMAIL'}).start()

    def fixture(self):
        c = read(self.code/'config/ectt_protocol_v1.json')
        lock = read(self.code/'config/sources.lock.json')
        for index, (slug, p) in enumerate(c['datasets'].items()):
            total = 1500 if slug in ('bj', 'hz') else 6000
            cutoff = int(.8*total)
            t = np.arange(total, dtype=float)
            value = np.empty((2, total, 2))
            value[..., 0] = np.stack([(t % 47)+index, ((t+8) % 47)+index])
            value[..., 1] = -999
            path = self.project/'work/reference/snapshots'/lock['reference_commit']/p['path']
            path.parent.mkdir(parents=True, exist_ok=True); np.save(path, value)
            p.update(shape=[2, total, 2], container_sha256=digest(path), source_train_raw=[0, int(.6*total)],
                     source_train_targets=[12, int(.6*total)], source_validation_targets=[int(.6*total), int(.7*total)],
                     test_targets=[cutoff, total])
            for q, b in p['budgets'].items():
                size = int(float(q)*cutoff); start = cutoff-size; middle = start+size//2
                b.update(raw_rows=size, support_raw=[start, middle], verification_raw=[middle, cutoff],
                         support_target_rows=[min(start+12, middle), middle],
                         verification_target_rows=[min(max(start+12, middle), cutoff), cutoff],
                         eligible_for_learned_adaptation=size//2 >= 24 and cutoff-middle >= 12)
        write(self.code/'config/ectt_protocol_v1.json', c)
        e = read(self.code/'config/ectt_execution_v1.json')
        e['scientific_contract_sha256'] = digest(self.code/'config/ectt_protocol_v1.json')
        e['statistical_simulation'].update(streams=30, origins=64)
        write(self.code/'config/ectt_execution_v1.json', e)
        r = read(self.code/'config/shared_baselines_v1.json')
        r['scientific_contract_sha256'] = digest(self.code/'config/ectt_protocol_v1.json')
        write(self.code/'config/shared_baselines_v1.json', r)
        f = r['foundation']; source = self.project/'work/code_versions'/f['pilot_commit']
        source.mkdir()
        entries = []
        import hashlib
        for name in base.PILOT_DEPENDENCIES:
            path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.code/name, path)
            data = path.read_bytes()
            entries.append({'path': name, 'sha': hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
        write(source/'.source_snapshot.json', {'repository': 'rtqayyum/nature-llm', 'commit': f['pilot_commit'], 'files': entries})
        data_store = self.project/'outputs/ectt/admission'/f['data_identity']/'ectt-preflight'
        write(data_store/'result.json', {'synthetic_fixture': True}); finish(data_store, f['data_identity'])
        pilot_store = self.project/'outputs/ectt/admission'/f['pilot_identity']/'ectt-gpu-pilot'
        gates = [{'case': name, 'passed': True, 'maximum_absolute_difference': 0.}
                 for name in ('single_origin', 'origin_permutation', 'later_companion_mutation', 'node_permutation')]
        pilot = {'stage': 'ectt-gpu-pilot', 'execution_kind': 'REAL_GPU_SUPPORT_SMOKE',
                 'allocated_cpus': 5, 'visible_gpus': 1, 'test_evaluation_performed': False,
                 'new_method_objective_evaluated': False,
                 'pretraining': gates, 'posttraining': gates, 'strict_reconstruction_pass': True,
                 'fit': {'completed': True, 'optimizer_steps': 6}, 'synthetic_fixture': True}
        write(pilot_store/'result.json', pilot); finish(pilot_store, f['pilot_identity'])
        job = self.project/'results/slurm_jobs'/f['pilot_job']
        write(job/'job_status.json', {'stage': 'ectt-gpu-pilot', 'execution_state': 'COMPLETED', 'shell_exit_code': 0})
        write(job/'report/ectt_result.json', pilot)
        return c, r

    def raw(self, c, slug='pems08'):
        lock = read(self.code/'config/sources.lock.json')
        path = self.project/'work/reference/snapshots'/lock['reference_commit']/c['datasets'][slug]['path']
        return np.load(path, allow_pickle=False)[..., 0]

    def test_foundation_requires_matching_pilot_and_frozen_dependencies(self):
        self.fixture()
        self.assertTrue(base.foundation(self.project, self.code, full=True)['full_artifact_verification'])
        path = self.code/'scripts/ectt_model.py'; path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'dependency'):
            base.foundation(self.project, self.code)

    def test_fitting_and_selection_ignore_verification_and_test_values(self):
        c, r = self.fixture(); p = c['datasets']['pems08']; raw = self.raw(c)
        first = base.fit_case(raw, p, '0.05', r)
        changed = raw.copy(); changed[:, p['budgets']['0.05']['support_raw'][1]:] += 100000
        self.assertEqual(first, base.fit_case(changed, p, '0.05', r))
        self.assertEqual(first['models']['ridge']['scaler']['fit_rows'], p['budgets']['0.05']['support_raw'])

    def test_prediction_never_reads_current_or_future_target(self):
        c, r = self.fixture(); p = c['datasets']['pems08']; raw = self.raw(c)
        model = base.fit_case(raw, p, '0.05', r)['models']['ridge']
        origin = p['test_targets'][0]
        class Guard:
            def __getitem__(self, key):
                node, row = key
                if not isinstance(row, slice) or row.stop > origin:
                    raise AssertionError('Prediction read its target or future row')
                return raw[node, row]
        expected = base.predict(raw, [origin, origin+1], model, 64)
        np.testing.assert_array_equal(expected, base.predict(Guard(), [origin, origin+1], model, 64))
        changed = raw.copy(); changed[:, origin:] += 100000
        np.testing.assert_array_equal(expected, base.predict(changed, [origin, origin+1], model, 1))

    def test_zero_budget_and_infeasible_cases_keep_persistence(self):
        c, r = self.fixture(); raw = self.raw(c, 'bj'); p = c['datasets']['bj']
        for q, expected in [('0.0', 'ZERO_BUDGET_FALLBACK'), ('0.01', 'INFEASIBLE_BUDGET_FALLBACK')]:
            result = base.fit_case(raw, p, q, r)
            self.assertEqual(result['state'], expected); self.assertFalse(result['ridge_fitted'])
            self.assertEqual(result['supervised_rows_accessed'], [])

    def test_degenerate_inner_support_is_reported_without_heldout_scaling(self):
        c, r = self.fixture(); p = c['datasets']['pems08']; raw = self.raw(c)
        start, end = p['budgets']['0.05']['support_raw']
        split = start+int(.7*(end-start)); raw[:, start:split] = 3
        result = base.fit_case(raw, p, '0.05', r)
        self.assertEqual(result['state'], 'RIDGE_UNAVAILABLE_DEGENERATE_INNER_SUPPORT')
        self.assertFalse(result['ridge_fitted'])

    def test_ridge_equation_matches_direct_regularized_least_squares(self):
        from ectt_data import fit_scaler
        raw = np.stack([np.arange(100.) % 17, np.arange(100.) % 19])
        scaler = fit_scaler(raw, [0, 60]); bounds = [12, 60]
        a, y, _ = base.design(raw, list(range(*bounds)), scaler, fitting=True)
        a, y = a.reshape(-1, 37), y.reshape(-1)
        penalty = np.eye(37)*.01; penalty[0, 0] = 0
        expected = np.linalg.solve(a.T@a/len(y)+penalty, a.T@y/len(y))
        np.testing.assert_allclose(base.solve(base.moments(raw, bounds, scaler, 7), .01), expected, atol=1e-11)

    def test_metrics_include_zero_truth_and_independent_reduction(self):
        p = np.array([[1., 1.]]); y = np.array([[0., 1.]])
        result = base.scores(p, y)
        self.assertEqual(result['MAE'], .5); self.assertEqual(result['WMAPE_percent'], 100.)
        self.assertEqual(base.independent_scores(p, y)['WMAPE_percent'], 100.)
        self.assertIsNone(base.scores(p, np.zeros_like(y))['WMAPE_percent'])

    def test_full_32_cases_resume_and_corruption_detection(self):
        self.fixture()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.report, stop_after=3), 75)
            self.assertEqual(read(self.report/'baseline_progress.json')['completed_cases'], 3)
            self.assertEqual(base.run(self.project, self.code, self.report), 0)
        result = read(self.report/'baseline_result.json')
        self.assertEqual(result['completed_cases'], 32); self.assertEqual(result['ridge_cases_fitted'], 22)
        self.assertEqual(len((self.report/'baseline_results.csv').read_text().splitlines()), 97)
        self.assertFalse(read(self.report/'internal_statistical_review.json')['independent_review_complete'])
        ident = base.identity(self.code); root = self.project/'outputs/ectt/shared_baselines'/ident
        case = root/'cases/pems03/0.0'; marker = (case/'COMPLETE.json').read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.report), 0)
        replay = read(self.report/'baseline_result.json')
        self.assertEqual(replay['new_ridge_fits'], 0); self.assertFalse(replay['training_performed'])
        self.assertEqual(replay['cases'], result['cases']); self.assertEqual((case/'COMPLETE.json').read_bytes(), marker)
        (case/'forecasts.npz').write_bytes(b'corrupt')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.report), 2)
        self.assertEqual((case/'COMPLETE.json').read_bytes(), marker)

    def fake_slurm(self, args, **kwargs):
        if args[0] == 'scontrol': return subprocess.CompletedProcess(args, 0, 'PartitionName='+args[3]+' MaxTime=12:00:00', '')
        if args[0] == 'sbatch': return subprocess.CompletedProcess(args, 0, '90001\n', '')
        if args[0] == 'squeue': return subprocess.CompletedProcess(args, 0, '', '')
        raise AssertionError(args)

    def test_submission_cpu_only_and_notifications_required(self):
        self.fixture()
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=self.fake_slurm):
            slurm_submit.main(['--project', str(self.project), '--code', str(self.code), '--stage', base.STAGE])
        record = read(self.project/'results/slurm_jobs/90001/submission.json')
        self.assertEqual((record['cpus'], record['gpus']), (5, 0))
        self.assertFalse(any('--gres' in value or '--array' in value for value in record['command']))

    def test_new_stage_refuses_missing_email_and_active_earlier_study(self):
        self.fixture()
        with mock.patch.dict(os.environ, {'TRAFFIC_MAIL_USER': ''}), mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=self.fake_slurm) as calls:
            with self.assertRaisesRegex(ValueError, 'require saved email'):
                slurm_submit.main(['--project', str(self.project), '--code', str(self.code), '--stage', base.STAGE])
            self.assertFalse(any(call.args[0][0] == 'sbatch' for call in calls.call_args_list))
        write(self.project/'results/slurm_jobs/555/submission.json', {'stage': 'ectt-gpu-pilot'})
        def active(args, **kwargs):
            if args[:3] == ['squeue', '--noheader', '--format=%A']:
                return subprocess.CompletedProcess(args, 0, '555\n', '')
            return self.fake_slurm(args, **kwargs)
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=active) as calls:
            slurm_submit.main(['--project', str(self.project), '--code', str(self.code), '--stage', base.STAGE])
            self.assertFalse(any(call.args[0][0] == 'sbatch' for call in calls.call_args_list))

    def test_failed_pilot_and_changed_data_stop_work_preserving_prior_cases(self):
        c, r = self.fixture()
        status = self.project/'results/slurm_jobs'/r['foundation']['pilot_job']/'job_status.json'
        original = status.read_bytes(); item = read(status); item['execution_state'] = 'FAILED'; write(status, item)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.report), 2)
        status.write_bytes(original)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.report, stop_after=1), 75)
        first = self.project/'outputs/ectt/shared_baselines'/base.identity(self.code)/'cases/pems03/0.0/COMPLETE.json'
        preserved = first.read_bytes()
        lock = read(self.code/'config/sources.lock.json')
        path = self.project/'work/reference/snapshots'/lock['reference_commit']/c['datasets']['pems03']['path']
        path.write_bytes(b'corrupt canonical input')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(base.run(self.project, self.code, self.report), 2)
        self.assertEqual(first.read_bytes(), preserved)

    def test_monitor_displays_real_case_progress(self):
        import slurm_check
        job = self.project/'results/slurm_jobs/90000'
        write(job/'submission.json', {'project': str(self.project), 'job_id': '90000', 'job_name': 'ectt-local-base', 'stage': base.STAGE, 'partition': 'shortq'})
        write(job/'report/baseline_progress.json', {'phase': 'BASELINES', 'completed_cases': 7, 'total_cases': 32})
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['Check', '--project', str(self.project), '90000']), mock.patch.object(slurm_check, 'command', return_value={'stdout':'', 'stderr':'', 'exit_code':0}), redirect_stdout(output):
            slurm_check.main()
        self.assertIn('completed_cases', output.getvalue()); self.assertNotIn('ASSET WORKERS', output.getvalue())

    def test_bootstrap_forwards_new_stage_without_git(self):
        import bootstrap_hpc
        with mock.patch.object(bootstrap_hpc, 'resolve_source', return_value=('a'*40, [])), mock.patch.object(bootstrap_hpc, 'install_snapshot', return_value=self.code), mock.patch.object(bootstrap_hpc.subprocess, 'call', return_value=0) as call, redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap_hpc.main(['--project', str(self.project), '--stage', base.STAGE, '--submit']), 0)
        self.assertEqual(call.call_args.args[0][-2:], ['--stage', base.STAGE])

    def test_short_queue_limit_falls_back_to_longq(self):
        def partitions(args, **kwargs):
            limit = '00:30:00' if args[3] == 'shortq' else '23:00:00'
            return subprocess.CompletedProcess(args, 0, 'PartitionName='+args[3]+' MaxTime='+limit, '')
        with mock.patch.object(slurm_submit.subprocess, 'run', side_effect=partitions):
            partition, seconds, _ = slurm_submit.allocation(base.STAGE)
        self.assertEqual((partition, seconds), ('longq', 3600))

    def test_final_job_status_reports_baseline_cases_and_pending_S01(self):
        import slurm_job
        write(self.report/'baseline_result.json', {'completed_cases': 32, 'new_ridge_fits': 22,
                                                  'training_performed': True, 'scientific_status': 'LOCAL_BASELINE_FOUNDATION_ONLY'})
        with mock.patch.dict(os.environ, {'TRAFFIC_CODE_DIR': str(self.code), 'SLURM_JOB_ID': '90000',
                                         'TRAFFIC_STAGE': base.STAGE}), redirect_stdout(io.StringIO()):
            self.assertEqual(slurm_job.main(['finalize', '--project', str(self.project), '--shell-exit', '0', '--raw-exit', '0']), 0)
        status = read(self.report.parent/'job_status.json')
        self.assertEqual(status['benchmark_experiments_completed'], 32)
        self.assertEqual(status['new_ridge_fits'], 22)
        self.assertFalse(status['method_evaluated']); self.assertEqual(status['S01_independent_review'], 'PENDING')


class VerifierAuditTests(unittest.TestCase):
    def test_finite_parameter_and_extreme_width_guards(self):
        for rho in (np.nan, np.inf, -1, 0):
            with self.assertRaises(ValueError): confidence_sequence([0], [1], rho=rho)
        for minimum in (0, 1.5, True):
            with self.assertRaises(ValueError): decision([0], [1], minimum=minimum)
        with self.assertRaises(ValueError): paired_difference(np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), np.nan)
        self.assertTrue(np.isfinite(confidence_sequence([0], [1e153])[1]).all())
        with self.assertRaises(ValueError): confidence_sequence([0], [1e200])


if __name__ == '__main__':
    unittest.main()
