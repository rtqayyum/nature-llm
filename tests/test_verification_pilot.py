"""Exercise source-objective gradients, recovery and the reviewed scoring boundary."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'scripts'))
from ectt_common import digest, finish, read, write
from ectt_statistics import confidence_sequence, paired_difference
import slurm_submit
import transfer_train
import verification_pilot as pilot
import verification_pilot_score as scoring
import verification_training as training
from test_anchored_transfer_development import model


class VerificationPilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.contract = read(CODE/'config/ectt_protocol_v1.json')

    def test_margin_formula_and_memory_bounded_gradient(self):
        coefficients = self.contract['training_design']['mechanism_coefficients']
        verifier = self.contract['verifier']
        features = torch.linspace(-2., 5., 128).reshape(64, 2)
        reference = torch.full_like(features, 3.).double()
        truth = torch.full_like(features, 2.).double()
        weight = torch.tensor(.8, requires_grad=True)
        physical = features*weight
        loss = training.prefix_objective(physical, reference, truth, 1.5, 'verification_margin', coefficients, verifier)
        expected_grad, = torch.autograd.grad(loss, weight)
        d, b = paired_difference(reference.numpy(), physical.detach().clamp_min(0).numpy(), truth.numpy(), 1.5)
        _, upper = confidence_sequence(d, b, verifier['rho'], verifier['delta_per_stream'])
        tau = coefficients['smooth_hinge_temperature']
        expected = np.mean([tau*np.logaddexp(0, (upper[n-1]+coefficients['epsilon'])/tau)
                            for n in coefficients['prefixes']])
        self.assertAlmostEqual(float(loss), expected, places=12)
        batches = [list(range(i, i+2)) for i in range(0, 64, 2)]
        calls = []

        def predict(ids):
            calls.append(torch.is_grad_enabled())
            return features[ids]*weight

        def read_truth():
            self.assertEqual(calls, [True]*32)
            self.assertIsNone(weight.grad)
            return truth

        actual = training.prefix_backward(predict, batches, reference, read_truth, 1.5,
                                          'verification_margin', coefficients, verifier)
        self.assertAlmostEqual(actual, expected, places=12)
        torch.testing.assert_close(weight.grad, expected_grad)

    def test_real_source_trainer_recovers_each_objective(self):
        raw = {name: np.random.default_rng(i).uniform(1, 100, (3, 180)) for i, name in enumerate(('a', 'b'))}
        profiles = {name: {'source_train_raw': [0, 144], 'source_validation_targets': [144, 170]} for name in raw}
        recipe = read(CODE/'config/shared_transfer_v1.json')
        design = {**self.contract['training_design'], 'source_max_epochs': 1, 'target_updates': 4}
        compact = {**read(CODE/'config/compact_local_v1.json'), 'updates': 1,
                   'configurations': [{'width': 8, 'layers': 1, 'heads': 2}]}
        recipes = read(CODE/'config/shared_baselines_v1.json'), compact
        with tempfile.TemporaryDirectory() as directory:
            for arm in training.ARMS:
                task = {'family': 'B03', 'dataset': 'heldout', 'seed': 11, 'sources': list(raw), 'arm': arm}
                bank = transfer_train.episode_bank(task, {'datasets': profiles}, recipe, raw)
                args = (raw, task, bank, recipe, design, self.contract['verifier'], recipes, 'cpu')
                root = Path(directory)/arm
                full, _ = training.fit(model('persistence_residual'), *args, root/'full', arm, lambda **_: None)
                training.fit(model('persistence_residual'), *args, root/'resumed', arm, lambda **_: None, stop_after=1)
                training.fit(model('persistence_residual'), *args, root/'resumed', arm, lambda **_: None)
                self.assertTrue(transfer_train.equal_state(transfer_train.load(root/'full', arm),
                                                           transfer_train.load(root/'resumed', arm)))
                self.assertEqual(full['step'], 2)
                self.assertEqual(all(record[-1] == 0 for record in full['losses']), arm == 'suffix_mae')

    def test_scoring_checks_review_and_freezes_all_decisions_before_test(self):
        plan = pilot.matrix(CODE)
        contract = {**self.contract, 'datasets': {}, 'statistics': {
            **self.contract['statistics'], 'paired_bootstrap_replicates': 20}}
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            report = project/'report'
            report.mkdir()
            root = pilot.store_path(project, plan)
            collection = project/'results/slurm_jobs/14/report'
            write(collection.parent/'job_status.json', {'stage': pilot.REPORT, 'execution_state': 'COMPLETED', 'shell_exit_code': 0})
            write(collection/'transfer_result.json', {'identity': plan['identity'], 'failed': {},
                  'scientific_status': 'ECTT_PILOT_CANDIDATES_READY', 'diagnostic_cases_checked': 18})
            write(project/'work'/pilot.JOURNAL/(plan['identity']+'.json'), {'identity': plan['identity'],
                  'jobs': {'review': '11', 'probe': '12', 'fit': '13', 'report': '14'}})
            approval = pilot.approval_path(project, plan)
            review = {'status': 'PENDING_INDEPENDENT_REVIEW', 'independent_review_complete': False,
                      'pilot_identity': plan['identity'], 'reviewed_hashes': plan['reviewed_hashes'],
                      'reviewer': 'Synthetic test fixture', 'date': '2026-11-11', 'disposition': 'Fixture only'}
            write(approval, review)
            with self.assertRaisesRegex(ValueError, 'does not approve'):
                scoring.admission(project, CODE, plan)
            review.update(status='APPROVED', independent_review_complete=True)
            write(approval, review)
            for slug in plan['recipe']['development_targets']:
                contract['datasets'][slug] = {'shape': [2, 184, 2], 'container_sha256': slug,
                    'test_targets': [104, 184], 'budgets': {'0.05': {'support_raw': [0, 24],
                    'support_target_rows': [12, 24], 'verification_target_rows': [24, 104]}}}
            for task in plan['tasks']:
                case = root/'cases'/str(task['index'])
                (case/'target').mkdir(parents=True)
                (case/'best.pt').write_bytes(b'synthetic source artifact')
                (case/'target/fit.pt').write_bytes(b'synthetic residual artifact')
                residual = digest(case/'target/fit.pt')
                source = digest(case/'best.pt')
                write(case/'target/latest.json', {'file': 'fit.pt', 'sha256': residual})
                write(case/'result.json', {'task': task, 'scientific_status': 'ECTT_PILOT_FROZEN_CASE_COMPLETE',
                      'source': {'task': task, 'best_sha256': source, 'source_optimizer_steps': 7,
                                 'runtime': {'elapsed_seconds': 1.}},
                      'target': {'optimizer_steps': 200, 'runtime': {'elapsed_seconds': 1.}}})
                write(case/'frozen_candidates.json', {'task': task, 'budget': '0.05', 'support_raw': [0, 24],
                      'support_targets': [12, 24], 'unique_support_values': 48, 'support_scale': 1.,
                      'scaler': {'minimum': 1., 'maximum': 3.}, 'local_completion_sha256': task['dataset'],
                      'target_container_sha256': task['dataset'], 'verification_targets': [24, 104],
                      'test_targets': [104, 184], 'verification_history_union': [12, 103],
                      'test_history_union': [92, 183], 'verification_labels_used_for_scoring': 0,
                      'test_labels_used_for_scoring': 0, 'source_checkpoint_sha256': source,
                      'target_residual_sha256': residual})
                transfer = np.full((80, 2), {'suffix_mae': 12., 'prefix_mae': 10., 'verification_margin': 2.}[task['arm']])
                for name, start in (('verification_candidates', 24), ('test_candidates', 104)):
                    np.savez(case/(name+'.npz'), origins=np.arange(start, start+80),
                             local=np.full((80, 2), 10.), transfer=transfer)
                finish(case, plan['identity'])

            testcase = self
            accesses = []

            class TruthReader:
                def __getitem__(self, key):
                    rows = key[1]
                    accesses.append(rows)
                    if isinstance(rows, slice):
                        decisions, = (root/'evaluations').glob('*/decisions.json')
                        testcase.assertEqual(len(read(decisions)['decisions']), 18)
                        testcase.assertEqual((rows.start, rows.stop), (104, 184))
                        return np.full((2, 80), 2.)
                    testcase.assertTrue(24 <= rows < 104)
                    return np.full(2, 2.)

            with patch('transfer_worker.raw_arrays', return_value={name: TruthReader() for name in contract['datasets']}), \
                 patch('transfer_worker.raw_paths', return_value={}):
                result = scoring.score(project, CODE, report, plan, contract, lambda **_: None)
                before = len(accesses)
                scoring.score(project, CODE, report, plan, contract, lambda **_: None)
            self.assertEqual(len(accesses), before)
            self.assertTrue(result['comparison']['supports_held_out_design_review'])
            with (report/'pilot_metrics.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertEqual(sum(row['accepted'] == 'True' for row in rows), 6)

    def test_cpu_score_normal_submission_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project/'work').mkdir()
            with patch('verification_pilot_score.admission') as admit, \
                 patch.object(slurm_submit, 'prepare'), patch.object(slurm_submit, 'allocation', return_value=('shortq', 3600, 'fixture')), \
                 patch.object(slurm_submit.shutil, 'which', return_value='/mock/sbatch'), \
                 patch('job_notifications.preferences', return_value={'mail_user': 'fixture@example.invalid', 'mail_type': 'END'}), \
                 patch('job_notifications.queued_receipt', return_value={'state': 'fixture'}), \
                 patch.object(slurm_submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123\n', '')) as submit:
                self.assertEqual(slurm_submit.main(['--project', str(project), '--code', str(CODE), '--stage', pilot.SCORE]), 0)
            admit.assert_called_once()
            cmd = submit.call_args.args[0]
            self.assertIn('--cpus-per-task=5', cmd)
            self.assertFalse(any(part.startswith('--gres=') for part in cmd))
