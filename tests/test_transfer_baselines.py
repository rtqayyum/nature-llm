"""Regression coverage for transfer fold isolation and dependency/recovery paths."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'scripts'))
from ectt_common import read
import slurm_submit
import transfer_plan
import transfer_submit
import transfer_train


class TransferPlanTests(unittest.TestCase):
    def test_matrix_has_complete_target_exclusions_and_budgets(self):
        matrix = transfer_plan.plan(CODE)
        contract = read(CODE/'config/ectt_protocol_v1.json')
        self.assertEqual(len(matrix['source']), 42)
        self.assertEqual(len(matrix['target']), 48)
        self.assertEqual(sum(contract['datasets'][row['dataset']]['budgets'][q]['eligible_for_learned_adaptation']
                             for row in matrix['target'] for q in row['budgets']), 132)
        for target in matrix['target']:
            slug = target['dataset']
            source = matrix['source'][target['source_index']]
            expected = {'bike', 'taxi'} if slug in ('bike', 'taxi') else {slug}
            self.assertEqual(set(source['excluded']), expected)
            self.assertEqual(set(source['sources']), set(contract['datasets'])-expected)
            for key in ('seed', 'family'):
                self.assertEqual(source[key], target[key])

    def test_partial_submission_then_resume_and_no_duplicates(self):
        commands = []

        def launch(cmd, **kwargs):
            if cmd[0] == 'squeue':
                return subprocess.CompletedProcess(cmd, 0, '101\n102\n', '')
            commands.append(cmd)
            if len(commands) == 3:
                raise subprocess.CalledProcessError(1, cmd, output='', stderr='synthetic source rejection')
            job_id = 100+len(commands) if len(commands) <= 2 else 99+len(commands)
            return subprocess.CompletedProcess(cmd, 0, f'{job_id};testcluster\n', '')

        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            (project/'work').mkdir()
            with mock.patch.object(transfer_plan, 'foundation'), mock.patch.object(slurm_submit, 'prepare'), \
                 mock.patch.object(slurm_submit, 'allocation', return_value=('queue', 7200, 'fixture')), \
                 mock.patch('job_notifications.preferences', return_value={'mail_user': 'fixture@example.test', 'mail_type': 'END'}), \
                 mock.patch('job_notifications.queued_receipt', return_value={'state': 'fixture'}), \
                 mock.patch.object(transfer_submit.shutil, 'which', return_value='/mock/sbatch'), \
                 mock.patch.object(transfer_submit.subprocess, 'run', side_effect=launch):
                with self.assertRaisesRegex(ValueError, 'synthetic source rejection'):
                    transfer_submit.submit(project, CODE)
                transfer_submit.submit(project, CODE)
                transfer_submit.submit(project, CODE)
            self.assertEqual(len(commands), 6)
            self.assertIn('--array=0-1%2', commands[1])
            self.assertIn('--dependency=afterok:101', commands[1])
            self.assertIn('--array=0-41%4', commands[3])
            self.assertIn('--dependency=afterok:102', commands[3])
            self.assertIn('--array=0-47%4', commands[4])
            self.assertIn('--dependency=afterok:103', commands[4])
            self.assertIn('--dependency=afterany:101:102:103:104', commands[5])
            self.assertEqual((project/'work/LAST_JOB_ID').read_text().strip(), '105')


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(36, 4)
        self.head = torch.nn.Linear(4, 1)

    def encode(self, x):
        return self.layer(x.flatten(-2))


class TransferRecoveryTests(unittest.TestCase):
    def test_source_episode_and_residual_optimizer_recovery(self):
        torch.set_num_threads(1)
        recipe = read(CODE/'config/shared_transfer_v1.json')
        # Two domains and two epochs exercise a restart on both sides of the
        # epoch cursor. Tiny fixed updates exercise the same trainer code.
        task = {'sources': ['source-a', 'source-b'], 'seed': 11, 'family': 'B04'}
        config = {**read(CODE/'config/ectt_protocol_v1.json')['training_design'],
                  'source_max_epochs': 2, 'target_updates': 4}
        profiles = {slug: {'source_train_raw': [0, 144], 'source_validation_targets': [144, 170]}
                    for slug in task['sources']}
        contract = {'datasets': profiles}
        raw = {slug: np.random.default_rng(i).uniform(1, 100, (4, 180)) for i, slug in enumerate(task['sources'])}
        bank = transfer_train.episode_bank(task, contract, recipe, raw)
        self.assertTrue(all(e['suffix'][1] <= 144 for row in bank.values() for e in row['train']))
        self.assertTrue(all(row['validation']['suffix'] == [144, 160] for row in bank.values()))
        ridge = read(CODE/'config/shared_baselines_v1.json')
        compact = {**read(CODE/'config/compact_local_v1.json'), 'updates': 1,
                   'configurations': [{'width': 8, 'layers': 1, 'heads': 2}]}

        def model():
            torch.manual_seed(11)
            return TinyEncoder()

        def source(directory, stop_after=None):
            return transfer_train.source_fit(model(), raw, task, bank, recipe, config, (ridge, compact),
                                             'cpu', directory, 'source', lambda **_: None, stop_after)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            full = root/'full'
            resumed = root/'resumed'
            source(full)
            source(resumed, 2)
            _, updates = source(resumed)
            self.assertEqual(updates, 2)
            self.assertTrue(transfer_train.equal_state(transfer_train.load(full, 'source'),
                                                       transfer_train.load(resumed, 'source')))
            values = raw['source-a']
            args = (model(), values, [12, 48], transfer_train.fit_scaler(values, [0, 48]),
                    11, recipe, config, 'cpu', lambda **_: None)
            transfer_train.adapt(*args, checkpoint=root/'head-full', identity='head')
            transfer_train.adapt(*args, checkpoint=root/'head-resumed', identity='head', stop_after=2)
            _, result = transfer_train.adapt(*args, checkpoint=root/'head-resumed', identity='head')
            self.assertEqual(result['new_steps'], 2)
            self.assertTrue(transfer_train.equal_state(transfer_train.load(root/'head-full', 'head'),
                                                       transfer_train.load(root/'head-resumed', 'head')))


if __name__ == '__main__':
    unittest.main()
