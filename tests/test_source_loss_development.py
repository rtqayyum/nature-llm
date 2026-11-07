"""Regression checks for the paired source loss development experiment."""
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
from ectt_common import read
import source_loss_development as experiment
import source_loss_submit
import source_loss_train as modified
import slurm_submit
import transfer_train as original


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(36, 4)
        self.head = torch.nn.Linear(4, 1)
        torch.nn.init.constant_(self.head.bias, -2.)

    def encode(self, x):
        return self.layer(x.flatten(-2))


class LossComparisonTests(unittest.TestCase):
    def test_loss_gradient_and_validation_floor(self):
        x = torch.tensor([-2., 1.], requires_grad=True)
        truth = torch.tensor([3., 3.])
        clipped = (modified.training_forecast(x, 'clipped', 'train')-truth).abs().mean()
        raw = (modified.training_forecast(x, 'unclipped', 'train')-truth).abs().mean()
        grad, = torch.autograd.grad(clipped, x)
        other, = torch.autograd.grad(raw, x)
        self.assertEqual(grad[0], 0.)
        self.assertLess(other[0], 0.)
        self.assertTrue(torch.equal(modified.training_forecast(x, 'unclipped', 'validation'), x.clamp_min(0)))

    def test_control_reproduces_original_and_variant_resumes(self):
        torch.set_num_threads(1)
        task = {'sources': ['a', 'b'], 'seed': 11, 'family': 'B04'}
        profiles = {slug: {'source_train_raw': [0, 144], 'source_validation_targets': [144, 170]}
                    for slug in task['sources']}
        raw = {slug: np.random.default_rng(i).uniform(1, 100, (4, 180)) for i, slug in enumerate(task['sources'])}
        recipe = read(CODE/'config/shared_transfer_v1.json')
        training = {**read(CODE/'config/ectt_protocol_v1.json')['training_design'],
                    'source_max_epochs': 2, 'target_updates': 4}
        bank = original.episode_bank(task, {'datasets': profiles}, recipe, raw)
        compact = {**read(CODE/'config/compact_local_v1.json'), 'updates': 1,
                   'configurations': [{'width': 8, 'layers': 1, 'heads': 2}]}
        local = (read(CODE/'config/shared_baselines_v1.json'), compact)
        def model():
            torch.manual_seed(11)
            return Tiny()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = (raw, task, bank, recipe, training, local, 'cpu')
            original.source_fit(model(), *args, root/'legacy', 'control', lambda **_: None)
            modified.fit(model(), *args, root/'control', 'control', lambda **_: None, 'clipped')
            self.assertTrue(original.equal_state(original.load(root/'legacy', 'control'),
                                                 original.load(root/'control', 'control')))
            modified.fit(model(), *args, root/'full', 'variant', lambda **_: None, 'unclipped')
            modified.fit(model(), *args, root/'resumed', 'variant', lambda **_: None, 'unclipped', stop_after=2)
            modified.fit(model(), *args, root/'resumed', 'variant', lambda **_: None, 'unclipped')
            self.assertTrue(original.equal_state(original.load(root/'full', 'variant'),
                                                 original.load(root/'resumed', 'variant')))

    def test_matrix_and_submit_dependencies_survive_partial_submission(self):
        plan = experiment.matrix(CODE)
        self.assertEqual(len(plan['tasks']), 24)
        self.assertEqual(len(plan['probes']), 4)
        self.assertEqual({r['dataset'] for r in plan['tasks']}, {'pems08', 'bj'})
        commands = []
        def run(cmd, **kwargs):
            if cmd[0] == 'squeue':
                return subprocess.CompletedProcess(cmd, 0, '101\n', '')
            commands.append(cmd)
            if len(commands) == 2:
                raise subprocess.CalledProcessError(1, cmd, output='', stderr='synthetic rejection')
            return subprocess.CompletedProcess(cmd, 0, str(100+len(commands))+'\n', '')
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            (project/'work').mkdir()
            with patch.object(experiment, 'foundation', return_value=plan), \
                 patch.object(slurm_submit, 'prepare'), \
                 patch.object(slurm_submit, 'allocation', return_value=('queue', 7200, 'fixture')), \
                 patch('job_notifications.preferences', return_value={'mail_user':'fixture@example.invalid','mail_type':'END'}), \
                 patch('job_notifications.queued_receipt', return_value={'state':'synthetic'}), \
                 patch.object(source_loss_submit.shutil, 'which', return_value='/mock/sbatch'), \
                 patch.object(source_loss_submit.subprocess, 'run', side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    source_loss_submit.submit(project, CODE)
                # Simulate a crash before worker receipts were fully written.
                (project/'results/slurm_jobs/101/submission.json').unlink()
                source_loss_submit.submit(project, CODE)
                source_loss_submit.submit(project, CODE)
            self.assertEqual(len(commands), 4)
            self.assertIn('--array=0-3%4', commands[0])
            self.assertIn('--array=0-23%4', commands[2])
            self.assertIn('--dependency=afterok:101', commands[2])
            self.assertIn('--dependency=afterany:101:103', commands[3])
            self.assertTrue((project/'results/slurm_jobs/101/submission.json').exists())


if __name__ == '__main__':
    unittest.main()
