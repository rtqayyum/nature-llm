"""Exercise the actual anchored model, recovery, queue, packet and collector."""
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
from shared_baselines import independent_scores
from anchored_transfer_model import AnchoredTransfer, initial_check
import anchored_transfer_development as experiment
import source_loss_development as predecessor
import source_loss_submit as submitter
import source_loss_train
import transfer_train
import slurm_submit


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(36, 4)
        self.traffic = torch.nn.Module()
        self.traffic.output_linear = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))

    def encode(self, x):
        return self.layer(x.flatten(-2))

    def head(self, encoded):
        return self.traffic.output_linear(encoded)


def model(arm):
    torch.manual_seed(11)
    return AnchoredTransfer(Tiny(), arm)


class AnchorTests(unittest.TestCase):
    def test_matched_models_and_last_value_anchor(self):
        torch.set_num_threads(1)
        inputs = torch.randn(2, 3, 12, 3)
        direct, anchored = (model(arm) for arm in ('direct_zero', 'persistence_residual'))
        self.assertTrue(transfer_train.equal_state(direct.state_dict(), anchored.state_dict()))
        self.assertEqual(initial_check(direct, inputs)['maximum_normalized_difference'], 0.)
        self.assertEqual(initial_check(anchored, inputs)['maximum_normalized_difference'], 0.)
        # Both support heads receive identical features and parameter dimensions.
        self.assertTrue(torch.equal(direct.encode(inputs), anchored.encode(inputs)))
        self.assertTrue(torch.equal(anchored(inputs)-direct(inputs), inputs[..., -1, 0]))
        transfer_train.check_invariance(anchored, inputs, 'B04')
        # Exercise B04's nested single-linear-head route as well as B03's
        # released three-layer prediction module above.
        class QwenLike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.head = torch.nn.Linear(4, 1)
            def encode(self, x):
                return x.flatten(-2)[..., :4]
            def head(self, hidden):
                return self.model.head(hidden)
        qwen = AnchoredTransfer(QwenLike(), 'persistence_residual')
        initial_check(qwen, inputs)
        with self.assertRaises(ValueError):
            AnchoredTransfer(Tiny(), 'unknown')

    def test_actual_source_and_target_recovery_both_conditions(self):
        task = {'sources': ['a', 'b'], 'seed': 11, 'family': 'B04'}
        profiles = {slug: {'source_train_raw': [0, 144], 'source_validation_targets': [144, 170]}
                    for slug in task['sources']}
        raw = {slug: np.random.default_rng(i).uniform(1, 100, (3, 180)) for i, slug in enumerate(task['sources'])}
        recipe = read(CODE/'config/shared_transfer_v1.json')
        training = {**read(CODE/'config/ectt_protocol_v1.json')['training_design'],
                    'source_max_epochs': 2, 'target_updates': 4}
        bank = transfer_train.episode_bank(task, {'datasets': profiles}, recipe, raw)
        compact = {**read(CODE/'config/compact_local_v1.json'), 'updates': 1,
                   'configurations': [{'width': 8, 'layers': 1, 'heads': 2}]}
        local = read(CODE/'config/shared_baselines_v1.json'), compact
        from ectt_data import fit_scaler
        with tempfile.TemporaryDirectory() as td:
            for arm in ('direct_zero', 'persistence_residual'):
                root = Path(td)/arm
                args = (raw, task, bank, recipe, training, local, 'cpu')
                full, resumed = model(arm), model(arm)
                source_loss_train.fit(full, *args, root/'full', arm, lambda **_: None, 'unclipped')
                source_loss_train.fit(resumed, *args, root/'resume', arm, lambda **_: None, 'unclipped', stop_after=2)
                resumed = model(arm)
                source_loss_train.fit(resumed, *args, root/'resume', arm, lambda **_: None, 'unclipped')
                self.assertTrue(transfer_train.equal_state(transfer_train.load(root/'full', arm),
                                                           transfer_train.load(root/'resume', arm)))
                a, b = bank['a']['train'][0]['support']
                args = (resumed, raw['a'], [a+12, b], fit_scaler(raw['a'], [a, b]),
                        11, recipe, training, 'cpu', lambda **_: None)
                transfer_train.adapt(*args, checkpoint=root/'head-full', identity=arm)
                transfer_train.adapt(*args, checkpoint=root/'head-resume', identity=arm, stop_after=2)
                transfer_train.adapt(*args, checkpoint=root/'head-resume', identity=arm)
                self.assertTrue(transfer_train.equal_state(transfer_train.load(root/'head-full', arm),
                                                           transfer_train.load(root/'head-resume', arm)))

    def test_four_job_chain_recovers_partial_submission_without_duplicates(self):
        plan = experiment.matrix(CODE)
        self.assertEqual(len(plan['tasks']), 24)
        self.assertEqual(len(plan['probes']), 4)
        self.assertEqual({row['dataset'] for row in plan['tasks']}, {'pems08', 'bj'})
        for row in plan['tasks']:
            self.assertNotIn(row['dataset'], row['sources'])
        commands = []
        def run(command, **kwargs):
            if command[0] == 'squeue':
                return subprocess.CompletedProcess(command, 0, '101\n102\n', '')
            commands.append(command)
            if len(commands) == 3:
                raise subprocess.CalledProcessError(1, command, output='', stderr='synthetic rejection')
            return subprocess.CompletedProcess(command, 0, str(100+len(commands))+'\n', '')
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            (project/'work').mkdir()
            with patch.object(experiment, 'foundation', return_value=plan), \
                 patch.object(slurm_submit, 'prepare'), \
                 patch.object(slurm_submit, 'allocation', return_value=('queue', 7200, 'fixture')), \
                 patch('job_notifications.preferences', return_value={'mail_user': 'fixture@example.invalid', 'mail_type': 'END'}), \
                 patch('job_notifications.queued_receipt', return_value={'state': 'synthetic'}), \
                 patch.object(submitter.shutil, 'which', return_value='/mock/sbatch'), \
                 patch.object(submitter.subprocess, 'run', side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    submitter.submit(project, CODE, experiment=experiment)
                (project/'results/slurm_jobs/102/submission.json').unlink()
                submitter.submit(project, CODE, experiment=experiment)
                submitter.submit(project, CODE, experiment=experiment)
            self.assertEqual(len(commands), 5)
            self.assertFalse(any(c.startswith('--dependency=') for c in commands[0]+commands[1]))
            self.assertIn('--array=0-3%4', commands[1])
            self.assertIn('--array=0-23%4', commands[3])
            self.assertIn('--dependency=afterok:102', commands[3])
            self.assertIn('--dependency=afterany:101:102:104', commands[4])
            self.assertTrue((project/'results/slurm_jobs/102/submission.json').exists())

    def test_dispatch_uses_new_worker_and_review_stays_pending(self):
        plan = experiment.matrix(CODE)
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            (project/'work').mkdir()
            report = project/'report'
            report.mkdir()
            with patch('ectt_env.ensure', return_value=Path('/fixture/python')), \
                 patch.object(experiment.subprocess, 'call', return_value=0) as call:
                experiment.dispatch(project, CODE, report, experiment.REVIEW)
                self.assertEqual(call.call_args.args[0][2], str(CODE/'scripts/anchored_transfer_development.py'))
            with patch('statistical_audit.review', return_value={'independent_review_complete': False}):
                result = experiment.review_packet(project, CODE, report, plan, lambda **_: None)
            self.assertEqual(result['scientific_status'], 'S01_PACKET_PREPARED_SIGNOFF_PENDING')
            manifest = read(report/'S01_packet/manifest.json')
            self.assertFalse(manifest['independent_review_complete'])
            self.assertEqual(manifest['traffic_labels_read'], 0)
            for name, sha in manifest['code_files'].items():
                self.assertEqual(digest(report/'S01_packet'/name), sha)

    def test_collector_verifies_pairing_and_rejects_corrupt_case(self):
        plan = experiment.matrix(CODE)
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            report = project/'report'
            report.mkdir()
            jobs = {'review': '10', 'probe': '11', 'fit': '12', 'report': '13'}
            write(project/'work'/experiment.JOURNAL/(plan['identity']+'.json'), {'jobs': jobs})
            packet = project/'results/slurm_jobs/10/report/S01_packet'
            write(packet/'manifest.json', {'code_files': {}})
            write(packet.parent/'transfer_result.json', {'identity': plan['identity'],
                  'scientific_status': 'S01_PACKET_PREPARED_SIGNOFF_PENDING'})
            write(packet.parent.parent/'job_status.json', {'stage': experiment.REVIEW, 'execution_state': 'COMPLETED', 'shell_exit_code': 0})
            values = {'truth': np.ones((4, 2)), 'origins': np.arange(4), 'local': np.full((4, 2), 2.),
                      'transfer': np.full((4, 2), 1.5), 'persistence': np.full((4, 2), 2.)}
            for task in plan['tasks']:
                case = experiment.store_path(project, plan)/'cases'/str(task['index'])
                case.mkdir(parents=True)
                np.savez(case/'test_forecasts.npz', **values)
                scores = {name: independent_scores(values[name], values['truth']) for name in ('local', 'transfer', 'persistence')}
                result = {'task': task, 'source': {'epochs': 2, 'source_steps': 14,
                          'best_validation_normalized_MAE': 1., 'temporary_support_head_updates_in_completed_trajectory': 5600},
                          'target': {'metrics': scores, 'optimizer_steps': 200,
                                     'source_clipped_fraction': 0., 'adapted_clipped_fraction': 0.}}
                write(case/'result.json', result)
                finish(case, plan['identity'])
                write(project/'results/slurm_jobs'/('12_'+str(task['index']))/'job_status.json',
                      {'stage': experiment.FIT, 'execution_state': 'COMPLETED', 'shell_exit_code': 0})
                old = predecessor.store_path(project, plan['predecessor'])/'cases'/str(task['predecessor_index'])
                old.mkdir(parents=True, exist_ok=True)
                np.savez(old/'test_forecasts.npz', **values)
                old_task = plan['predecessor']['tasks'][task['predecessor_index']]
                write(old/'result.json', {'task': old_task, 'target': {'metrics': scores}})
                finish(old, plan['predecessor']['identity'])
            result = experiment.collection(project, CODE, report, plan)
            self.assertFalse(result['failed'])
            self.assertEqual(result['diagnostic_cases_checked'], 24)
            self.assertIn('independent s01', (report/'SUMMARY.md').read_text().lower())
            case = experiment.store_path(project, plan)/'cases/0'
            with (case/'test_forecasts.npz').open('ab') as stream:
                stream.write(b'corrupt')
            result = experiment.collection(project, CODE, report, plan)
            self.assertEqual(set(result['failed']), {'0'})
            self.assertEqual(result['scientific_status'], 'ANCHOR_DEVELOPMENT_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
