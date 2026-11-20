"""Cover joint exclusions, prospective policy timing and interrupted shared fits."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'scripts'))
import heldout_candidates as campaign
import slurm_submit
from ectt_common import digest, load_contract, read, write


class HeldoutTests(unittest.TestCase):
    def test_matrix_and_scheduler(self):
        plan = campaign.matrix(CODE)
        self.assertEqual(len(plan['tasks']), 45)
        self.assertEqual(sum(len(t['targets']) for t in plan['tasks']), 54)
        joint = [t for t in plan['tasks'] if 'bike' in t['targets']]
        self.assertEqual(len(joint), 9)
        for t in joint:
            self.assertEqual(t['targets'], ['bike', 'taxi'])
            self.assertTrue(set(t['sources']).isdisjoint(t['targets']))
        steps = campaign.submission_steps(plan)
        self.assertEqual([s[0] for s in steps], ['probe', 'fit', 'report'])
        self.assertEqual([len(s[3] or []) for s in steps], [0, 45, 0])
        with patch('source_loss_submit.submit', return_value=0) as submit:
            self.assertEqual(slurm_submit.main(['--project', str(CODE), '--code', str(CODE),
                                                '--stage', campaign.LAUNCH]), 0)
            submit.assert_called_once_with(CODE, CODE, retry=False, experiment=campaign)

    def test_one_submission_and_idempotent_resubmission(self):
        commands = []
        def slurm(args, **kwargs):
            if args[0] == 'squeue':
                output = ''
            elif args[0] == 'scontrol':
                output = 'PartitionName='+args[3]+' MaxTime=2-00:00:00'
            elif args[0] == 'sbatch':
                commands.append(args)
                output = str(700+len(commands))
            else:
                self.fail(str(args))
            return subprocess.CompletedProcess(args, 0, output, '')
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            code = project/'work/code_versions/fixture'
            code.mkdir(parents=True)
            (code/'Batch').write_bytes((CODE/'Batch').read_bytes())
            args = ['--project', str(project), '--code', str(code), '--stage', campaign.LAUNCH]
            plan = campaign.matrix(CODE)
            with patch.object(slurm_submit, 'ROOT', project), \
                    patch.object(campaign, 'foundation', return_value=plan), \
                    patch('source_loss_submit.shutil.which', return_value='sbatch'), \
                    patch('source_loss_submit.subprocess.run', side_effect=slurm), \
                    patch('job_notifications.preferences', return_value={'mail_user': 'fixture', 'mail_type': 'END'}), \
                    patch('job_notifications.queued_receipt', return_value={'state': 'TEST'}):
                self.assertEqual(slurm_submit.main(args), 0)
                self.assertEqual(slurm_submit.main(args), 0)
            self.assertEqual(len(commands), 3)
            self.assertIn('--array=0-44%4', commands[1])
            self.assertIn('--dependency=afterok:701', commands[1])
            self.assertIn('--dependency=afterany:701:702', commands[2])
            self.assertIn('--partition=gpuq', commands[1])
            self.assertIn('--partition=shortq', commands[0])
            self.assertIn('--partition=shortq', commands[2])
            self.assertTrue((project/'results/slurm_jobs/702_44/submission.json').is_file())

    def test_two_thirds_stops_reading_at_its_one_declared_look(self):
        for n in [11, 35, 36, 37]:
            for gain in [True, False]:
                read_rows = []
                def matured(origin):
                    read_rows.append(origin)
                    return np.array([0. if gain else 1.])
                args = (np.arange(n), np.ones((n, 1)), np.zeros((n, 1)), 1., matured)
                result = campaign.policy_decision(campaign.TWO_THIRDS, *args,
                                                  dict(rho=.01, delta=.05, eta=.01, minimum=12))
                expected = (2*n+2)//3
                self.assertEqual(read_rows, list(range(expected)))
                self.assertEqual(result['verification_origins_read'], expected)
                self.assertEqual(result['accepted'], gain and expected >= 12)

    def test_partial_nyc_group_resumes_and_collector_checks_artifacts(self):
        plan = campaign.matrix(CODE)
        task = next(t for t in plan['tasks'] if t['targets'] == ['bike', 'taxi'])
        plan = {**plan, 'tasks': [task]}
        contract, _ = load_contract(CODE)
        profiles = {
            d: {'shape': [1, 60, 2], 'test_targets': [40, 42], 'container_sha256': d,
                'budgets': {'0.05': {'support_raw': [0, 20], 'support_target_rows': [12, 20],
                                      'verification_target_rows': [20, 40]}}}
            for d in task['targets']}
        contract = {**contract, 'datasets': profiles}
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project/'outputs'
            jobs = {'probe': '101', 'fit': '102', 'report': '103'}
            def job(name, stage):
                directory = project/'results/slurm_jobs'/name
                write(directory/'job_status.json', dict(stage=stage, execution_state='COMPLETED', shell_exit_code=0))
                report = directory/'report'
                report.mkdir()
                return report
            pre = job(jobs['probe'], campaign.PROBE)
            fit = job('102_'+str(task['index']), campaign.FIT)
            report = job(jobs['report'], campaign.REPORT)
            write(project/'work'/campaign.JOURNAL/(plan['identity']+'.json'), {'jobs': jobs})
            plan['reviewed_hashes'] = {}
            write(pre/'S01_packet/manifest.json', {'identity': plan['identity'], 'files': {}, 'tasks': [task]})
            def source(project, code, report, group, task, plan, contract, progress, probe):
                (group/'best.pt').write_bytes(b'checkpoint')
                return {'task': task, 'best_sha256': digest(group/'best.pt'), 'new_optimizer_steps': 7}
            def freeze(project, code, report, case, target, plan, contract, source, progress):
                slug = target['dataset']
                profile = contract['datasets'][slug]
                ledger = profile['budgets']['0.05']
                if slug == 'taxi' and freeze.interrupt:
                    freeze.interrupt = False
                    raise RuntimeError('simulated interruption')
                folder = case/'target'
                folder.mkdir(exist_ok=True)
                (folder/'weights.pt').write_bytes(b'head')
                sha = digest(folder/'weights.pt')
                write(folder/'latest.json', {'file': 'weights.pt', 'sha256': sha})
                meta = {'task': target, 'source_checkpoint_sha256': source['best_sha256'],
                        'target_residual_sha256': sha, 'target_container_sha256': slug,
                        'support_raw': ledger['support_raw'], 'support_targets': ledger['support_target_rows'],
                        'verification_targets': ledger['verification_target_rows'], 'test_targets': profile['test_targets'],
                        'test_labels_used_for_scoring': 0, 'verification_labels_used_for_scoring': 0, 'budget': '0.05',
                        'local_family': 'ridge', 'scaler': {}, 'support_scale': 1., 'local_completion_sha256': 'local'}
                write(case/'frozen_candidates.json', meta)
                for kind in ['verification', 'test']:
                    origins = np.arange(*meta[kind+'_targets'])
                    np.savez(case/(kind+'_candidates.npz'), origins=origins, local=origins[:, None], transfer=origins[:, None])
                return {'optimizer_steps': contract['training_design']['target_updates'], 'new_optimizer_steps': 3}
            freeze.interrupt = True
            args = (project, CODE, fit, root, task, plan, contract, lambda **kw: None)
            with patch.object(campaign.pilot, 'train_source', side_effect=source) as train, \
                    patch.object(campaign.pilot, 'freeze_target', side_effect=freeze) as adapt:
                with self.assertRaises(RuntimeError):
                    campaign.fit_group(*args)
                self.assertEqual(campaign.fit_group(*args)['new_optimizer_steps'], 3)
                self.assertEqual(campaign.fit_group(*args)['new_optimizer_steps'], 0)
                self.assertEqual(train.call_count, 1)
                self.assertEqual(adapt.call_count, 3)
            result = campaign.collect(project, report, root, plan, contract)
            self.assertEqual(result['failed'], [])
            self.assertEqual(result['diagnostic_cases_checked'], 2)
            inventory = read(report/'candidate_inventory.json')['records']
            (Path(inventory[0]['case'])/'test_candidates.npz').write_bytes(b'damaged')
            result = campaign.collect(project, report, root, plan, contract)
            self.assertEqual(len(result['failed']), 1)
            self.assertEqual(result['scientific_status'], 'HELDOUT_CANDIDATES_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
