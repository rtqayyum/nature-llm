"""CPU-only checks for node exclusion and verified window result reuse."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'scripts'))
from ectt_common import digest, finish, read, write


def module(name, **fields):
    result = types.ModuleType(name)
    result.__dict__.update(fields)
    return result


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, CODE/'scripts'/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class SchedulerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self.fixture.cleanup)
        self.project = Path(self.fixture.name)
        (self.project/'work').mkdir()
        self.commands = []
        self.experiment = module('source_loss_development',
                                 PROBE='preflight', FIT='fit', REPORT='report',
                                 GPU_STAGES=('fit',), JOURNAL='window_submissions')
        self.plan = {'identity': 'fixture-identity', 'tasks': [{'index': n} for n in range(24)],
                     'probes': [], 'recipe': {'maximum_concurrent_gpu_tasks': 4}}
        self.experiment.foundation = lambda *_: self.plan
        self.experiment.submission_steps = lambda _: [
            ('probe', 'preflight', 'ectt-window-preflight', None),
            ('fit', 'fit', 'ectt-window-fit', self.plan['tasks']),
            ('report', 'report', 'ectt-window-report', None)]
        launcher = module('slurm_submit', prepare=lambda *_: None,
                          allocation=lambda _: ('fixture', 7200, 'fixture'),
                          STUDY_STAGES=('preflight', 'fit', 'report'))
        dependencies = {
            'source_loss_development': self.experiment,
            'slurm_submit': launcher,
            'compact_grid_submit': module('compact_grid_submit', _job_id=lambda r: r.stdout.strip()),
            'transfer_submit': module('transfer_submit', clock=lambda _: '02:00:00', timestamp=lambda: 'fixture'),
            'job_notifications': module('job_notifications',
                preferences=lambda _: {'mail_user': 'fixture@example.invalid', 'mail_type': 'END'},
                queued_receipt=lambda *_: {'state': 'fixture'}),
        }
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, dependencies))
        self.stack.enter_context(patch.dict(os.environ, {'TRAFFIC_EXCLUDE_NODES': ''}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.launcher = load_script('source_loss_submit')
        self.stack.enter_context(patch.object(self.launcher.shutil, 'which', return_value='/fixture/sbatch'))
        self.stack.enter_context(patch.object(self.launcher.subprocess, 'run', side_effect=self.command))

    def command(self, command, **kwargs):
        if command[0] == 'squeue':
            return subprocess.CompletedProcess(command, 0, '', '')
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, str(900+len(self.commands))+'\n', '')

    def submit(self, **kwargs):
        return self.launcher.submit(self.project, CODE, experiment=self.experiment, **kwargs)

    def test_retry_excludes_gpu_node_and_preserves_completed_artifacts(self):
        old_jobs = {'probe': '3090915', 'fit': '3090916', 'report': '3090917'}
        journal = self.project/'work/window_submissions/fixture-identity.json'
        write(journal, {'identity': self.plan['identity'], 'jobs': old_jobs,
                        'receipts': {}, 'previous_attempts': []})
        write(self.project/'results/slurm_jobs/3090917/job_status.json', {'execution_state': 'FAILED'})
        saved = self.project/'outputs/completed-case/result.json'
        write(saved, {'rows': list(range(9))})
        finish(saved.parent, self.plan['identity'])
        before = {p: digest(p) for p in saved.parent.iterdir()}
        os.environ['TRAFFIC_EXCLUDE_NODES'] = 'gn05,gn06'
        self.assertEqual(self.submit(retry=True), 0)
        self.assertEqual(len(self.commands), 3)
        self.assertIn('--exclude=gn05,gn06', self.commands[1])
        self.assertFalse(any(arg.startswith('--exclude=') for cmd in (self.commands[0], self.commands[2]) for arg in cmd))
        self.assertIn('--array=0-23%4', self.commands[1])
        self.assertIn('--dependency=afterok:901', self.commands[1])
        self.assertIn('--dependency=afterany:901:902', self.commands[2])
        state = read(journal)
        self.assertEqual(state['previous_attempts'], [old_jobs])
        self.assertEqual(state['identity'], self.plan['identity'])
        self.assertEqual(state['receipts']['fit']['excluded_nodes'], ['gn05', 'gn06'])
        self.assertEqual(state['receipts']['report']['excluded_nodes'], [])
        self.assertEqual(read(self.project/'results/slurm_jobs/902_12/submission.json')['task_index'], 12)
        self.assertEqual({p: digest(p) for p in before}, before)
        self.assertEqual(self.submit(), 0)
        self.assertEqual(len(self.commands), 3)

    def test_default_submissions_have_no_node_exclusion(self):
        self.assertEqual(self.submit(), 0)
        self.assertFalse(any(arg.startswith('--exclude=') for cmd in self.commands for arg in cmd))

    def test_invalid_node_list_fails_before_submission_or_journal_changes(self):
        for value in ('gn05 --hold', '-gn05', 'gn05,', 'gn[05-06]', 'gn05\nother'):
            os.environ['TRAFFIC_EXCLUDE_NODES'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'comma-separated'):
                self.submit()
        self.assertEqual(self.commands, [])
        self.assertFalse((self.project/'work/window_submissions').exists())

    def test_completed_window_is_verified_and_returns_without_fitting(self):
        dependencies = {
            'transfer_plan': module('transfer_plan', fingerprint=lambda _: 'fixture'),
            'verification_pilot': module('verification_pilot'),
            'verification_pilot_candidate_audit': module('verification_pilot_candidate_audit', admission=lambda *_: None),
        }
        with patch.dict(sys.modules, dependencies):
            worker = load_script('verification_window_campaign')
        task = {'index': 0, 'dataset': 'bj', 'window': 1, 'seed': 11}
        plan = {'identity': self.plan['identity'], 'tasks': [task]}
        case = worker.store_path(self.project, plan)/'cases/0'
        write(case/'result.json', {'task': task, 'rows': list(range(9)), 'new_optimizer_steps': 200})
        finish(case, plan['identity'])
        before = {p: digest(p) for p in case.iterdir()}
        report = self.project/'report'
        report.mkdir()
        with patch.object(worker, 'foundation', return_value=plan), \
             patch.object(worker, 'load_contract', return_value=({}, {})), \
             patch.object(worker, 'fit_window', side_effect=AssertionError('Completed case must not train')), \
             patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '0'}):
            self.assertEqual(worker.run(self.project, CODE, report, worker.FIT), 0)
            self.assertEqual(read(report/'transfer_result.json')['new_optimizer_steps'], 0)
            self.assertEqual({p: digest(p) for p in before}, before)
            write(case/'result.json', {'corrupted': True})
            with self.assertRaisesRegex(ValueError, 'Missing/corrupt completed artifact'):
                worker.run(self.project, CODE, report, worker.FIT)


if __name__ == '__main__':
    unittest.main()
