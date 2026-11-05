"""Regression tests for the read-only transfer diagnostic."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ectt_common import digest, write
from ectt_data import fit_scaler
from shared_baselines import independent_scores
from transfer_diagnostics import MATRIX, PRODUCER, diagnose_case, foundation


class TransferDiagnosticTests(unittest.TestCase):
    def test_completed_collection_is_bound_to_published_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            job = project/'results/slurm_jobs/621882'
            write(job/'job_status.json', {'job_id': '621882', 'stage': 'shared-transfer-report',
                  'execution_state': 'COMPLETED', 'shell_exit_code': 0})
            submission = {'project': str(project), 'code': str(project/'work/code_versions'/PRODUCER)}
            write(job/'submission.json', submission)
            write(job/'report/transfer_result.json', {'stage': 'shared-transfer-report', 'identity': MATRIX,
                  'source_jobs': {'data': '621878', 'probe': '621879', 'source': '621880', 'target': '621881'},
                  'verified': {'source': {'verified': 42, 'expected': 42}}, 'failed': {},
                  'benchmark_experiments_completed': 192, 'scientific_status': 'UNCONDITIONAL_TRANSFER_BASELINES_COMPLETE'})
            with patch('transfer_diagnostics.plan.plan', return_value={'identity': MATRIX}):
                _, path = foundation(project, project/'code')
                self.assertEqual(path, job/'report/transfer_metrics.csv')
                write(job/'submission.json', {**submission, 'code': str(project/'wrong')})
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    foundation(project, project/'code')

    def test_scores_saved_test_but_only_inventories_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            case = directory/'0.05'
            (case/'fit').mkdir(parents=True)
            raw = np.arange(80, dtype=np.float64).reshape(2, 40)
            support, verification, test = [0, 24], [24, 30], [30, 40]
            profile = {'budgets': {'0.05': {'support_raw': support, 'verification_target_rows': verification,
                                           'eligible_for_learned_adaptation': True}}, 'test_targets': test}
            truth = raw[:, slice(*test)].T
            local, transfer = truth+1, truth+4
            np.savez(case/'test_forecasts.npz', origins=np.arange(*test), truth=truth, local=local, transfer=transfer)
            candidates = {'origins': np.arange(*verification), 'local': np.ones((6, 2)), 'transfer': np.ones((6, 2))*2}
            np.savez(case/'verification_candidates.npz', **candidates)
            checkpoint = case/'fit/checkpoint.pt'
            torch.save({'cursor': {'step': 40, 'losses': [2.]*20+[1.]*20}}, checkpoint)
            sha = digest(checkpoint)
            write(case/'fit/latest.json', {'file': checkpoint.name, 'sha256': sha, 'step': 40})
            write(case/'frozen_candidates.json', {'scaler': fit_scaler(raw, support),
                  'local_family': 'ridge', 'fallback': False, 'test_targets': test, 'source_checkpoint_sha256': 'source',
                  'verification_labels_used': 0, 'test_labels_used_for_fitting': 0, 'support_raw': support,
                  'unique_support_values': 48, 'verification_targets': verification,
                  'verification_history_union': [12, 29], 'support_scale': 1.,
                  'target_residual_sha256': sha, 'local_forecast_sha256': 'local'})

            def artifacts():
                return {str(p.relative_to(directory)): digest(p) for p in case.rglob('*') if p.is_file()}

            row = {'eligible': 'True', 'optimizer_steps': '40', 'local_family': 'ridge'}
            for model, value in [('local', local), ('transfer', transfer)]:
                row.update({model+'_'+k: str(v) for k, v in independent_scores(value, truth).items()})
            task = {'family': 'B04', 'dataset': 'pems08', 'seed': 11}

            def audit():
                return diagnose_case(directory, artifacts(), task, '0.05', row, raw, profile,
                                     {'source_best_sha256': 'source'}, 40)

            result, evidence = audit()
            self.assertEqual(result['MAE_ratio'], 4.)
            self.assertEqual(result['transfer_signed_bias'], 4.)
            self.assertEqual(result['fraction_origins_transfer_worse'], 1.)
            self.assertEqual(result['adaptation_initial20_normalized_MAE'], 2.)
            self.assertEqual(result['adaptation_final20_normalized_MAE'], 1.)
            self.assertEqual(evidence['mean_candidate_disagreement'], 1.)
            np.savez(case/'verification_candidates.npz', **candidates, truth=raw[:, 24:30].T)
            with self.assertRaisesRegex(ValueError, 'unexpected columns'):
                audit()


if __name__ == '__main__':
    unittest.main()
