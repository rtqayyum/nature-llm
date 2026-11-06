"""Regression coverage for paired blocks, floor effects and frozen replay."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ectt_statistics import decision
from frozen_verifier_replay import replay
from slurm_submit import stage_conflicts
from transfer_parallel import CPU, GPU, block_means, bootstrap, gradient_summary


class ParallelAnalysisTests(unittest.TestCase):
    def test_common_indices_across_arms_even_in_partial_last_block(self):
        series = np.column_stack((np.arange(1., 8.), np.arange(1., 8.)+4, 3*np.arange(1., 8.)))
        sampled = block_means(series, 3, 200, np.random.default_rng(17))
        np.testing.assert_allclose(sampled[:, 1]-sampled[:, 0], 4)
        np.testing.assert_allclose(sampled[:, 2], sampled[:, 0]*3)
        report = bootstrap({'short': series}, [3], 200)['3']
        self.assertTrue(report['domains']['short']['unstable_descriptive'])
        self.assertEqual(report['macro']['B04']['MAE_ratio'], 3.)
        missing = bootstrap({'zero': np.zeros((8, 3))}, [3], 200)['3']
        self.assertIsNone(missing['macro']['B04']['MAE_ratio_95_interval'])

    def test_floor_blocks_gradient_below_zero(self):
        physical = torch.tensor([[-2., -1., 1.]], requires_grad=True)
        truth = torch.tensor([[4., 0., 2.]])
        floor = (physical.clamp_min(0)-truth).abs().mean()
        clipped_grad, = torch.autograd.grad(floor, physical)
        full_grad, = torch.autograd.grad((physical-truth).abs().mean(), physical)
        np.testing.assert_array_equal(clipped_grad.numpy()[:, :2], [[0., 0.]])
        self.assertTrue(torch.all(full_grad < 0))
        report = gradient_summary(physical.detach().numpy(), truth.numpy(), 1.)
        self.assertEqual(report['negative_pre_floor_values'], 2)
        self.assertEqual(report['positive_truth_with_blocked_output_gradient'], 1)
        self.assertGreater(report['normalized_MAE_without_floor'], report['normalized_MAE_with_floor'])

    def test_only_complementary_analyses_may_overlap(self):
        self.assertFalse(stage_conflicts(CPU, GPU))
        self.assertFalse(stage_conflicts(GPU, CPU))
        self.assertTrue(stage_conflicts(GPU, GPU))
        self.assertTrue(stage_conflicts(CPU, 'shared-transfer-source'))


class ChronologicalReplayTests(unittest.TestCase):
    def test_stops_reading_at_first_crossing_and_freezes_candidates(self):
        origins = np.arange(10, 90)
        local, transfer = np.full((80, 2), 5.), np.zeros((80, 2))
        expected = decision(np.full(80, -5.), np.full(80, 5.))
        accessed = []

        def label(origin):
            accessed.append(origin)
            local[:] = 0
            return np.zeros(2)

        result = replay(origins, local, transfer, 1., label)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['selected_candidate'], 'transfer')
        self.assertEqual(result['first_certificate'], expected['first_certificate'])
        self.assertEqual(accessed, origins[:expected['first_certificate']].tolist())
        self.assertEqual(result['verification_node_values_read'], 2*len(accessed))

    def test_censoring_and_prelabel_validation(self):
        seen = []

        def label(origin):
            seen.append(origin)
            return np.zeros(1)

        predictions = np.zeros((12, 1))
        result = replay(np.arange(12), predictions, predictions, 1., label)
        self.assertTrue(result['censored'])
        self.assertEqual(result['first_certificate'], 13)
        self.assertEqual(result['selected_candidate'], 'local')
        self.assertEqual(seen, list(range(12)))
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            replay(np.arange(12)*2, predictions, predictions, 1., label)
        self.assertEqual(seen, list(range(12)))


if __name__ == '__main__':
    unittest.main()
