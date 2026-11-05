"""Regression coverage for checkpoint-paired development forecast scoring."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from transfer_attribution import compare, predict_pair


class DevelopmentReplayTests(unittest.TestCase):
    def test_source_and_residual_share_one_encoded_pass(self):
        class Source:
            def __init__(self):
                self.encodes = 0

            def encode(self, x):
                self.encodes += 1
                self.assert_no_grad = not torch.is_grad_enabled()
                return x

            def head(self, h):
                return h*2

        model = Source()
        features = torch.tensor([[[1.], [2.]], [[3.], [4.]]])
        with patch('compact_training.data_batch', return_value=(features, None)) as batch:
            before, after = predict_pair(model, lambda embedding: embedding+1, None, [12, 14], {}, 2, 'cpu')
        self.assertEqual(model.encodes, 1)
        self.assertTrue(model.assert_no_grad)
        self.assertEqual(batch.call_count, 1)
        np.testing.assert_array_equal(before, [[2, 4], [6, 8]])
        np.testing.assert_array_equal(after, [[4, 7], [10, 13]])

    def test_physical_units_clipping_and_replay_rejection(self):
        saved = {'truth': np.array([[0., 15.]]), 'local': np.array([[0., 14.]]),
                 'transfer': np.array([[0., 12.]])}
        before = np.array([[-1.1, .8]])
        after = np.array([[-1.2, .2]])
        scaler = {'minimum': 10., 'maximum': 20.}
        result = compare(before, after, saved, scaler)
        self.assertEqual(result['before_MAE'], 1.5)
        self.assertEqual(result['after_MAE'], 1.5)
        self.assertEqual(result['local_MAE'], .5)
        self.assertEqual(result['before_signed_bias'], 1.5)
        self.assertEqual(result['after_signed_bias'], -1.5)
        self.assertEqual(result['before_clipped_fraction'], .5)
        self.assertEqual(result['after_replay_clipped_fraction'], .5)
        self.assertEqual(result['replay_max_physical_delta'], 0.)
        with self.assertRaisesRegex(ValueError, 'differs from the archive'):
            compare(before, after+.1, saved, scaler)


if __name__ == '__main__':
    unittest.main()
