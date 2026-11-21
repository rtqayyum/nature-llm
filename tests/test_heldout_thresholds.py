"""Threshold regressions through the CPU scorer's chronological decision entry point."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'scripts'))
from ectt_common import load_contract
from heldout_score import decisions_for
from heldout_thresholds import PI_HI, PI_LO, paired_row, sequential_decision


class ThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(CODE)[0]

    def decide(self, local, transfer, truth, scale=1.0):
        n = len(truth)
        meta = {'verification_targets': [0, n], 'support_scale': scale}
        forecasts = {name: np.full((n, 1), value, dtype=float)
                     for name, value in [('local', local), ('transfer', transfer)]}
        return decisions_for(meta, forecasts, np.asarray([truth]), self.contract)

    def test_sequential_boundaries(self):
        # References evaluated with 110 decimal digits from each exact binary64
        # forecast. The one-sided oracle uses erf directly, independently of the
        # positive series. Adjacent floats straddle each strict boundary.
        cases = [
            ('locked_two_sided', 0.0807838110213078, False,
             '-0.00999999999999998937135649128291259969882174588820747514014562286428470'),
            ('locked_two_sided', 0.08078381102130787, False,
             '-0.00999999999999999824609418152962644853922671620765941441128218964014208'),
            ('locked_two_sided', 0.08078381102130788, True,
             '-0.01000000000000000002104171957896798265747096061892201591815373712023038'),
            ('one_sided_mixture', 0.053026306393153776, False,
             '0.99999999999999973835460761682255288928985098122221031215687429991604002'),
            ('one_sided_mixture', 0.05302630639315378, True,
             '1.00000000000000000793168536453623621993830996568998986910663280569536622'),
        ]
        for rule, value, expected, reference in cases:
            with self.subTest(rule=rule, forecast=value):
                report = self.decide(value, 0., [0.] * 12)[rule]
                self.assertEqual(report['accepted'], expected)
                self.assertEqual(report['verification_origins_read'], 12)
                point = report['trace'][-1]
                key = 'upper_endpoint_interval' if rule == 'locked_two_sided' else 'evidence_times_delta_interval'
                low, high = map(Decimal, point[key])
                self.assertLessEqual(low, Decimal(reference))
                self.assertLessEqual(Decimal(reference), high)
                self.assertFalse(point['unresolved_at_precision_limit'])

    def test_ambiguous_endpoint_uses_refinement_and_conservative_fallback(self):
        # Place the exact sample sum within a chosen distance of the crossing.
        # The 140-digit construction gives more precision than either pass.
        with localcontext() as context:
            context.prec = 140
            width_sum = (Decimal('1.01') * Decimal('40400').ln()).sqrt()
            crossing = Fraction(-Decimal('.12') - width_sum)
        rho, delta, margin = Fraction(1, 100), Fraction(1, 20), Fraction(1, 100)
        for displacement, expected, unresolved in [
            (-Fraction(1, 10**55), True, False),
            (Fraction(1, 10**55), False, False),
            (-Fraction(1, 10**95), False, True),
        ]:
            decision, trace = sequential_decision(
                'locked_two_sided', crossing + displacement, Fraction(1), 12, rho, delta, margin)
            self.assertEqual(decision, expected)
            self.assertEqual(trace['unresolved_at_precision_limit'], unresolved)

    def test_fixed_look_ties_and_neighbours(self):
        for rule, available, look in [('fixed_two_thirds_mean', 33, 22), ('fixed_budget_mean', 36, 36)]:
            for value in (np.nextafter(9., 8.), 9., np.nextafter(9., 10.)):
                prefix = [value] + [9.] * (look // 2 - 1) + [8.] * (look // 2)
                expected_excess = (Fraction(9) - Fraction(float(value))) * 2 / (100 * look)
                for sequence in (prefix, prefix[::-1]):
                    with self.subTest(rule=rule, value=value, first=sequence[0]):
                        truth = sequence + [10.] * (available - look)
                        report = self.decide(10., 8., truth, 100.)[rule]
                        self.assertEqual(report['accepted'], expected_excess > 0)
                        self.assertEqual(report['verification_origins_read'], look)
                        exact = Fraction(report['trace'][-1]['exact_gain_above_margin'])
                        self.assertEqual(exact, expected_excess)
                        self.assertFalse(any(point['accepted'] for point in report['trace'][:-1]))

    def test_binary_forecasts_and_scale(self):
        local = np.array([np.nextafter(1., 2.), 1e20, 0.])
        transfer = np.array([1., 1e20, 0.25])
        truth = np.array([0.125, 1e-20, np.nextafter(0., 1.)])
        scale = Fraction.from_float(1.1)
        observed = paired_row(local, transfer, scale, 13, lambda i: truth)
        l, r, y = [list(map(Fraction.from_float, x.tolist())) for x in (local, transfer, truth)]
        expected_difference = sum(abs(b-c)-abs(a-c) for a, b, c in zip(l, r, y)) / (len(y)*scale)
        expected_bound = sum(abs(b-a) for a, b in zip(l, r)) / (len(y)*scale)
        self.assertEqual(observed, (expected_difference, expected_bound))

    def test_pi_enclosure_for_half_normal_mixture(self):
        def atan_interval(k):
            terms = 80  # even length: partial sum is a lower bound
            lo = sum(Fraction((-1)**i, (2*i+1)*k**(2*i+1)) for i in range(terms))
            return lo, lo + Fraction(1, (2*terms+1)*k**(2*terms+1))
        low5, high5 = atan_interval(5)
        low239, high239 = atan_interval(239)
        self.assertLess(Fraction(PI_LO), 16*low5 - 4*high239)
        self.assertLess(16*high5 - 4*low239, Fraction(PI_HI))


if __name__ == '__main__':
    unittest.main()
