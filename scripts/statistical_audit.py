"""Internal implementation review; cannot clear independent S01 sign-off."""
import math
import numpy as np
from ectt_common import write, read
from ectt_statistics import confidence_sequence, decision, paired_difference, simulate


def review(store, code, progress):
    store.mkdir(parents=True, exist_ok=True)
    x, w = np.polynomial.hermite.hermgauss(96)
    errors = []
    for rho in (.01, .5, 1.):
        for ratio in (0., .1, 1., 3.):
            variance = rho*ratio
            for sign in (-1., 0., 1.):
                s = sign*math.sqrt(rho)/2
                lam = x*math.sqrt(2/rho)
                numeric = float(np.sum(w*np.exp(lam*s-lam*lam*variance/2))/math.sqrt(math.pi))
                analytic = math.sqrt(rho/(variance+rho))*math.exp(s*s/(2*(variance+rho)))
                errors.append(abs(numeric-analytic)/analytic)
    if max(errors) > 1e-10:
        raise ValueError('Gaussian mixture quadrature disagrees with closed form')
    # Independent direct equation comparison on ordinary inputs, plus a case
    # whose old ratio/product overflowed although its width is representable.
    d = np.array([.1, -.2, .3]); b = np.array([.5, .6, .7])
    low, high = confidence_sequence(d, b)
    n = np.arange(1, 4); v = np.cumsum(b*b)
    expected = np.sqrt((v+.01)*np.log((v+.01)/(.01*.05**2)))/n
    np.testing.assert_allclose((high-low)/2, expected, rtol=1e-13)
    if not np.isfinite(confidence_sequence([0.], [1e153])[1]).all():
        raise ValueError('Stable extreme-width calculation failed')
    for rho in (float('nan'), float('inf')):
        try:
            confidence_sequence([0.], [1.], rho=rho)
        except ValueError:
            pass
        else:
            raise ValueError('Nonfinite verifier parameter accepted')
    rejected = decision(np.zeros(30), np.ones(30))
    if not rejected['censored'] or rejected['first_certificate'] != 31:
        raise ValueError('Nonacceptance censoring differs')
    # All sensors are identical: duplicating them cannot multiply evidence.
    l, t, y = np.array([[0.], [1.]]), np.array([[1.], [2.]]), np.array([[2.], [3.]])
    first = paired_difference(l, t, y, 1.)
    repeated = paired_difference(np.tile(l, (1, 20)), np.tile(t, (1, 20)), np.tile(y, (1, 20)), 1.)
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    contract = read(code/'config/ectt_protocol_v1.json')
    execution = read(code/'config/ectt_execution_v1.json')
    simulate(store, execution['statistical_simulation'], contract['verifier'], progress)
    simulations = read(store/'result.json')
    result = {'status': 'INTERNAL_CHECKS_PASSED_INDEPENDENT_SIGNOFF_PENDING',
              'independent_review_complete': False, 'new_theorem_claim': False,
              'quadrature_cases': len(errors), 'maximum_relative_quadrature_error': max(errors),
              'synthetic_simulations': simulations,
              'guarantee': 'Observed-prefix average conditional paired MAE; no arbitrary future-test protection',
              'remaining_review': ['independent derivation and filtration sign-off',
                                   'audit frozen candidate/scale/label timing at deployment',
                                   'formal floating-point coverage beyond numerical regression checks']}
    write(store/'internal_review.json', result)
    return result
