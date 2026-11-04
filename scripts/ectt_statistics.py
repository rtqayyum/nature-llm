"""Existing Gaussian-mixture Hoeffding construction; not a new theorem."""
import numpy as np
from ectt_common import write


def confidence_sequence(difference, envelope, rho=.01, delta=.05):
    d, b = np.asarray(difference, dtype=np.float64), np.asarray(envelope, dtype=np.float64)
    if d.ndim != 1 or b.shape != d.shape or not np.isfinite(d).all() or not np.isfinite(b).all():
        raise ValueError('Finite paired one-dimensional sequences are required')
    if not np.isfinite(rho) or not np.isfinite(delta) or rho <= 0 or not 0 < delta < 1 or np.any(b < 0) or np.any(np.abs(d) > b+1e-12):
        raise ValueError('Invalid predictable bound, rho or delta')
    n = np.arange(1, len(d)+1, dtype=np.float64)
    # Separate square roots and log differences avoid overflow in the direct
    # ratio/product even when the mathematically equivalent width is finite.
    try:
        with np.errstate(over='raise', invalid='raise', divide='ignore'):
            variance = np.cumsum(b*b)
            log_ratio = np.logaddexp(np.log(variance), np.log(rho))-np.log(rho)-2*np.log(delta)
            width = (np.sqrt(variance+rho)/n)*np.sqrt(log_ratio)
            mean = np.cumsum(d)/n
    except FloatingPointError as exc:
        raise ValueError('Sequential calculation exceeds finite FP64 range') from exc
    if not np.isfinite(width).all() or not np.isfinite(mean).all():
        raise ValueError('Nonfinite sequential calculation')
    return mean-width, mean+width


def decision(difference, envelope, rho=.01, delta=.05, eta=.01, minimum=12):
    if not np.isfinite(eta) or eta < 0 or isinstance(minimum, bool) or not isinstance(minimum, (int, np.integer)) or minimum < 1:
        raise ValueError('Finite nonnegative eta and positive integer minimum required')
    lower, upper = confidence_sequence(difference, envelope, rho, delta)
    indices = np.flatnonzero((np.arange(len(upper))+1 >= minimum) & (upper < -eta))
    tau = int(indices[0]+1) if len(indices) else len(upper)+1
    return {'accepted': bool(len(indices)), 'first_certificate': tau,
            'verification_origins_available': len(upper), 'censored': not bool(len(indices)),
            'restricted_evidence_fraction': tau/(len(upper)+1),
            'upper': upper.tolist(), 'lower': lower.tolist(),
            'guarantee_scope': 'Observed prefix average conditional paired MAE only; no arbitrary future-test guarantee'}


def paired_difference(local, transfer, truth, scale):
    l, t, y = [np.asarray(v, dtype=np.float64) for v in (local, transfer, truth)]
    if l.shape != t.shape or l.shape != y.shape or l.ndim != 2 or l.shape[1] == 0 or not np.isfinite(scale) or scale < 1:
        raise ValueError('Expected matching origins x nodes and support scale >=1')
    if not all(np.isfinite(v).all() for v in (l, t, y)):
        raise ValueError('Nonfinite paired forecasts')
    return (np.abs(t-y).mean(1)-np.abs(l-y).mean(1))/scale, np.abs(t-l).mean(1)/scale


def simulate(store, settings, verifier, progress):
    rng = np.random.default_rng(settings['seed'])
    streams, origins = settings['streams'], settings['origins']
    results = []
    for scenario in ('rademacher_null', 'predictable_envelope_null', 'negative_mean_alternative'):
        crossed, accepted = 0, 0
        for i in range(streams):
            signs = rng.choice([-1., 1.], size=origins)
            b = np.ones(origins)
            if scenario == 'predictable_envelope_null':
                b[1:] = .2+.8*(signs[:-1] > 0)
            d = signs*b if scenario != 'negative_mean_alternative' else np.where(signs < 0, -1., .2)
            lower, upper = confidence_sequence(d, b, verifier['rho'], verifier['delta_per_stream'])
            # The simulated null conditional mean is exactly zero. Under the
            # alternative it is -0.4; no coverage assertion is made about
            # future deployment risk.
            mu = -.4 if scenario == 'negative_mean_alternative' else 0.
            crossed += int(np.any((lower > mu) | (upper < mu)))
            accepted += int(decision(d, b, verifier['rho'], verifier['delta_per_stream'])['accepted'])
            if (i+1) % 250 == 0:
                progress(phase='SYNTHETIC_VERIFIER', scenario=scenario, completed_streams=i+1, total_streams=streams)
        rate = crossed/streams
        tolerance = float(verifier['delta_per_stream']+4*np.sqrt(verifier['delta_per_stream']*(1-verifier['delta_per_stream'])/streams))
        results.append({'scenario': scenario, 'streams': streams, 'origins': origins,
                        'coverage_failure_fraction': rate, 'accepted_fraction': accepted/streams,
                        'monte_carlo_admission_limit': tolerance, 'passed': bool(rate <= tolerance)})
    result = {'stage': 'ectt-stat-check', 'simulation_settings': settings, 'scenarios': results,
              'execution_kind': 'SYNTHETIC', 'training_performed': False, 'model_inference_performed': False,
              'S01_independent_mathematical_review': 'PENDING_NOT_CLEARED_BY_SIMULATION',
              'scientific_status': 'METHOD_NOT_EVALUATED'}
    write(store / 'result.json', result)
    (store / 'SUMMARY.md').write_text('# Synthetic sequential-verifier checks\n\n'+
        '\n'.join('%s: coverage failures %.4f, accepted %.4f, pass=%s' %
        (r['scenario'], r['coverage_failure_fraction'], r['accepted_fraction'], r['passed']) for r in results)+
        '\n\nSynthetic streams only. No traffic accuracy or future-risk claim.\n'+
        'S01 independent mathematical review remains pending.\n')
    if not all(r['passed'] for r in results):
        raise ValueError('Synthetic verifier admission failed')
    return result
