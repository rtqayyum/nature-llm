"""Reviewable chronological consumer of one frozen candidate pair.

The caller supplies one matured truth row at a time. This module does not
open project data, choose candidate pairs or score a final test block.
"""
import numpy as np

from ectt_statistics import confidence_sequence


def replay(origins, local, transfer, scale, read_matured_label, *, rho=.01, delta=.05, eta=.01, minimum=12):
    origins = np.array(origins, copy=True)
    local, transfer = (np.array(values, dtype=np.float64, copy=True) for values in (local, transfer))
    if (origins.ndim != 1 or origins.dtype.kind not in 'iu' or len(origins) == 0
            or origins[0] < 0 or np.any(np.diff(origins.astype(np.int64)) != 1)
            or local.ndim != 2 or transfer.shape != local.shape or local.shape[0] != len(origins)
            or local.shape[1] == 0 or any(not np.isfinite(p).all() or np.any(p < 0) for p in (local, transfer))
            or not np.isfinite(scale) or scale < 1):
        raise ValueError('Require contiguous origins, finite physical forecasts and fixed support scale >=1')
    if (not np.isfinite(rho) or rho <= 0 or not np.isfinite(delta) or not 0 < delta < 1
            or not np.isfinite(eta) or eta < 0 or isinstance(minimum, bool)
            or not isinstance(minimum, (int, np.integer)) or minimum < 1):
        raise ValueError('Invalid confidence sequence settings')

    # Freeze all forecast-only bounds before obtaining any labels.
    bounds = np.abs(transfer-local).mean(axis=1)/scale
    if not np.isfinite(bounds).all():
        raise ValueError('Nonfinite predictable disagreement')
    differences, trace = [], []
    accepted = False
    for index, origin in enumerate(origins):
        truth = np.asarray(read_matured_label(int(origin)), dtype=np.float64)
        if truth.shape != local[index].shape or not np.isfinite(truth).all() or np.any(truth < 0):
            raise ValueError(f'Invalid matured label at origin {origin}')
        d = (np.abs(transfer[index]-truth).mean()-np.abs(local[index]-truth).mean())/scale
        differences.append(d)
        n = index+1
        lower, upper = confidence_sequence(differences, bounds[:n], rho, delta)
        accepted = n >= minimum and upper[-1] < -eta
        trace.append({'origin': int(origin), 'difference': float(d), 'bound': float(bounds[index]),
                      'lower': float(lower[-1]), 'upper': float(upper[-1]), 'accepted': bool(accepted)})
        if accepted:
            break
    first = len(trace) if accepted else len(origins)+1
    return {'accepted': bool(accepted), 'censored': not accepted, 'first_certificate': first,
            'selected_candidate': 'transfer' if accepted else 'local',
            'verification_origins_available': len(origins), 'verification_origins_read': len(trace),
            'verification_node_values_read': len(trace)*local.shape[1],
            'restricted_evidence_fraction': first/(len(origins)+1), 'trace': trace}
