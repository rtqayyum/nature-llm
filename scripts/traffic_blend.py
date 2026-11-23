"""Predeclared scalar forecast combinations for the 23 November amendment."""
from fractions import Fraction
import numpy as np

QUARTERS = (0, 1, 2, 3, 4)


def fit_weight(local, transfer, truth, scale):
    """Minimize prefix MAE/scale + 0.01*w on the quarter grid, ties toward local."""
    from heldout_thresholds import binary_units
    if local.shape != transfer.shape or truth.shape != local.shape or not len(truth):
        raise ValueError('Unmatched verification panel')
    if not np.isfinite(truth).all() or np.any(truth < 0) or not np.isfinite(scale) or scale < 1:
        raise ValueError('Invalid verification inputs')
    losses = [Fraction(0) for _ in QUARTERS]
    for left, right, y in zip(local, transfer, truth):
        encoded = [binary_units(row) for row in (left, right, y)]
        bits = max(b for _, b in encoded)
        l, r, a = [[v << (bits-b) for v in units] for units, b in encoded]
        residual = [4*(v-target) for v, target in zip(l, a)]
        step = [v-u for u, v in zip(l, r)]
        denominator = 4 * (1 << (bits-1))
        for k in QUARTERS:
            losses[k] += Fraction(sum(abs(base+k*delta) for base, delta in zip(residual, step)), denominator)
    unit = Fraction.from_float(float(scale))*truth.size/400
    return min(QUARTERS, key=lambda k: losses[k]+unit*k)/4


def forecast(local, transfer, weight):
    if weight == 0:
        return local
    if weight == 1:
        return transfer
    return (1-weight)*local + weight*transfer
