"""Strict decisions on archived binary forecasts under the written decimal protocol."""
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction


# Consecutive decimal bounds on pi. The regression test verifies them with
# rational alternating-series bounds for Machin's identity.
PI_LO = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
PI_HI = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')


def binary_units(row):
    """Represent stored floats as integers with a common power-of-two denominator."""
    ratios = [value.as_integer_ratio() for value in row.tolist()]
    bits = max(denominator.bit_length() for _, denominator in ratios)
    units = [numerator << (bits - denominator.bit_length()) for numerator, denominator in ratios]
    return units, bits


def paired_row(local, transfer, scale, origin, matured):
    """Compute the predictable envelope before requesting this origin's label."""
    left, left_bits = binary_units(local)
    right, right_bits = binary_units(transfer)
    bits = max(left_bits, right_bits)
    left = [value << (bits - left_bits) for value in left]
    right = [value << (bits - right_bits) for value in right]
    divisor = len(left) * (1 << (bits - 1)) * scale
    envelope = sum(abs(r - l) for l, r in zip(left, right)) / divisor

    actual, label_bits = binary_units(matured(origin))
    common_bits = max(bits, label_bits)
    forecast_shift, label_shift = common_bits - bits, common_bits - label_bits
    difference = 0
    for l, r, y in zip(left, right, actual):
        y <<= label_shift
        difference += abs((r << forecast_shift) - y) - abs((l << forecast_shift) - y)
    return difference / (divisor * (1 << forecast_shift)), envelope


def decimal_ratio(rational):
    return Decimal(rational.numerator) / Decimal(rational.denominator)


def outward(value, up):
    # Decimal exp, ln and sqrt round to nearest even, including inside a
    # directed context. Widen their results by one representable decimal.
    return value.next_plus() if up else value.next_minus()


def two_sided_endpoint(total, variance, n, rho, delta, up):
    adjusted = variance + rho
    log_term = outward(decimal_ratio(adjusted / (rho * delta**2)).ln(), up)
    width = outward((decimal_ratio(adjusted) * log_term).sqrt(), up) / n
    upper = decimal_ratio(total / n) + width
    return upper, width


def half_normal_ratio(gain, variance, rho, delta, up, precision):
    """Bound delta times the positive half-normal mixture, for positive gain."""
    adjusted = variance + rho
    q = decimal_ratio(gain**2 / (2 * adjusted))
    weight = outward(decimal_ratio(delta**2 * rho / adjusted).sqrt(), up)
    # exp(q)*erf(sqrt(q)) = 2*sqrt(q/pi)*sum(t_k),
    # t_0=1 and t_(k+1)/t_k = 2q/(2k+3). All terms are nonnegative.
    total = term = Decimal(1)
    twice_q = 2*q
    denominator = 3
    tolerance = Decimal(1).scaleb(-precision)
    while True:
        ratio = twice_q / denominator
        term = term * ratio
        if ratio <= Decimal('0.5') and term <= total*tolerance:
            # Ratios decrease, so the remaining sum is at most 2*term.
            if up:
                total += 2*term
            break
        total += term
        denominator += 2
    root = outward((q / (PI_LO if up else PI_HI)).sqrt(), up)
    return weight * (outward(q.exp(), up) + 2*root*total)


def sequential_decision(rule, total, variance, n, rho, delta, margin):
    gain = -total - n*margin
    one_sided = rule == 'one_sided_mixture'
    if one_sided and gain <= 0:
        # The mixture is at most 1 and 1/delta > 1.
        return False, {'statistic': 0., 'statistic_is_upper_bound': True}
    cutoff = Fraction(1) if one_sided else -margin
    for precision in (40, 80):
        endpoints = []
        for up in (False, True):
            rounding = ROUND_CEILING if up else ROUND_FLOOR
            with localcontext(Context(prec=precision, rounding=rounding)):
                if one_sided:
                    bound = half_normal_ratio(gain, variance, rho, delta, up, precision)
                else:
                    bound, width = two_sided_endpoint(total, variance, n, rho, delta, up)
                endpoints.append(bound)
        low, high = endpoints
        if one_sided:
            accepted, declined = low > cutoff, high <= cutoff
        else:
            accepted, declined = high < cutoff, low >= cutoff
        if accepted or declined:
            break

    if one_sided:
        # The legacy trace reports log evidence. Only the enclosed ratio decides.
        with localcontext(Context(prec=precision)):
            fields = {'statistic': float((high/decimal_ratio(delta)).ln()),
                      'evidence_times_delta_interval': [str(low), str(high)]}
    else:
        with localcontext(Context(prec=precision, rounding=ROUND_FLOOR)):
            lower = decimal_ratio(total / n) - width  # width from the upward pass
        fields = {'lower': float(lower), 'upper': float(high),
                  'upper_endpoint_interval': [str(low), str(high)]}
    fields['unresolved_at_precision_limit'] = not (accepted or declined)
    return accepted, fields


def replay(rule, origins, local, transfer, scale, matured, settings, limit):
    """Consume only the requested chronological prefix, once per policy."""
    scale = Fraction.from_float(float(scale))
    rho, delta, margin = (Fraction(str(settings[k])) for k in ('rho', 'delta', 'eta'))
    minimum = settings['minimum']
    if scale < 1 or rho <= 0 or not 0 < delta < 1 or margin < 0 or minimum < 1:
        raise ValueError('Invalid frozen verification settings')
    fixed = rule in ('fixed_budget_mean', 'fixed_two_thirds_mean')
    if not fixed and rule not in ('locked_two_sided', 'one_sided_mixture'):
        raise ValueError('Unknown held-out policy')
    total = variance = Fraction(0)
    trace = []
    accepted = False
    for n, origin in enumerate(origins[:limit], start=1):
        diff, envelope = paired_row(local[n-1], transfer[n-1], scale, int(origin), matured)
        total += diff
        variance += envelope**2
        if fixed:
            mean_gain_above_margin = -total/n - margin
            crossed = mean_gain_above_margin > 0
            fields = {'statistic': float(mean_gain_above_margin)}
            if n == limit:
                fields['exact_gain_above_margin'] = str(mean_gain_above_margin)
            accepted = n == limit and n >= minimum and crossed
        else:
            crossed, fields = sequential_decision(rule, total, variance, n, rho, delta, margin)
            accepted = n >= minimum and crossed
        trace.append({'origin': int(origin), 'difference': float(diff), 'bound': float(envelope),
                      **fields, 'accepted': accepted})
        if accepted:
            break
    read = len(trace)
    result = {'accepted': accepted, 'trace': trace, 'verification_origins_read': read}
    if rule == 'locked_two_sided':
        first = read if accepted else len(origins)+1
        result.update(censored=not accepted, first_certificate=first,
                      selected_candidate='transfer' if accepted else 'local',
                      verification_origins_available=len(origins), verification_node_values_read=read*local.shape[1],
                      restricted_evidence_fraction=first/(len(origins)+1))
    return result
