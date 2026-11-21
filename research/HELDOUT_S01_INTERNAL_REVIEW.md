# Internal statistical review of the held-out GPT-2 scorer

**Disposition on 22 November 2026: request a numerical correction before formal scoring.** This review was conducted by the implementation collaborator. Independent S01 disposition remains pending.

The review covers scoring revision `2c932bf74c1cd403902a2a66ea593324e71c8e75` for collector 3100396. Seven held-out preparation and scoring tests passed locally. The review also exercised two synthetic threshold cases that those tests missed. The production archive was identified by its HPC path and the reported review identity beginning `7a03ac367ea6`. Its bytes and the 54 production candidate pairs were unavailable to this reviewer. No real verification or test outcomes were used.

## Findings

1. **The primary sequential rule can declare a crossing on the wrong side of the margin.** At origin 12, repeated local forecasts of 0.0807838110213078, transfer forecasts and truth of zero, and a support scale of 1 yield an FP64 upper bound of -0.010000000000000009. The implementation accepts. Evaluating the same formula using 85-digit decimal arithmetic and the exact binary64 input forecast gives an upper bound of approximately -0.00999999999999998937 under the written decimal parameters. It also lies above the cutoff when the binary64 parameter values are used. A strict crossing at this origin is incorrect.
2. **The two fixed-look policies can accept an exact tie.** Use local 10, transfer 8 and support scale 100. For a two-thirds look after 22 of 33 available rows, verification truths of eleven 9s followed by eleven 8s give a true normalized difference of exactly -0.01. The scorer accepts this order and declines the reversed order. The fixed full-budget rule gives the same order dependence at an exact tie. A fixed-look empirical mean over an identical multiset has one decision in exact arithmetic.
3. **Interpret label counts as labels consulted by the decision rule.** The input history for each forecast uses the preceding 12 observed rows. In Hangzhou the two-thirds rule reads 22 of 33 verification rows. All 11 later rows still appear among the 12 inputs to the first test forecast. The design can measure savings in verification work while assuming continuing access to ordinary traffic observations. These data do not demonstrate fewer sensor measurements.

## Checks that passed

For one frozen pair with a support-fixed positive scale, the absolute paired MAE difference at one origin is bounded by mean absolute forecast disagreement divided by that scale. This follows by the reverse triangle inequality at each node and averaging across nodes. With that bound measurable at issue time and each preceding response available by the next issue, the conditional Hoeffding variance contribution is its square. The Gaussian-mixture formula in `ectt_statistics.py` matches the real-arithmetic two-sided boundary. The interpretation concerns average conditional paired risk through the observed verification prefix, one stream at a time. No future-block harm or simultaneous coverage guarantee is implied. See Howard, Ramdas, McAuliffe and Sekhon, *Time-uniform, nonparametric, nonasymptotic confidence sequences*, equation (3.7).

The input generators examined use history ending before the predicted row. The scorer enforces chronological label access and the rule-specific stopping point. It writes all verification decisions before calculating test accuracy. It counts actual verification labels separately from the restricted first-acceptance statistic, averages seeds within domain, and pairs forecasts across policies. Deployment-time availability of traffic observations is an assumption of this ordered-row benchmark. The 54 pairs and four rules give 216 policy outcomes; these share data and do not give 216 independent replications.

## Reproduce the threshold cases

Run from the reviewed GitHub snapshot using Python and NumPy. This uses synthetic inputs only.

```bash
PYTHONPATH=scripts python3 - <<'PY'
from decimal import Decimal as D, getcontext
import numpy as np
from frozen_verifier_replay import replay
from heldout_candidates import TWO_THIRDS, policy_decision

getcontext().prec = 85
opts = dict(rho=.01, delta=.05, eta=.01, minimum=12)
b = 0.0807838110213078
n = 12
result = replay(np.arange(n), np.full((n, 1), b), np.zeros((n, 1)),
                1., lambda origin: np.zeros(1), **opts)
exact_b = D.from_float(b)
rho, delta = D('.01'), D('.05')
variance = n * exact_b**2
upper = -exact_b + ((variance + rho) * ((variance + rho) / (rho * delta**2)).ln()).sqrt() / n
print(result['accepted'], result['trace'][-1]['upper'], upper)

# With 33 available rows the two-thirds policy consumes precisely 22.
# The full-budget example uses 36 rows, half with each truth value.
for rule, available, used in [(TWO_THIRDS, 33, 22), ('fixed_budget_mean', 36, 36)]:
    prefix = [9.] * (used//2) + [8.] * (used//2)
    for sequence in (prefix, prefix[::-1]):
        truth = np.array(sequence + [9.] * (available-used))
        result = policy_decision(rule, np.arange(available), np.full((available, 1), 10.),
                                 np.full((available, 1), 8.), 100., lambda i: truth[i:i+1], opts)
        print(rule, result['accepted'], result['verification_origins_read'])
PY
```

## Correction and disposition

Preserve the fits, frozen forecasts, decision margin, evaluation panel and label budgets. Implement an error-aware comparison for strict threshold decisions. Resolve ambiguous cases from the frozen numeric inputs using exact or conservatively bounded higher-precision arithmetic. Keep the local forecast when a strict crossing cannot be established. Cover ties, both sides of the boundary, permutations for fixed looks, and the one-sided rule's decision threshold with regression tests. Pin the corrected scorer and supply its hash to the independent reviewer before opening held-out scores. These small synthetic errors establish an implementation defect. Their prevalence or effect on the real study is unknown.
