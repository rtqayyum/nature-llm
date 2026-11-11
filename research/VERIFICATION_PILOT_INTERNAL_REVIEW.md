# Internal review of the frozen development verifier

The 18 cases collected by job 2992399 use anchored GPT-2 on PEMS08 and Beijing.
This review was prepared by the implementation collaborator. Independent S01
review is pending. The exploratory CPU replay describes the behaviour of the
locked policy on these development cases and cannot clear S01.

## Derivation and scope

For each origin, reverse triangle inequality bounds the scaled difference of
mean node absolute errors in the interval [-b_i,b_i], where b_i is the scaled
mean absolute forecast disagreement. Conditional Hoeffding uses interval length
2b_i, giving variance contribution b_i squared. This requires b_i, both
forecasts and the support-derived scale to be fixed at issue time. The preceding
outcome must belong to the history available by the next issue time.

The normal mixture integral with prior precision rho yields
sqrt(rho/(V+rho)) times exp(S squared divided by 2(V+rho)). Setting this to
1/delta and dividing the boundary by the prefix length gives the width used in
the implementation. The estimand is the average conditional paired risk over
the verification prefix. Test accuracy, acceptance and harm are empirical.
The per-stream delta does not supply simultaneous coverage across arms and seeds.

## Implementation review

- Candidate validation checks artifact hashes, frozen checkpoint identifiers,
  matching local forecasts and scalers, and contiguous origin ledgers.
- The replay uses the pinned rho, delta, margin and minimum prefix. The callback
  reads only the next matured verification row and stops at a strict crossing.
- The exploratory worker saves all decisions before reading test truth. It
  uses local predictions at all test origins for each nonaccepted stream.
- The threshold and budget remain locked. The report distinguishes actual
  labels consumed from the restricted crossing time with censored cases.
- The floating-point check permits a small tolerance on the computed envelope.
  A numerical implementation review must assess whether finite-precision error
  can affect a crossing near the threshold.
- Independently check the archived forecasts' issue-time information sets,
  actual one-step outcome availability, and the intended multiplicity scope
  before drawing a formal conclusion from this verifier.

The CPU stage `shared-transfer-ectt-exploratory-replay` produces a descriptive
report using frozen artifacts. Its result explicitly records S01 as pending.
The reviewed scoring stage continues to require the independent disposition.
