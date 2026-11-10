# Independent S01 review packet

Status: prepared for review. Neither this document nor internal simulations
constitute independent sign-off. The reviewer must record identity, date,
reviewed code/contract hashes, findings and an explicit disposition.

For one frozen local/transfer pair, let F_i contain observations available at
forecast issue time. The forecasts, fixed support-derived scale s >= 1, and
b_i = mean_nodes(abs(T_i-L_i))/s must be F_i measurable. At one-step label
maturity set D_i = (MAE(T_i,y_i)-MAE(L_i,y_i))/s and
mu_i = E[D_i | F_i]. Reverse triangle inequality gives |D_i| <= b_i, without
requiring independent nodes or bounded counts. Review the filtration between
successive issue times, labels appearing in rolling histories and feedback.

For every real lambda, conditional Hoeffding gives
E[exp(lambda*(D_i-mu_i)) | F_i] <= exp(lambda^2*b_i^2/2).
Consequently the product exponential with S_n=sum(D_i-mu_i) and
V_n=sum(b_i^2) is a nonnegative supermartingale. Mixing lambda with the
N(0, 1/rho) density gives
sqrt(rho/(V_n+rho))*exp(S_n^2/(2*(V_n+rho))).
Ville's inequality at 1/delta yields the two-sided time-uniform width
sqrt((V_n+rho)*log((V_n+rho)/(rho*delta^2)))/n.
Check the constants, predictable envelope, conditional mgf and mixture steps.
These instantiate existing confidence-sequence results cited in RELATED_WORK;
no new concentration theorem is claimed.

The interval covers the observed prefix's average conditional paired risk.
It does not certify each origin, future test risk or arbitrary drift. Accept
the transfer candidate at the first n >= 12 whose upper bound is below -0.01,
with locked rho=0.01 and delta=0.05 per pair/stream. No acceptance is recorded
as censored at available_prefix+1. Different runs do not share an automatic
study-wide 95% claim. Multiple candidates in a single decision need an explicit
simultaneous error allocation. Do not choose rho, delta or candidates after
looking at their verification outcomes.

Required independent checks:

- Check support/verification/test boundaries, warmup and counted label budget.
- Freeze candidate weights, scalers and support-derived s before verification.
- Audit chronological issue/label-maturity ordering and each bound's inputs.
- Ensure no origin/node duplication multiplies statistical evidence.
- Audit optional stopping, strict acceptance, minimum length and censoring.
- Check candidate multiplicity, seeds and the scope of uncertainty statements.
- Review finite-precision arithmetic, under/overflow and tolerance implications;
  numerical regression does not prove formal floating-point coverage.
- Inspect source training surrogate separately. Its source labels optimize
  weights, so its training margin is not an out-of-sample certificate.
- Require exact later deployment consumer hashes before traffic scoring.

The supplied CPU tests exercise numerical integration, predictable-envelope
synthetic streams and a callback consumer that stops label reads at acceptance.
They cannot demonstrate that actual traffic data access meets this filtration.
Do not clear S01 from a successful CPU scheduler state or an unsigned template.
