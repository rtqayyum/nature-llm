# Sequential comparison derivation and review status

Status: S01 INDEPENDENT REVIEW PENDING. This derivation and the synthetic code
checks do not constitute an independent review or a new theorem.

For one frozen target candidate pair, predictions and b_i must be measurable
before the next one-step label y_i. Define D_i as the difference of node-mean
absolute errors divided by the fixed support-validation scale s. Reverse
triangle inequality gives |D_i| <= b_i = mean_node|T_i-L_i|/s. It does not
require bounded labels or independent nodes.

Let mu_i = E[D_i | F_(i-1)]. Conditional on F_(i-1), D_i-mu_i lies in an
interval of length 2b_i. Hoeffding's lemma therefore gives

    E[exp(lambda*(D_i-mu_i)) | F_(i-1)] <= exp(lambda^2*b_i^2/2).

For S_n=sum(D_i-mu_i), V_n=sum(b_i^2), the nonnegative process
exp(lambda*S_n-lambda^2*V_n/2) is a supermartingale for each fixed lambda.
Mix lambda against N(0,1/rho). The integral is

    M_n = sqrt(rho/(V_n+rho))*exp(S_n^2/(2*(V_n+rho))).

Ville's inequality bounds the probability of ever crossing 1/delta by delta.
Solving that boundary yields

    |S_n|/n <= sqrt((V_n+rho)*log((V_n+rho)/(rho*delta^2)))/n.

This is the existing Gaussian-mixture Hoeffding boundary cited in the design.
Its center estimates the average forecast-origin conditional loss difference
over that observed prefix. It says nothing by itself about arbitrary future
blocks, per-origin safety, RMSE, WMAPE or a whole multi-candidate study's error.

Independent review must check:

- The filtration matches actual data and feedback availability.
- Source/target models and statistical scale are frozen before target E labels.
- No across-origin attention, adaptive refitting or unallocated candidate search.
- Predictable bounds are computed before labels, in the same units as D_i.
- Floating-point implementation, stable logarithms and all stopping paths.
- The source training margin is only a surrogate because theta sees its labels.
- Multiple streams are reported separately, without a study-wide 95% claim.
- Delayed/multi-horizon extensions have a separate argument before execution.

S02 simulates fixed null and alternative processes with known conditional
means and predictable varying envelopes. It uses 2000 streams of 256 origins,
seed 20261102. Its admission tolerance is delta plus four binomial standard
errors. Such tests can expose bugs; passing them cannot prove general coverage.

Primary sources: Choe and Ramdas, https://arxiv.org/abs/2110.00115;
Howard et al., https://doi.org/10.1214/20-AOS1991.
