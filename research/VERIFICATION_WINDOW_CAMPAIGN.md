# ECTT development window campaign

Exploratory replay on PEMS08 and Beijing using the 18 archived B03 source
checkpoints from job 2992398. Every source checkpoint excludes its target
domain. Each window fits a new residual head for each archived objective
using the same support rows, optimizer and seed. It also fits the existing
persistence, ridge and compact local comparator recipe on that support.
The local selection and scale are shared by all objectives and rules in each
window and seed. Final forecasts retain their pinned histories and clipping.

Windows are fixed by index before inspecting any outcomes. Each entry
lists support start, verification start, test start and test end, using
half-open intervals. Support and verification retain the original 5% budget
row counts. Test length equals verification length. The locked 12-row history
is available by design. The new windows precede the earliest original target
support interval, including that of the 10% condition.

| Domain | Window boundaries |
|---|---|
| PEMS08 | 8572/8929/9286/9643, 9643/10000/10357/10714, 10714/11071/11428/11785, 11785/12142/12499/12856 |
| Beijing | 864/900/936/972, 972/1008/1044/1080, 1080/1116/1152/1188, 1188/1224/1260/1296 |

These are disjoint chronological target windows for the archived pilot.
Other historical work may have used rows from these datasets. In particular,
the datasets can occur in source runs for other target folds. Results are
development evidence, with three paired seeds per window.

Compare the locked two-sided normal-mixture rule, a fixed-budget decision
using the final empirical mean, and a one-sided normal-mixture rule. All use
the same scale, required gain of 0.01 and maximum verification allocation.
Sequential rules use the existing minimum of 12 rows. The fixed-budget rule
uses every available verification row. For the one-sided rule, set
S to the sum of negative scaled paired differences minus the gain margin per
row and V to the sum of the squared predictable bounds. At an eligible prefix
compare the half-normal mixture to 1/delta using the original rho and delta.
The formula follows Proposition 6 of Howard et al. (2021),
https://arxiv.org/abs/1810.08240. No thresholds or windows are tuned on outcomes.
The comparison is descriptive and independent S01 review is pending.

The Slurm campaign uses one CPU preflight, 24 GPU tasks with four active at
most, and a CPU collector. GPU tasks freeze each window's candidates and all
nine decisions before scoring test truth. Every accepted candidate uses
transfer across the test block. Every other case uses the local comparator.
Report test MAE, RMSE and WMAPE, the support and verification labels consumed,
and useful and harmful acceptances. Failed tasks are listed in the collector
report. Completed artifacts are reused on compatible retries.
