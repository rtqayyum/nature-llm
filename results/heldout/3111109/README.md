# Six-domain traffic-transfer results from job 3111109

On 23 November 2026 the researcher supplied the final Slurm accounting record and the campaign's printed summary. Slurm recorded COMPLETED, exit code 0:0, elapsed 00:01:12, for the [pinned campaign](https://github.com/rtqayyum/nature-llm/commit/a9ed0d77145c94d27f38f2c71c08642553738047). The data tables in this directory transcribe the precision printed in that summary. The full-precision, per-seed bundle was not available for this update. The review was internal and AI-assisted.

## Registered and secondary results

The registered comparison of margin versus prefix training under the locked two-sided rule produced +0.00 percentage points of local MAE at displayed precision. The locked margin arm matched local MAE in all six domain averages to three decimal places. A domain average alone cannot identify individual choices or verification time.

The prospective secondary weighting rule lowered equal-domain MAE by 0.79 percentage points of local MAE relative to local. The descriptive 95% paired circular block interval with 144-row blocks spans 0.74–0.83 points of improvement on this panel. The practical benchmark for the paper was 2 points. Relative to the two-thirds binary rule with the same margin candidate and label budget, the weighted rule lowered MAE by 0.36 points. Holding weighting fixed, its improvement over ordinary prefix training was 0.58 points and over ordinary suffix training was 0.11 points. The comparison with a blind half-weight gives 2.47 points, but that control consults zero verification labels.

Using rounded domain MAEs, improvement relative to local was 3.12% on NYC taxi, 1.25% on Shanghai Metro, and 0.38% on Hangzhou Metro. PEMS03, PEMS04 and NYC bike tied local at reported precision. Omitting both NYC targets leaves an approximate 0.41-point reduction across the other four domains. NYC taxi accounts for roughly two-thirds of the six-domain reduction under equal domain weighting. The 144-row intervals describe these domains; they do not quantify uncertainty over new cities.

Only 4 of 18 unconditional margin candidates improved test MAE. For all four the locked boundary could have been crossed along the realized forecast path under hypothetical most-favourable labels. This distinguishes mathematical reachability from the evidence actually observed. The log does not report all realized acceptances, their timing or material harm.

## Released checkpoints

On common rows with equal physical observations, Zhang et al.'s released checkpoint had lower MAE in all six domains. The six overlap sizes range from 304 to 2,608 rows and cover approximately 29%–92% of our test rows. The largest gap was on NYC taxi: 73.851 for the weighted forecast versus 16.743 for the checkpoint. The controlled comparison with the study's B03 adaptation asks a different question. Both systems there use the study's shared target budget and causal protocol. The margin-trained weighted method improved on B03 with the same weighting procedure by 0.69 points. Different training access, scaling and batch behavior limit the interpretation of the original-checkpoint comparison.

[Domain and common-row scores](domains.csv) · [All ten paired policy contrasts](policy_contrasts.csv) · [Frozen protocol](../../../research/SPRINT_PLAN.md)

## Remaining verification

The expected archive is `a9ed0d77145c94d27f38f2c71c08642553738047_report.tar.gz`. Check its receipt, hashes and row counts, realized decisions and selected weights, per-seed harm, label costs, RMSE, WMAPE and 72/288-row sensitivity before finalizing the source-data supplement. All six study domains now have exposed outcomes. Any new model selected using this panel needs a fresh evaluation cohort.
