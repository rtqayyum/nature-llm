# Traffic-transfer publication progress

**Last verified 23 November 2026.** Target: a measured six-domain result and an evidence-backed *Nature Communications* submission decision within 48 hours of cluster access. [Campaign plan](SPRINT_PLAN.md) · [Statistical review](SPRINT_STATISTICAL_REVIEW.md) · [Working manuscript](../paper/traffic_transfer_draft.md)

| Milestone | Evidence presently available | Status |
|---|---|---|
| Public reference assessment | 37 verified assets, eight evaluated checkpoints, five domains matched the three published S1 metrics at displayed precision. Six fresh single-domain fits. | Partial reproduction completed |
| Ordinary transfer and local controls | 96 local-baseline records. All 132 learned ordinary-transfer cases lost to their matched local model in MAE, RMSE and WMAPE. | Completed |
| Mechanism development | Anchored transfer improved 10/12 matched comparisons. Locked verifier selected local for all 18 pilot cases. Eight later-analyzed development windows yielded 216 policy replays. | Completed; development evidence |
| Held-out candidate preparation | 45 source-fit tasks produced 54 frozen anchored forecast pairs in collection 3100396. | Completed |
| Statistical review | 43 packet source files verified. Original synthetic tests and a separate high-precision numerical oracle exercised. Author requested AI-assisted internal review in lieu of outside review. | Completed internally |
| Campaign, results and manuscript | Analysis scripts, reference overlap comparator, full plan, restartable Slurm launcher, checked development table and manuscript draft published. | Prepared; private HPC run required |
| Held-out accuracy, harm, scripts and data for figures | Only an actual cluster aggregate can populate this row. | Awaiting Slurm receipt and aggregate output |
| Journal submission | Requires verified results, a defensible claim, complete declarations and author approval. | Pending |

## Evidence log

- 22 November: internal packet review identified the final pre-scoring inventory. No held-out comparison scores were in the packet.
- 23 November: took over the statistical review, checked the locked derivation, reran the tests and registered a separate empirically reviewed scoring campaign. The amendment's secondary tests were specified before observing new held-out outcomes. The working manuscript contains measured development findings.
- 23 November: recomputed development outcomes from job 3100357. On four Beijing windows the full-window empirical rule with margin training averaged 47.536 versus 50.435 local MAE (5.75% lower); ordinary suffix training with the same rule reached 47.372 (6.07% lower). All PEMS08 choices remained local. These results support running a matched attribution test, not selecting an overall winner. [Sanitized development scores](../results/development/window_scores.csv)

Next evidence to record: actual Slurm job ID, state, run revision, final case counts, primary paired effect, amended-policy performance, harm, cost, common-row coverage and any failures. Update this page at each verified change. The project owner has authorized recurring updates over the 48-hour window.
