# Traffic-transfer publication progress

**Last verified 23 November 2026.** Target: a measured six-domain result and an evidence-backed *Nature Communications* submission decision within 48 hours of cluster access. [Campaign plan](SPRINT_PLAN.md) · [Statistical review](SPRINT_STATISTICAL_REVIEW.md) · [Working manuscript](../paper/traffic_transfer_draft.md)

| Milestone | Evidence presently available | Status |
|---|---|---|
| Public reference assessment | 37 verified assets, eight evaluated checkpoints, five domains matched the three published S1 metrics at displayed precision. Six fresh single-domain fits. | Partial reproduction completed |
| Ordinary transfer and local controls | 96 local-baseline records. All 132 learned ordinary-transfer cases lost to their matched local model in MAE, RMSE and WMAPE. | Completed |
| Mechanism development | Anchored transfer improved 10/12 matched comparisons. Locked verifier selected local for all 18 pilot cases. Eight later-analyzed development windows yielded 216 policy replays. | Completed; development evidence |
| Held-out candidate preparation | 45 source-fit tasks produced 54 frozen anchored forecast pairs in collection 3100396. | Completed |
| Statistical review | 43 packet source files verified. Original synthetic tests and a separate high-precision numerical oracle exercised. Author requested AI-assisted internal review in lieu of outside review. | Completed internally |
| Campaign, results and manuscript | Analysis scripts, reference overlap comparator, full plan, restartable Slurm launcher, checked development table and manuscript draft published. | Submitted to UAEU Slurm as CPU job `3111109` |
| Held-out accuracy, harm, scripts and data for figures | Only an actual cluster aggregate can populate this row. | Awaiting completion evidence and aggregate output |
| Journal submission | Requires verified results, a defensible claim, complete declarations and author approval. | Pending |

## Evidence log

- 22 November: internal packet review identified the final pre-scoring inventory. No held-out comparison scores were in the packet.
- 23 November: took over the statistical review, checked the locked derivation, reran the tests and registered a separate empirically reviewed scoring campaign. The amendment's secondary tests were specified before observing new held-out outcomes. The working manuscript contains measured development findings.
- 23 November: recomputed development outcomes from job 3100357. On four Beijing windows the full-window empirical rule with margin training averaged 47.536 versus 50.435 local MAE (5.75% lower); ordinary suffix training with the same rule reached 47.372 (6.07% lower). All PEMS08 choices remained local. These results support running a matched attribution test, not selecting an overall winner. [Sanitized development scores](../results/development/window_scores.csv)

- 23 November: the HPC launcher verified campaign revision `a9ed0d77145c94d27f38f2c71c08642553738047` and Slurm accepted CPU job `3111109`. The submission receipt printed `results/traffic_publication/campaign-3111109.log` and the expected bundle `results/traffic_publication/a9ed0d77145c94d27f38f2c71c08642553738047_report.tar.gz`. Queue state, exit code and scores await the next cluster record.

Next evidence to record: the Slurm state and exit code, followed by verified case counts, primary paired effect, amended-policy performance, harm, cost, reference overlap and any failures in the aggregate report.

## Run or check the campaign

The frozen experiment code is [a9ed0d7](https://github.com/rtqayyum/nature-llm/commit/a9ed0d77145c94d27f38f2c71c08642553738047). A later documentation update can safely coexist with this immutable run identity. Paste the following block on the UAEU HPC. It requests one CPU job and prints its log and aggregate report paths. Running it again reports an existing job, completed archive, or a retry after a recorded failure.

```bash
module load slurm && cd /home/tariq.qayyum/nature-llm && \
rev=a9ed0d77145c94d27f38f2c71c08642553738047 && \
python3 work/code_versions/a99dae5b0644ac5baa7f9c7f3fe29b1ea21631c7/scripts/bootstrap_hpc.py --project "$PWD" --revision "$rev" --sync-only && \
python3 "work/code_versions/$rev/scripts/submit_traffic_campaign.py" --project "$PWD"
```

The aggregate report includes the held-out tables and a populated working manuscript. The figure script also renders the secondary paired-effect plot from that report. [Download the full research plan as a PDF](traffic_transfer_research_plan.pdf).
