# A 48-hour plan for the traffic-transfer paper

**Goal for 23–25 November 2026.** Finish a reproducible six-domain test of when cross-city traffic forecasts help, prepare the corresponding manuscript and make a journal-readiness decision from the measured effects. The delivery clock starts with access to the private forecast store. Publication in *Nature Communications* is the journal target. Editorial review and acceptance have their own timeline.

## Research question

With a 5% target-label budget and a strong local forecast, can a source-trained traffic model deliver a correction whose benefit can be identified before deployment? A second question asks whether limiting the size of the correction retains more of that benefit. An operator can use the answer even when a foundation model behaves differently in different cities.

## Why change the experiment

The original comparison across the eight public datasets and 132 learned ordinary-transfer cases establishes the strength of the local baseline. The development margin-trained candidate beat the local average in Beijing (47.373 vs 50.670 MAE) and lost on PEMS08 (16.311 vs 14.717). The locked verifier selected zero of 18 candidates. Across four later-analyzed Beijing development windows the full-window empirical switch achieved 47.536 with the margin objective and 47.372 with ordinary suffix training against 50.435 locally. Eight beneficial and one harmful switch occurred under margin training. Candidate quality, detectability and the loss incurred when switching are distinct quantities.

Wang et al. document simple combinations as strong forecasting controls. Choe and Ramdas and Howard et al. establish sequential forecast comparison and time-uniform bounds. OpenCity demonstrates the relevance of cross-region transfer; Tan et al. show why a language-model contribution needs an architectural control. Zhang et al. already demonstrate multimodal traffic forecasting. These works motivate testing correction size, evidence use and attribution under common inputs. A five-weight search, forecast combination, transformer substitution or confidence sequence alone would be insufficient novelty. [Reference framework](https://doi.org/10.1038/s41467-026-73610-2) · [Forecast combinations](https://doi.org/10.1016/j.ijforecast.2022.11.005) · [Forecast comparison](https://doi.org/10.1287/opre.2021.0792) · [Confidence sequences](https://doi.org/10.1214/20-AOS1991) · [OpenCity](https://arxiv.org/abs/2408.10269) · [LLM ablations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html)

## Changes frozen before six-domain method scoring

1. Preserve the original primary comparison: margin versus prefix-MAE training under the locked two-sided rule. Retain all four original policy outputs and the suffix-MAE control. The qualified internal statistical review replaces the external-review dependency for this amended evaluation. The original scorer and its independent-review field retain their historical meaning.
2. Introduce a **secondary**, scalar partial correction. For frozen local forecast L and transfer forecast T, deploy L + w(T − L). At ceil(2N/3) verification rows, choose w from {0, .25, .5, .75, 1} to minimize mean physical MAE on that prefix divided by the fixed support scale, plus 0.01w. Ties choose the smaller weight. Fitting only the endpoints gives the existing two-thirds switch's strict 0.01 margin. The grid therefore tests the value of intermediate corrections at identical label cost. No per-city hyperparameters or per-node model selection.
3. Compare the learned weight against local, always-transfer, blind half weight, half of the accepted full-window correction, and the original binary rules. Apply the same grid to suffix-, prefix- and margin-trained candidates, ordinary B03 (the shared-protocol adaptation of LLM-UTP), and ordinary B04 (Qwen2.5). Equal blending consumes zero verification labels; support labels still count.
4. Explain zero switches. Replay the registered boundary using the most favorable possible label at each observed forecast pair. The reverse triangle inequality makes the largest possible row gain equal to the mean absolute difference between the two forecasts divided by the support scale. Using T itself as hypothetical truth achieves the bound for that pair. If even this path cannot cross, the observed forecast disagreements and budget make certification impossible on that path. If the boundary can cross but does not, realized evidence was insufficient. These are retrospective pathwise diagnostics on fixed forecast pairs. Report misses and future harm beside them.
5. Run two reference comparisons. The controlled comparison uses B03 with identical support budget and causal origins. The descriptive comparison uses job 607564's original checkpoint predictions only at identical target row IDs with exactly equal physical truth; report overlap size, both denominators, MAE, RMSE and WMAPE. Different training access and preprocessing remain part of the interpretation. An absent archive, empty overlap or unequal observations results in a visible missing-comparison status. S1 table values alone cannot substantiate a head-to-head gain.

## Registered matrix and analysis

| Item | Specification |
|---|---|
| Development only | PEMS08 and Beijing; four windows per domain, three paired seeds. Their outcomes informed this amendment. |
| Six-domain panel | PEMS03, PEMS04, Shanghai, Hangzhou, NYC bike, NYC taxi. The two NYC targets exclude both domains during source training. Existing ordinary baselines were previously viewed. New held-out objective and policy outcomes had not been accessed for this plan. |
| Inventory | 54 frozen anchored target pairs and 36 archived ordinary target pairs (two families × six targets × three seeds). No additional GPU training. |
| Anchor arm accounting | 54 × nine policy records = 486. The 216 original records sit within these 486. |
| Ordinary inputs | 36 × four policies = 144. Total table = 630 rows; decisions sharing inputs are paired, correlated observations. |
| Primary estimand | Seed-mean deployed MAE within domain, divided by seed-mean local MAE; equal mean across six domains. Report margin minus prefix under locked rule. Report absolute error, decision label counts, restricted acceptance times, useful/harmful selections and material-harm frequency. |
| Secondary estimand | Same quantity for the margin-trained weighted forecast minus each fixed comparator; original two-thirds binary and ordinary-prefix same-weight comparisons distinguish deployment from training. Report actual w and labels. |
| Statistical uncertainty | 2,000 paired circular row-block resamples, lengths 144 primary and 72/288 sensitivity. Seeds averaged first; aligned random block starts by row in NYC. These intervals condition on the observed domains. Report domain effects and full block counts. Hangzhou has two complete 144-row blocks. No pooled 630-trial significance test or post hoc selection of a winning policy. |
| Exposure | All verification weights and binary decisions saved and hashed before test scoring. Rolling inputs may use later observed histories; decision label counts do not measure sensor savings. |
| Resources | One longq CPU Slurm job, five CPUs, 12 GiB, three-hour request. Actual queue delay and runtime remain open. Existing NumPy project environment, no new weights. |
| Artifact trail | Original inventory and code hash, campaign revision, receipt, decision digest, metrics, block intervals, cohort coverage, manuscript draft and small aggregate bundle. Raw observations and per-node forecasts stay in their project stores. |

## What would support each claim

- **Training adds value.** Lead with the original margin-versus-prefix contrast under the same locked rule; report the suffix control and all label and harm tradeoffs. If the rule always returns local, the deployment effect is exactly zero. Unconditional Beijing improvements do not rescue this endpoint.
- **Partial correction adds value.** The amended experiment can support a bounded statement about these six datasets if it improves over strong local and same-budget binary controls, the effect is visible across targets, and harm and sensitivity analyses remain favorable. Show against blind averaging and the ordinary-training weighted controls even if they win. If the weighting rule explains the gain, attribute it to the deployment policy.
- **Magnitude matters.** For the manuscript decision, use two percentage points of local-normalized MAE as a practical benchmark, show domain-specific differences, 144-row descriptive intervals and sensitivity to block length. Examine whether dropping the shared NYC fold reverses the point estimate. A 2% improvement against a weak baseline, or a result driven by one city, is inadequate for a broad claim.
- **Comparison with Zhang et al.** Use the exact scope of the result: shared-protocol B03 or a named common test-row checkpoint comparison. A statement that we beat the original eleven-domain study would require aligned tasks, input access and those missing datasets. Under matching eligible rows, report the actual percent change and coverage for each domain, including losses.
- **If the result is mixed or negative.** Lead with the measured gap between useful transfer and verifiable transfer only if reachability, realized decisions and future utility consistently distinguish the mechanisms. If neither a transferable improvement nor a robust explanatory pattern appears, release the results and defer journal submission. A new objective or threshold learned from these six results requires a new time block or external dataset.

## Work schedule and paper deliverables

| Elapsed time | Output and dependency |
|---|---|
| 0–4 h | Freeze the question, exposure record, methods, review, code, labels and synthetic tests. Publish the working manuscript and this dated protocol. |
| 4–12 h | Submit the single CPU campaign where the archived artifacts reside. Check accounting, source hashes, seals, eligible domains and record counts. A queue delay shifts downstream work. |
| 12–24 h | Inspect aggregate output and label ledgers. Produce the six-domain main table and reference overlap table, report cost and harm, compare margins to ordinary controls, and examine NYC-fold sensitivity. |
| 24–36 h | Prepare method schematic, domain effect plot, verifier reachability/utility counts and reference coverage. Display seed variability and distinguish descriptive intervals from cross-city inference. Integrate held-out numbers and limitations into the full manuscript and abstract. |
| 36–48 h | Audit every number against the archived CSV, reconcile dataset counts, package source-data tables, and make the journal-readiness decision. Final author list, affiliations, declarations, funding, data rights and author approval are needed for actual submission. |

A larger backbone campaign is a separate decision. If the held-out objective beats ordinary controls, preregister an equal-parameter backbone ablation and a fresh untouched cohort before claiming that language pretraining or Qwen causes the effect. A variance-adaptive sequential comparator is also justified by the literature if the current boundary is frequently unreachable; it needs a valid bound for our predictable score ranges and fresh evaluation. Relaxing thresholds after reading this panel would answer a different question.

## Reproduction

From the HPC project directory use the installed, pinned original scripts/bootstrap_hpc.py with --revision set to the campaign commit and --sync-only. Then run scripts/submit_traffic_campaign.py from the installed campaign snapshot. Repeating the launcher reports the active job or completed aggregate and permits recovery following a recorded Slurm failure. The main result is results/traffic_publication/COMMIT/SUMMARY.md. The sibling COMMIT_report.tar.gz contains the report, manuscript draft, statistics and aggregate CSVs.
