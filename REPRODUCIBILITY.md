# Reproducibility before method development

Current scope: finish a defensible assessment of the reference results before
designing or evaluating the proposed method. The causal adaptation is on hold.
No new-method architecture, backbone download or training is part of this step.

## Primary evidence

- Zhang et al., Nature Communications 17, 7466 (2026),
  [paper](https://doi.org/10.1038/s41467-026-73610-2),
  [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13408357/):
  implementation, training strategy, Table 1, Methods and data availability.
- [Released author source](https://github.com/LLM-UTP/LLM-UTP/tree/50a4f5b0e76d09c952a01f07390bf57c77b22a26),
  pinned by config/sources.lock.json. Do not silently change the author revision.
- Actual HPC reports from the explicit IDs in config/reproduction_review.json.
  Numerical results, contracts, weights, forecasts and integrity ledgers remain
  on the HPC. Local synthetic tests validate software behavior only.

## What must be reconciled

| Question | Paper/release evidence | Consequence |
|---|---|---|
| S1 | Paper specifies eleven-domain joint training, then per-domain fine-tuning and testing | Eight target-checkpoint evaluations cannot establish the joint training history; three full restricted datasets are missing |
| S2 | Joint-stage model is tested without domain fine-tuning | Need an identified joint-stage checkpoint and training contract; do not infer identity from filenames |
| S3 | Independent single-dataset training and testing | Existing six fresh fits are independent stage1/pretrain experiments; the S3-specific array, split and trainability contract remains unresolved |
| Freeze policy | Paper describes all LLM parameters open during joint training and attention frozen during fine-tuning; released constructor sets only retained-block MLP parameters trainable | Preserve source behavior in faithful runs; do not assume a constructor policy certifies a particular paper strategy |
| Active architecture | Constructor retains six GPT-2 blocks, released forward executes two; paper reports two layers | Retained parameter counts and actually executed/gradient-receiving parameters differ; document both |
| Data and scale | Loader normalizes the whole supplied array across both channels; auxiliary channel differs between stages | Preserve released behavior for reproduction; a training-only scaler or reconstructed trend is a separate adaptation |
| Inverse correction | Our original independent-training evaluator used the target registry's max-only rule on a stage-one array with a negative minimum | Corrected saved forecasts use the recorded affine min/max inverse; historical outputs remain preserved. This repair is not a new method |
| Forecast batch | Two implicit softmax calls act across companion windows under the pinned runtime | Preserve behavior for release reproduction, report dependence; a feature-axis change is a separate architecture adaptation |
| Time coordinates | Public arrays do not provide verified timestamps or service-gap origins | Report row cohorts; do not invent dates, remove gaps or silently resample |

Code evidence locations: main_run.py (joint ConcatDataset, Adam, learning rate,
early stopping), fine-tuning.py (target route and loaded pretrained checkpoint),
model/LLM_UTP.py (fusion softmax, retained blocks, executed blocks, freeze policy),
data/unified_dataset.py and data/dataset_registry.py (split and normalization),
prediction.py (inverse and evaluation). The review hashes these exact files.

## Current execution: reproduction-review

One small Slurm CPU job reads historical evidence; it does not rerun models.
Its fixed plan includes checkpoint job 607564, diagnostic job 607596,
training array 609006 and correction job 610878. It never picks a different
run because its test score is closer to a paper value.

The report includes:

1. Eight checkpoint-versus-S1 comparisons with each metric difference and
   two-decimal agreement, without assigning checkpoint training strategy.
2. Six corrected independent-training results and mean/sample standard
   deviation by domain, with diagnostic S3 anchors clearly separated.
3. Validation of completed project registrations, result/contract links,
   selected-weight and forecast hashes, source revisions and metric agreement.
4. All protocol discrepancies, restricted-data limitations and the availability
   of earlier source-archive, causality and smoke reports.
5. A hash ledger of every read artifact. Any required evidence failure exits
   nonzero and leaves a failure report plus partial ledger.

This is evidence consolidation, not an independent attestation of historical
Slurm accounting or a fresh numerical rescore. Earlier validation assertions
remain prior job evidence. Missing optional reports are not treated as passed
scientific checks. Completed review is not full paper reproduction.

## Questions that could justify more reproduction experiments

- Which exact data hashes, origin ledgers, checkpoint hashes and S1/S2/S3
  labels generated Table 1, particularly PEMS04, Shanghai and Beijing?
- What were the S3 input stage, auxiliary-channel construction, split route,
  GPT-2 initialization, trainability settings, seeds and stopping history?
- Was the stage-one bike/taxi duplicate intentional, or is a corrected payload
  available with a documented hash and preprocessing history?
- Can the three restricted full datasets be obtained through the stated access
  process, with their timestamps, preprocessing and use conditions?
- Which source settings produced the supplied second channel without accessing
  future observations, and which batching convention was used for evaluation?

No message is sent to authors by this project stage. Author clarification or a
new official artifact must be versioned before changing the reproduction
contract. Do not search offsets, layer choices or scaler variants until a
paper row happens to match. The fixed diagnostics already performed remain
diagnostics, not a replacement selected reference baseline.

## Transition to our solution

The reproducibility phase ends with an explicit evidence matrix: demonstrated,
discrepant, unavailable, or unresolved. Do not label unavailable experiments
successful or keep launching blind repeats. If exact reproduction remains
blocked, state the limitation and freeze the supported reference scope.

Method development then needs a separate research contract: a falsifiable
hypothesis, a mechanism beyond preprocessing repairs, a primary-literature
novelty check, matched data and evaluation conditions, validation-only design
choices, at least three seeds, component ablations, uncertainty and measured
runtime/memory cost. Novelty and superiority are hypotheses until supported.
The prepared causal adaptation may later serve as an experimental control;
it is not itself the proposed contribution or a published-S3 reproduction.
