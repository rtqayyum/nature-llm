# Index-causal reference baseline, version 1

This is a causal adaptation of the released LLM-UTP architecture, not the
proposed new method and not an exact reproduction of published S3. The
corrected released-behavior runs remain separate. The following choices are
fixed before observing accuracy from the causal runs.

## Data access and cohorts

- Domains: PEMS08 and Beijing Metro. Seeds: 11, 22, 33. No joint training.
- Use only released channel zero. Verify pinned stage-one and stage-two
  payload hashes and require their raw channel-zero arrays to be identical.
  Never substitute another input because its test score is preferable.
- Use the released pretrain route's scored target rows and tail dropping.
  Training, validation and test target sets must be chronological/disjoint.
- PEMS08 scored targets: train [12,14284), validation [14284,16060), test
  [16071,17847). Beijing: train [12,1004), validation [1008,1504), test
  [1512,1800). Runtime derives/verifies the ledgers from pinned source.
- Input contains exactly the preceding 12 observations; predict one next
  observation. PEMS08's native interval is five minutes, Beijing's fifteen.
- Evaluation is rolling one-step forecasting: earlier observed validation
  or test values may enter later histories after they have become available.
  No model/optimizer/scaler fitting occurs on validation or test labels.
- Retain released row order. Do not invent timestamps, fill gaps, interpolate
  or silently interpret packed service rows as an uninterrupted calendar.
  Absolute dates and service-gap semantics remain unverified.

## Training-only scaling and causal trend

Fit a single global node min/max only on the union of scored training input
and target rows. For the locked cohorts that support is [0,14284) for PEMS08
and [0,1004) for Beijing. Validation/test rows never determine the scaler.

Replace the supplied second channel with prefix means reset separately
inside each history window. For observed history x[0:12], the auxiliary
value at position j is mean(x[0:j+1]). No older rows or unobserved future
rows are used, and no smoothing parameter is tuned. This provides a
deterministic trend feature without increasing the raw context beyond 12.

Normalize both historical channels and targets with the frozen training
min/max. Do not clip normalized held-out values to [0,1]. Reconstruct flow
with z*(maximum-minimum)+minimum, then apply nonnegative physical clipping.
MAE/RMSE score every cell. Preserve the documented released WMAPE mask
(numerator truth>0, denominator sum of truth) and verify it independently.

## Explicit architecture adaptation

Keep the author encoder, GPT-2 initialization, retained/active blocks,
parameter names and constructor freeze policy. Change only the two fusion
softmax calls in TrendEnhancementModule.forward from implicit dimension
to explicit dim=-1, over features within each forecast window/node. This
removes the known softmax operation across the forecast-window batch.

An AST edit must find exactly the two expected calls. The generated forward
source/hash is saved. The installed module must preserve state_dict keys
and parameter requires_grad flags. Original pinned author source is never
edited, and the faithful inference path remains separate. This adaptation
does not establish the paper's intended parameter-training policy.

## Mandatory causal checks

Before training, changing all raw rows at/after the first validation target
must leave both the training scaler and that target's historical features
unchanged. The raw channel, context limits and disjoint target sets are
recorded in data_contract.json.

Before training and after loading the best validation checkpoint, evaluate
the first training minibatch with dropout disabled. Require:

1. Replacing all companion windows by zeros or ones leaves the first
   prediction unchanged within atol=1e-6, rtol=1e-5.
2. Evaluating that first origin alone agrees with its batched prediction.
3. Permuting the batch only permutes predictions.
4. Predictions are finite and parameter versions do not change during the
   controlled forwards. A first CUDA GRU-layout warmup precedes version checks.

No accuracy metric is used to select a patch or adjust these tolerances.
Failure stops the task with nonzero exit status. These are finite functional
probes backed by source inspection, not a proof for every input/device or
verification of the unavailable calendar metadata.

## Training, selection and evaluation

Start from the pinned official GPT-2 base and fresh seeded author layers.
Never load a released target checkpoint or one of the earlier trained runs.
Use FP32, disabled TF32, Adam 1e-4 and MSE, inner batch 16/outer batch 1.
Shuffle whole inner batches with the existing documented epoch-specific
seeds; do not shuffle observations within an input/target window.

Maximum 1000 epochs, patience 100. Select the lowest validation MSE starting
at epoch two, matching the locked reference-training engineering convention.
Test only after training terminates and the selected-model causal checks
pass. Do not compare against a paper row to select weights or a protocol.

Save model/Adam/CPU+CUDA RNG states, cursor and losses every epoch and
periodically after five minutes at 50-batch boundaries. Retain two recent
states and the referenced best weights. Compatible checkpoints resume;
completed results require valid result/model/forecast hashes and causal status.

Store all new results under outputs/causal_reference/ and new Slurm report
directories. Hash the method-independent data contract so all seeds and
future methods can verify identical available observations, scalers, target
rows and metric conventions. Report all seeds and sample standard deviations.

The implementation remains under the ordered-row assumption. A successful
causal run is not a timestamp-verified full-paper reproduction. Any later
claim about the proposed method requires the same data contract and broader
eligible-domain evaluation; these two domains alone do not establish that claim.
