# Development comparison of source training losses

This amendment is exploratory and follows the exposed transfer results.
Jobs 676089 and 676092 confirmed poor conditional-on-panel baseline accuracy
and zero output gradients below the source loss floor in selected-checkpoint
source validation replays. Their results motivate this experiment and do not
establish that changing the loss will improve forecasting.

The matrix contains 24 freshly initialized source fits. It crosses the two
families B03/B04, held-out development domains PEMS08/Beijing, seeds 11/22/33,
and the following two loss conditions.

| Condition | Source suffix training prediction | Source validation and final physical forecast |
|---|---|---|
| clipped | max(0, m + (M-m) z) | max(0, m + (M-m) z) |
| unclipped | m + (M-m) z | max(0, m + (M-m) z) |

Both conditions minimize the same normalized absolute error expression.
All other code paths reuse the archived episode generator, compact/ridge
support scale, temporary support head optimizer, first-order gradient
semantics, source optimizer, precision, batch schedule and target residual
adaptation. Source stopping remains at most 100 epochs with patience ten.
Both conditions select by the clipped physical source validation MAE.
Each selected source is adapted to its excluded development domain at the
prespecified 5% budget. The target fit has 200 support-only updates. The
saved local baseline and exact test truths come from the original grid.

The original matrix, models and scientific contract remain frozen. This is
a versioned development experiment, with the original failed outcomes
retained. The six confirmatory target test panels are excluded from this
new evaluation. Verification labels and sequential deployment decisions
are outside this experiment. S01 independent review remains pending.

The launcher queues four recovery probes, then the 24 fits, then a CPU
collector. Each fit task includes source training and target adaptation,
with durable source and target checkpoints. Up to four one-GPU/five-CPU
tasks run concurrently. Probe success gates fitting. The collector follows
termination and records unsuccessful or missing tasks. Compatible completed
source fits and optimizer checkpoints are reused on an explicit retry.
