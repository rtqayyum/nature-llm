# Persistence anchor development amendment

This exploratory amendment follows the user-reported completed collection
1116448. Removing the source training floor improved seven of twelve paired
development runs, but all twelve unclipped runs remained worse than local.
Better source-validation scores occurred in ten pairs; four of those had worse
target test MAE. These exposed results motivate a new test, not a confirmatory
claim or proof of a particular failure mechanism.

Test whether an explicit persistence reference stabilizes the transferred
forecast. Use the frozen B03 and B04 encoders, PEMS08/Beijing development folds,
three seeds, original source exclusions, primary 5% target budget, and archived
episode, optimizer, support and validation schedules. Keep source suffix
training loss unclipped and validation/inference physical clipping unchanged.

Both fresh conditions zero-initialize the numerical output head and append the
last observed normalized value to the encoder features seen by the temporary
and target residual heads. The direct_zero condition predicts the numerical
head alone. The persistence_residual condition adds that last observed value
to the identical head output. No additional learnable parameter is introduced
by the skip connection. Neither uses the selected local model as its anchor;
persistence and the full selected local model are reported separately.

Thus the 24 new fits isolate the persistence skip connection conditional on
common zero-head initialization and residual features. The twelve archived
unclipped results are reference scores, not a fresh matched control for this
architecture. A favorable result would still require later untouched-domain
evaluation and controls. Do not select seeds or hyperparameters on these tests.

Four one-epoch probes test initial forecast identity, trained-model invariance
and exact source model/Adam/RNG/cursor recovery before fitting. Each fit trains
a fresh source model, then a fresh 200-update target support residual head.
Up to four one-GPU/five-CPU tasks run concurrently. Save normalized source-only
and adapted forecasts, the fixed scalers, physical predictions, common truths,
local and persistence scores, checkpoints and runtime. Collect all failures.
Report source updates separately from the temporary 200-update support fits.

A parallel CPU stage exports the derivation, code hashes and reviewer checklist
and reruns internal synthetic checks. It does not constitute an independent
mathematical review, score traffic verification labels, or authorize ECTT runs.
Independent S01 sign-off remains pending. The final CPU collector verifies this
packet and all 24 cases. All predecessor stores and scientific contracts remain
unchanged.

Reporting correction: predecessor CSV source_validation_MAE is dimensionless
local-error-scaled source suffix MAE. Its source_clipped_fraction describes
source-only predictions on the held-out target test origins, not source
validation suffixes. New reports label these quantities explicitly.
