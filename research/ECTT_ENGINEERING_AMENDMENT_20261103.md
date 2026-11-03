# Numerical implementation correction: spatial pooling

Date: 3 November 2026. Parent source: b97d18f0ec62dc24e9d83e1ba99da93d6d1b73b7.
Affected HPC attempt: 616045, identity
01294e6a3b6c6e24599e1ae17b582d13c9fcf11bd43b5525264098c76f38bf57.

## Observed failure and scope

Environment setup, pinned model acquisition/loading and reuse of CPU admissions
passed. Node-permutation admission failed before training with maximum output
difference 0.09887021780014038. Optimizer steps and benchmark evaluations were
zero; original stores/reports remain preserved for audit/resource accounting.

## Correction

The spatial context remains the node-mask-weighted mean of spatial embeddings,
computed separately for each forecast origin. Only its numerical accumulation
changes: multiply/sum/divide in FP64, then cast the context back to FP32 before
projection fusion and the existing BF16 Qwen conversion. FP32 summation can
vary with node order; its tiny differences may cross BF16 rounding boundaries.
Local synthetic probes confirm different BF16 tokens under the old reduction.
The repaired probes are identical under reversed or randomly permuted nodes.

This is an implementation precision correction, not a new learning mechanism
or change to objectives, episodes, data, splits, labels, seeds, LoRA settings,
model architecture or hypothesis thresholds. Numeric protocol configuration
files remain byte-identical. The actual computation changes, so its code hash
and GPU run identity change. Both admission tolerances remain atol=rtol=1e-3.
No old GPU checkpoint/result can clear the new pilot, and no favorable accuracy
outcome was consulted when choosing this correction.

## Evidence required

The regression checks BF16 input tokens at hidden width 1536 and 170/358 nodes,
uniform/nonuniform masks, a tiny 28-block BF16 Qwen, gradients and recovery.
A deliberately node-biased predictor must still fail the original gate.
These local checks use CPU Torch 2.0.0, not the HPC 2.4.1+cu118 runtime. They
do not prove the full-size CUDA error is resolved. The unchanged HPC gate,
six support-only updates, posttraining gate and strict reconstruction must pass
before engineering admission is counted complete. Per-case gate artifacts now
survive failure. No test accuracy or ECTT objective result is established.

## Prior admission reuse

CPU data/statistical stages never import or execute ectt_model or ectt_pilot.
Exclude those modules and workflow orchestration from their narrow compatibility
fingerprint; pin every remaining scientific/execution/data/registry/helper
input to the original admitted version and verify all completion artifacts.
Only the original d70d0aa... CPU stores may be reused through this rule. Original
stores are not rewritten, and old GPU results are never imported. Existing
project-local dependencies and Qwen cache are revalidated normally.
