# Research execution progress

## Acquisition completed

The researcher supplied completed HPC logs for Slurm job 605458, named
nutp-dl-audit, on cn119. Accounting reports COMPLETED, exit 0:0, elapsed
52 seconds, four allocated CPUs and zero GPUs. The report records all
37 requested assets verified, 5.494 GiB, zero failures, and all sixteen
public stage-array entries structurally audited. This is real HPC execution
reported by the researcher, not a synthetic test or a direct remote inspection
by the coding assistant. Local code tests remain separately identified.

The acquisition result is COMPLETED_WITH_REVIEW. Stage-one NYC bike and taxi
have identical hashes. Full Public8 joint training remains blocked pending
that release issue, and the full eleven-domain original retrain additionally
needs the three restricted archives. These limitations do not prevent a
separate PEMS08 checkpoint/split investigation.

Verified PEMS08 arrays are reported as 170 nodes by 17856 rows by two channels
for both stages. Original timestamps, channel provenance, checkpoint S1/S2/S3
identity and exact original GPT-2 initialization remain unresolved. No traffic
forecast, retraining run or replicated MAE/RMSE/WMAPE result exists yet.

## Next job

The pems08-audit stage submits nutp-p08-preflight, one CPU, 4 GiB RAM and
zero GPUs. It requests 30 minutes in shortq if that queue's observed MaxTime
supports the request, otherwise longq. No dataset/model downloads occur.

It verifies the selected existing hashes and pinned source, compares both
PEMS08 arrays, records the original split ranges and inner-batch tail loss,
counts train/validation target overlap, checks scaling/metric conventions,
and disassembles checkpoint metadata without unpickling tensors. It writes
six hashed origin CSVs and one debug-window NPZ. This job does not perform
strict model loading or prediction and cannot certify the training strategy.

The completed preflight report will determine the next checkpoint-inference
configuration. Faithful released-code behavior and a later clean protocol
must remain separate.
