# Traffic forecasting reproduction

Step 1 acquires and audits the LLM-UTP release on the HPC. It downloads both
pretraining and fine-tuning arrays for the eight publicly released domains,
all eleven released target checkpoints, utility source payloads, the Source
Data archive and pinned official GPT-2 base files. It does not run inference,
train a model, submit a Slurm job or claim successful paper reproduction.

The package contains code and source locks only. No datasets or weights need
to pass through the Mac or this repository. All downloads happen on the HPC.

## Run on this HPC

Heavy work is submitted through Slurm. See [HPC_PROFILE.md](HPC_PROFILE.md)
for the cluster profile and [AGENTS.md](AGENTS.md) for future project changes.
The stable root remains /home/tariq.qayyum/nature-llm/ so existing downloads
and reports are retained.

Download the latest standalone scripts/bootstrap_hpc.py from GitHub into
/home/tariq.qayyum/nature-llm/submit_hpc.py, then run:

```bash
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py --project /home/tariq.qayyum/nature-llm --submit --handoff
```

This performs a small code update and submits Batch through sbatch; it does
not stream the full acquisition in the terminal. The real job ID is printed.
Git is unnecessary. --handoff pauses only a positively identified unmanaged
project downloader to resume its partial files in Slurm. It never cancels a
Slurm job or signals an unrelated project. Repeat with --submit for recovery;
an already pending/running download job is not duplicated.

Batch requests longq, one task with four CPUs, 8 GiB RAM and zero GPUs. Its
23-hour time limit is capped by observed longq MaxTime when available. Four
HTTP workers acquire independently named assets and write separate logs.
The job then audits the arrays. srun launches exactly one pipeline process;
the job does not launch forty copies of the same download.

All environments, caches, temporary files and outputs remain under this
project. Batch loads anaconda3 and creates .venv locally; dependencies are
never installed into the shared Anaconda prefix. Existing work/audit_env is
preserved. Use a node allowed by the institution for external transfers;
compute-node HTTPS access has not yet been verified.

Each GitHub code archive is verified against the exact committed tree and
installed at work/code_versions/COMMIT/. A queued job pins that source path.
The original checkout stays intact. The reference source and every dataset
retain their pinned hashes; completed assets are checked and reused. Reserve
at least 8 GiB of project quota for this acquisition, with more for training.

Check the newest submitted job with:

```bash
bash /home/tariq.qayyum/nature-llm/Check
```

Append a job ID to inspect an older recorded project job. Check displays
queue/accounting reasons, exit codes, CPU/memory observations, asset worker
counts, failed worker logs, recent output and result locations. The Batch
logs are logs/nutp-dl-audit_JOB_ID.out/.err; per-asset logs have their own
JOB_ID_workers directory. Final reports are under
results/slurm_jobs/JOB_ID/report/ and the job summary is
results/slurm_jobs/JOB_ID/JOB_REPORT.md. No training experiments are started.

--sync-only installs code without a job. --audit-only submits a job to verify
existing assets without downloading dataset/model payloads. The explicit
--interactive option is reserved for small development/debug checks, not
long downloads or experiment grids.

## Data scope and verification

Acquisition job 605458 completed on the HPC with all 37 assets verified.
See [PROGRESS.md](PROGRESS.md) for the distinction between real execution,
local synthetic checks and remaining scientific uncertainties.

The next command, after refreshing submit_hpc.py from GitHub, is:

```bash
python3 /home/tariq.qayyum/nature-llm/submit_hpc.py --project /home/tariq.qayyum/nature-llm --stage pems08-audit --submit
```

This submits nutp-p08-preflight with one CPU, 4 GiB RAM and no GPU. It asks
for 30 minutes in shortq when the observed queue limit supports that request,
otherwise longq. The job reads existing arrays/checkpoint/source, writes exact
origin ledgers, and checks scaling and static checkpoint metadata. It does
not download data, deserialize tensors, run a model or train anything.
Check displays five stage-specific checks and the report remains under the
new job ID. Strict model compatibility and numerical inference remain later
milestones. The known NYC source duplication does not block this isolated
PEMS08 preflight.

The pinned upstream commit is
`50a4f5b0e76d09c952a01f07390bf57c77b22a26` in
[LLM-UTP/LLM-UTP](https://github.com/LLM-UTP/LLM-UTP).
The publication is [Zhang et al.](https://doi.org/10.1038/s41467-026-73610-2).
The archived release is [Zenodo 19503649](https://doi.org/10.5281/zenodo.19503649).
`config/sources.lock.json` records every original source blob identifier,
plus exact Git LFS hashes and sizes inspected in that commit. The reference
source archive is independently verified without executing Git. It is kept in
`work/reference/snapshots/COMMIT/`; payload files are hydrated in that snapshot. The source ZIP in Zenodo contains pointers; the script obtains
their actual objects and verifies SHA256 rather than treating the ZIP as all data.
It does not substitute standard PEMS mirrors for the authors' prepared arrays.

Public full-array targets are Beijing, Shanghai and Hangzhou Metro; PEMS03,
PEMS04 and PEMS08; NYC bike and NYC taxi. Both stage-one and stage-two files
are acquired with original names, including the author's PEMS04 spelling.

The full Nanning Metro, Beijing Bus and railway archives are restricted and
are not obtainable from this public release. Their public example arrays and
released target checkpoints are retained, but examples are never registered
as full training domains. Eleven-domain training is therefore not enabled.

The inspected release has identical stage-one bike/taxi object hashes.
The script reports that as a blocker for treating them as distinct source
domains. It does not replace one file with another dataset or silently change
the release. A corrected author release will require an explicit reviewed
source-lock update and a separate work directory.

The GPT-2 files come from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2) at
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
These are official base files, not proof of the original authors' exact
initialization. The original `model/GPT2` placeholder is preserved; base files
are stored separately under `work/models/gpt2/REVISION/`. LFS weight files
use SHA256; small Hugging Face files are checked using their pinned Git blob
SHA1. A measured SHA256 is also recorded for every downloaded file.

NumPy arrays are memory-mapped, loaded without pickle and scanned in chunks
for shape, dtype, zeros, negatives and nonfinite values. Their original order
is node by time by feature, matching the author loader. Timestamps are not
invented from row counts. No model pickle or upstream Python code is executed.
Licenses, full timestamp provenance, service gaps, raw/denoised channel meanings,
split origins and S1/S2/S3 checkpoint identity remain scientific audit tasks.

## Reports and exit codes

`work/LATEST_REPORT.txt` identifies the latest pipeline report directory.
`work/LAST_JOB_ID` identifies the latest submitted Slurm job. Each report includes:

- `SUMMARY.md`: readable findings and next-step blockers.
- `acquisition_audit.json`: arrays, duplicates, missing files and all verified hashes.
- `asset_progress.json`: incrementally saved download progress.
- `sources.lock.json`: the exact source inventory used for this run.
- `hpc_environment.json`: hostname, Python, filesystem space, quota and scheduler/GPU observations.
- `audit_environment.txt`: audit package versions, separate from training dependencies.
- `status.json`: timestamps and final process status.

Exit 0 means the acquisition/array checks finished without detected duplicate
domains. It never means training reproduction passed. Exit 2 means missing,
invalid or failed assets; rerun to retry. Exit 3 means downloaded arrays passed
structural checks but identical cross-domain payloads require scientific review.
For this pinned release, the bike/taxi duplication makes pipeline exit 3
expected even after every download succeeds. Batch maps that specific review
state to Slurm exit 0 and records COMPLETED_WITH_REVIEW; scientific progress
remains blocked. Required-file failures remain nonzero Slurm exits. Setup failures exit 1. No failure is silently
reported as a successful reproduction.

`bash run.sh --audit-only` verifies existing files and writes a new report
without downloading assets. The first numerical milestone after this audit
is a strictly identified PEMS08 checkpoint, matched splits/scaling/masks and
independently recomputed metrics. That will be added as the next code change.

## Local verification of this code

`python3 -m unittest discover -s tests -v` checks downloader resumption,
ignored ranges, corrupt payload rejection, array auditing and the scientific
duplicate-domain gate using tiny synthetic fixtures. Checks also cover source archive integrity, path traversal, symlinks, source
modification, update locking and an update with Git absent from PATH. These
are code checks, not forecasting experiments or a claim of HPC connectivity.
