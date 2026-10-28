# Traffic forecasting reproduction

Step 1 acquires and audits the LLM-UTP release on the HPC. It downloads both
pretraining and fine-tuning arrays for the eight publicly released domains,
all eleven released target checkpoints, utility source payloads, the Source
Data archive and pinned official GPT-2 base files. It does not run inference,
train a model, submit a Slurm job or claim successful paper reproduction.

The package contains code and source locks only. No datasets or weights need
to pass through the Mac or this repository. All downloads happen on the HPC.

## Run

Use Python 3.8–3.12 and outbound HTTPS. No Git executable, Git module, sudo
or administrator installation is required. The entry point is
`scripts/bootstrap_hpc.py`. A standalone copy can be downloaded from this
repository with Python and run as:

```bash
python3 "$HOME/nature-llm-bootstrap.py" --project "$HOME/nature-llm"
```

The bootstrap resolves this repository's latest committed main revision,
downloads that exact source archive and verifies every file against GitHub's
commit/tree inventory. Each source version is installed under
`work/code_versions/COMMIT/`. It runs from that verified version and preserves
the original checkout. `work/CURRENT_CODE.json` records the version in use.

Data, models, logs, reports and the audit environment remain under the stable
`work/` directory in the project folder. Repeating the bootstrap command updates
code while retaining these assets. If you use a different allocated filesystem,
export `TRAFFIC_WORK_ROOT` before the bootstrap command. Standard Python
venv/pip support is needed for the isolated NumPy audit environment. If your
cluster needs a Python module, the optional `config/hpc.local.sh` in the stable
project folder is sourced by the launcher after source installation.

Allow at least 8 GiB of personal quota for this initial step and extra space
for later experiments. The pinned large assets total about 5.5 GiB. A filesystem
free-space check cannot determine personal HPC quota; the environment report
includes `quota -s` output when available. Use the institution-permitted node
for outbound transfers. No GPU is requested for this step.

The worker detaches from the SSH terminal and continues if it disconnects.
The launcher displays its log while connected. Each run writes a separate
dated report and preserves previous reports. It prints its PID, log path,
report path and final status. No GPU allocation is requested for this step.

## Update from GitHub

Repeat the standalone Python bootstrap command to fetch the latest verified
code version and run this step. `--sync-only` installs code without starting
acquisition. `--audit-only` writes an audit of existing assets without
downloading dataset or model payloads; the bootstrap may still download new code.
Inside a new version, `bash update_and_run.sh` calls the same bootstrap.
Older checkouts that still require Git should use the standalone Python entry.

Code updates stop while acquisition holds its work-directory lock. Existing
code versions are checked for modifications, and changed source is preserved
rather than silently overwritten. Dataset payloads use separate SHA256 checks;
verified files are reused and partial downloads resume when byte ranges are
supported. A server ignoring ranges triggers a restart of that object.

The same GitHub repository remains authoritative for code, configurations and
tests. Raw datasets, checkpoints, environments, logs and local reports stay on
the HPC. The HTTP bootstrap currently supports this public repository; no
GitHub credentials are needed or stored. It uses GitHub's public API for the
latest code inventory and reports service/network failures explicitly.

## Data scope and verification

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

`work/LATEST_REPORT.txt` identifies the latest run directory. Each report includes:

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
For this pinned release, the bike/taxi duplication makes exit 3 expected even
after every download succeeds. Setup failures exit 1. No failure is silently
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
