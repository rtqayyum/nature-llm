# Frozen May NYC forecast comparison

**25 November 2026, before May 2026 demand access.** May is a chronological development validation. The 7B text audit used 57 May notices and informed the choice of a prompt. April demand informed the numerical recipes and practical comparators. This record fixes the remaining comparison before collecting May travel outcomes. Later confirmation requires separate periods and corridors with a new registration.

### Forecast origins, population and joins

Use the 1,777 original NYC subway notices from May 1 through May 31 local time in the released ledger. Predict the first complete local hour after issue (computed in UTC, then converted to New York time). Keep target hours ending by June 1 00:00. Retain original event and alert IDs. Neither revisions nor end times enter the features. Use the January–April pipeline at commit 660fb2e, extending its month boundaries. Use the final seven April days to assign each dock to the nearest April 29 complex coordinate within 500 m, with the lowest numeric complex ID breaking an exact distance tie. Apply the May map to target and lag. Leave absent rail aggregates missing and count zero recorded bike starts only for a mapped catchment in an observed citywide hour.

Use the dated base GTFS, including exceptions and preceding-day service. The spring version ends May 16. The succeeding feed has version date May 13, a May 17 service start, and its earliest verified capture in this audit was May 18 at 00:26:20 UTC. For the frozen primary comparison exclude any origin whose target service date requires the new feed before that capture. Preserve those 34 notices in an identified sensitivity group. Do not infer publication from the version-date field. All 1,777 original notices have in-May targets; 1,743 pass the conservative time-of-availability condition before route and outcome checks. Never use the retrospective effective-service archive as an input. [Source checksums and captures](sources.json)

Primary validation covers all eligible alert-exposed station-hours, matching the April pilot. Classify maintenance/project, marine bridge, fault/response wording and unclassified notices using the frozen cue function. Report the fault/response subgroup separately. It is a proxy for unscheduled incidents. Planned-service specificity and a genuinely unscheduled-only primary population remain requirements for a confirmatory claim. Define each case using the native affected routes, independent of either extractor. Assign each complex-hour to the earliest eligible original, tie-breaking by numeric event ID. Retain rows with both observed targets and both same-local-hour weekly lags for all primary arms. Report the entire pre-mask denominator.

### Train, extract and score

Fit on completed January–April targets ending by May 1 00:00. Purge training origins sharing a frozen 24-hour recurrence group with any May notice, including May notices that lose all target rows. A group links consecutive identical normalized initial headers and route sets within 24 hours. Fit TF-IDF, the 16-component SVD and category vocabulary inside training only. Use the released April seasonal-residual histogram boosting recipe per mode, with its absolute-error loss, fixed 100 iterations, leaf size and other settings unchanged, and the same nonnegative clipping. Use the weekly-count comparator directly. All models in this primary analysis assume timely operator access to completed counts.

1. Compare weekly seasonal, boosted native/cue, boosted full text, and boosted full text plus rule names and roles.
2. Add Qwen named-station indicators and then Qwen role indicators to the full-text-plus-rules head. Retain the same native data, text, rule features, local station coordinates, splits, hyperparameters and scoring keys in both Qwen arms. The reference geometry supplies station identity and coordinates in every arm. The added mention indicator identifies a locally named station. This comparison measures the contribution of the extra Qwen role fields conditional on the shared rule features.
3. Freeze the example prompt from the 20-notice follow-up and the quantized runtime from the audit. Run on every distinct (routes, original header, original description) in January–May. Cache identical model inputs once across notices. Permit only the nine defined labels and source-grounded names resolving through the same frozen station dictionary used in the audit. Drop invalid fields, retain all forecast rows and count unresolved, invalid and unknown cases. Keep rule and Qwen feature columns distinct. No gold labels are features.
4. For the role-exchange sensitivity, use the trained full Qwen head. Select the first two retained Qwen-named complexes by ascending numeric-ID pair whose nonempty role sets are disjoint within the same notice; exchange those two role vectors only, retaining all station names and rule features. Score on the same two stations under both assignments. Report the number of distinct contributing incidents and dates. Report named-station subsets separately from the whole network.
5. Compute per-mode MAE in natural units and equal-mode percentage reduction in MAE on identical rows. For a fixed May comparison use the best network non-Qwen arm by mode from the April study: weekly seasonal for subway, boosted full text for bikes (the name/role variants were tied). Report the Qwen-full versus Qwen-names contrast separately. Use the strongest *May*-validated eligible non-Qwen arm by mode for later confirmation, with ties resolved in favor of fewer inputs. Keep this selection out of any claim of untouched May performance.
6. Use all simultaneous stations and both modes together in 5,000 circular seven-day block resamples, seed 20261125, with three-day blocks as sensitivity. Report the 2.5th and 97.5th percentiles as descriptive within-month uncertainty. The 57 parser-audited IDs, identical model inputs and linked recurrence groups are marked in the ledger. Repeat the headline aggregate after removing their assigned rows. Report counts and dates. This exclusion cannot remove all shared wording across notices.

### Limits and decision

The May primary is the one-hour operator-lag, alert-exposed comparison. Ordinary-hour risk, three- and six-hour horizons and a release-time-qualified public-input model need separate evidence. As an explicitly lag-free sensitivity, fit using train labels completed through March 31, predict raw counts by the same fixed boosting recipe after removing weekly lag from every feature vector and objective, and compare against a train-only median for each station/hour of week (fall back to station median, then mode median). Use the primary mask for paired reporting. Label the one-month embargo as an information restriction until true public release dates are verified.

Freeze both Qwen forecasts and non-Qwen comparisons before interpreting validation results. Advance a superiority claim only with a practically relevant effect that survives the role-blind comparison and stronger baselines, then estimate precision requirements for unseen data. A null or harmful May result remains reportable. No 72B grid is warranted by the extraction audit alone. The publication decision also depends on independent periods, city replication and ordinary-hour performance as set out in the [study design](../../MULTIMODAL_FORECASTING_DIRECTIONS_20261123.md).

### Extraction batches

The [origin ledger](origins.csv) contains issue and target times, recurrence IDs, exposure flags and model-input hashes. The 9,050 Jan–May originals reduce to 8,310 exact model inputs. The collector partitions them into 96 deterministic shards using the prompt, model identity and exact raw inputs. One representative event is stored per identical input; join back by its key. Each completed shard is written atomically. With the verified [model and local server](../may_parser/README.md), run the following from the repository root; change the shard index for successive batches.

```bash
python research/alert_study/may_validation/collect_text.py /path/to/jan-apr-notices /path/to/may_notices.json --shard 0 --server http://127.0.0.1:18443
```

The source directory holds 2026-01.json through 2026-04.json matching the [published hashes](../role_screen/sources.json). The May source is pinned separately. A receipt for a parser shard measures extraction coverage; forecast accuracy requires the later paired demand evaluation.


The first completed batch contains 90 verified model inputs linked to 94 first notices. All 90 responses had a valid array shape and 13 were nonempty. This receipt covers shard 00. Valid array shape alone says nothing about correctness. [Raw shard](parser_shards/00.json.gz) · [Extraction receipt](parser_shards/receipt.json)

The [May panel and frozen numerical controls](panel/README.md) record the outcome join, timetable sensitivity, assigned station-hours, independently verified May 27 trace and paired forecast keys.

### Resume extraction and freeze the language-model forecasts

The [verified extraction inventory](parser_shards/progress.json) gives the completed shard numbers, hashes, linked notices and coverage of the 8,310 distinct inputs at the latest checkpoint. Its counts describe raw extraction and deterministic station resolution. The checker matches every stored output to its exact prompt/input key and the corresponding representative notice. It retains a station–role pair when the role is one of the nine defined labels and the normalized station name occurs at a word boundary in the original text and resolves in the frozen directory. An unresolved or invalid pair contributes no role features. The forecast population is fixed independently of this check.

To resume, collect the missing shard numbers using the command above. Existing complete shards are checked on restart. After all 96 have been recovered, the following command validates the entire inventory and the non-Qwen prediction hashes before fitting the two Qwen arms and the within-notice role exchanges. It writes forecasts and their checksums before error analysis.

```bash
root=research/alert_study/may_validation
python "$root/freeze_qwen.py" "$CACHE" "$NOTICE_CACHE" "$root/qwen_forecasts"
```

### Run the extractor on a persistent CPU allocation

The [batch runner](run_extraction_job.py) uses the fixed CPU model, binary, prompt and serving settings. It verifies model and source hashes, imports published shards and resumes from the first missing shard. After each batch it writes a complete, independently verifiable checkpoint to `may_extractions.tar.gz` in the work directory. The bundle holds raw outputs, their hashes, the number of exact inputs covered, any Slurm job ID supplied by the scheduler, and the shards produced in this invocation. Generation receipts retain the per-shard hashes and job identifiers across restarts. A returned Slurm ID can be reconciled with scheduler accounting. The weights and travel outcomes are outside the bundle. A partial bundle can be shared and checked before all batches finish.

From any directory on the UAEU login node, this block downloads and verifies the bootstrap for the pinned extraction run. It installs the snapshot under the alert-study work directory, checks for an active job and requests one CPU job. The compute node needs Python 3.9+, curl, tar, outbound access for uncached files, and Linux x86-64 compatibility with the pinned llama.cpp release. The requested time and memory are allocation settings.

```bash
(
  set -e
  module load slurm python/3.10.20
  work="$HOME/nature-llm/work/alert_study/may_extraction"
  rev=57a979ff5f0a91f899403994607c7a4533a1b39b
  active=$(squeue -h -u "$USER" -n may-alert-7b -o '%i %T')
  if [ -n "$active" ]; then
    printf 'Existing job: %s\n' "$active"
    exit 0
  fi
  mkdir -p "$work"
  setup="$work/bootstrap-$rev.py"
  curl -fsSL --retry 3 -o "$setup" \
    "https://raw.githubusercontent.com/rtqayyum/nature-llm/$rev/scripts/bootstrap_hpc.py"
  printf '%s  %s\n' f19d2e2a6fd438162e1bba587557cc43215e682683aa1f7dc4d1d9b1b7d3b443 "$setup" | sha256sum -c -
  TRAFFIC_WORK_ROOT="$work/bootstrap/work" python3 "$setup" \
    --project "$work/bootstrap" --revision "$rev" --sync-only
  runner="$work/bootstrap/work/code_versions/$rev/research/alert_study/may_validation/run_extraction_job.py"
  test -s "$runner"
  sbatch --parsable -p longq --cpus-per-task=6 --mem=16G --time=16:00:00 \
    --job-name=may-alert-7b --output="$work/slurm-%j.log" \
    --wrap="exec python3 -u '$runner' '$work'"
)
```

The same work directory can be reused after an interruption. To verify existing files and package them without starting inference, run `python3 "$runner" "$work" --limit 0`. For a returned bundle, compare any duplicate shards to the published hashes before combining them. Preserve any discrepancy for review. Use [read_extractions.py](read_extractions.py) to validate all exact keys against the five original-notice archives and the frozen dictionary.

Pass `--shards "$work/parser_shards"` to both forecast freezing and scoring to use a returned checkpoint in place.

After running [freeze_qwen.py](freeze_qwen.py), use `python "$root/score_frozen.py" "$root/qwen_forecasts" "$root/scores"`. The [scorer](score_frozen.py) verifies the raw shard hashes and both sets of frozen prediction files before joining outcomes. It reports the primary and lag-free errors on identical rows, the roles-versus-names and role-exchange contrasts, audit exclusions, fault/response subsets, and paired three- and seven-day resampling. The separate [revision timing audit](revision_timing/README.md) describes later information and leaves the primary features and population intact.
