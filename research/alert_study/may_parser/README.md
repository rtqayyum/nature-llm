# Can Qwen recover the stations and instructions in a first notice?

**May 2026 NYC text-only development audit.** The frozen rules recovered 30 of 42 reference station–role mentions in a 28-notice sample. Qwen2.5-7B-Instruct recovered one with the initial prompt. On 20 different notices, adding worked examples increased Qwen's recovery from 4/23 to 18/23 and produced 23 extra assignments. The errors often involved stations that the notice really named: the model supplied an operational role the wording did not support. This audit evaluates extraction; forecasting against May travel counts is a separate experiment.

## Fixed samples and reference

We queried the MTA service-alert archive for all 1,777 NYC subway version-zero notices issued in May local time. No demand records entered the sampling, prompt, annotation or scoring scripts. The primary sample divided the month into four issue-date blocks (1–7, 8–14, 15–21, 22–31) and counted the presence of five groups of wording cues. Within each block we ranked event IDs by a fixed SHA-256 and took two cases with zero groups, two with one and three with two or more. We report unweighted audit results; this design deliberately oversamples instruction-like wording. Nineteen of the 28 notices had reference roles. The 42 assignments included 31 express boundaries, seven local boundaries, two reroute boundaries, one short turn and one boarding instruction. Evidence for other roles requires different notices. [Fixed primary sample, prompt and labels](https://github.com/rtqayyum/nature-llm/commit/32437608999b4b01db6b98a093bfe9f7249eedc3)

The supplemental challenge used the same initial prompt and nine distinct additional notices selected by lexical triggers for transfer, substitute service, boarding, turnbacks and suspension. Selection exhausted the eligible substitute-service stratum after one draw. The sample contained 38 reference assignments. A third sample used 20 additional IDs, four from each of five mutually exclusive wording categories. This follow-up tested one response to the first audit's omissions. The two prompts were compared on identical notices. The new prompt explicitly describes train movements as instructions and gives three invented examples. All 57 notice IDs are distinct. Some wordings and affected stations recur. [Challenge set and labels](https://github.com/rtqayyum/nature-llm/commit/e07dd364b60b82f02512099a57782453473a18d5) · [Follow-up prompt, sample and labels](https://github.com/rtqayyum/nature-llm/commit/18a908d8328111d41f8dc611873f28aedabec101)

The reference is one internal AI-assisted reading using the notice and station directory. The reviewer recorded the named station, current operational role and complex ID before viewing output for that sample. There was no independent second annotation. We excluded cause-only locations, ordinary direction names, unnamed bypassed stops, separate planned-work reminders and restored normal service. A service boundary and a short turn can both apply to one station. In follow-up event 251620 the shortened “14 St” identifies a short turn in the text, but the notice and L-line description do not establish a unique complex. We included this label for the mention task and masked that mention–role pair in the complex task.

## Exact station mentions and roles

An assignment is one distinct pair within a notice. Matching ignores case and repeated whitespace, counts both endpoint stations, and keeps every observed notice in the denominator. A duplicated prediction counts once. Precision is the share of emitted assignments supported by the reference. Recall is the share of reference assignments recovered. An invalid response supplies zero assignments and fails exact-notice agreement. The follow-up initial-prompt run had one shape error; the other runs produced valid arrays. One surrounding JSON code fence is accepted.

| Fixed sample | Extractor | Correct | Extra | Missed | Precision | Recall | Exact notices |
|---|---|---:|---:|---:|---:|---:|---:|
| Primary, 28 notices | Rules | 30 | 0 | 12 | 100% | 71.4% | 23/28 |
| Primary, 28 notices | Qwen, initial prompt | 1 | 0 | 41 | 100% | 2.4% | 9/28 |
| Challenge, 9 notices | Rules | 17 | 0 | 21 | 100% | 44.7% | 2/9 |
| Challenge, 9 notices | Qwen, initial prompt | 8 | 1 | 30 | 88.9% | 21.1% | 1/9 |
| Follow-up, 20 notices | Rules | 10 | 0 | 13 | 100% | 43.5% | 13/20 |
| Follow-up, 20 notices | Qwen, initial prompt | 4 | 6 | 19 | 40.0% | 17.4% | 7/20 |
| Follow-up, 20 notices | Qwen, example prompt | 18 | 23 | 5 | 43.9% | 78.3% | 7/20 |

The primary Qwen precision is based on just one emitted assignment. The initial prompt also returned an empty array on a synthetic two-endpoint express notice. In a single-slot CPU replay with prompt caching disabled, two real notices reproduced their initial responses, and all four checked rendered prompts contained the requested system text, user notice and assistant start delimiter. This check reproduced the failure with one slot. It does not establish performance for other quantizations or serving backends. [Diagnostic inputs and outputs](diagnostic_replays.json)

**What changed with examples.** In event 249364, the example prompt correctly marked Hoyt-Schermerhorn Sts and Broadway Junction as local boundaries and Nostrand Av as the lower-level boarding location. In 250633 it recovered both express boundaries and labelled 155 St as skipped, although the notice gave 155 St only as the location of a brake investigation. In 251281 it copied five stations from an appended planned-work reminder and assigned them the opposite of the stated stopping instruction. All 23 extra follow-up assignments named strings present in the notice. Checking literal grounding alone would preserve these errors. [Paired outputs](followup_examples.json) · [Case-level errors](errors.csv)

**Mapping to real complexes.** Both methods used the frozen station directory and the same affected-route disambiguation. An unresolved name remains a failed complex prediction. One ambiguous reference assignment was masked as described above. On the follow-up, rules correctly mapped 8/22 complex–role labels, the initial Qwen prompt 4/22 and the example prompt 17/22. The corresponding extra assignments were 1, 6 and 23. The example-prompt complex-level F1 was 0.548 versus 0.516 for rules, while the rules had substantially greater precision. In the primary sample the rules' complex-level F1 was 0.806 versus 0.047 for the initial prompt. The three samples answer different development questions and are reported separately. [All metrics](scores.csv) · [Per-notice counts](case_scores.csv)

## Reproduction and evidence

The local CPU runs used the official Qwen2.5-7B-Instruct GGUF Q4_K_M release, revision `bb5d59e06d9551d752d08b292a50eb208b07ab1f`. The two shards had SHA-256 `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` and `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`. The Ubuntu x64 CPU archive for llama.cpp b12721 (commit `79c69f4b0`) had SHA-256 `ab56524ce899526e51c04e77f089a5792c59282b49736b239900970d3947be47`. Four slots shared a total context of 8,192 tokens. The run used six threads, temperature 0, seed 20261125 and a 384-token output limit. All 77 audit completions stopped normally. One syntactically valid JSON answer had the wrong nesting. These measurements come from local inference. [Official model revision](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/bb5d59e06d9551d752d08b292a50eb208b07ab1f) · [Runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b12721)

Start a local server in the unpacked runtime directory, with both verified model shards together:

```bash
./llama-server -m /path/to/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --alias nyc-qwen --host 127.0.0.1 --port 18443 -ngl 0 -np 4 -c 8192 -t 6 -tb 6 --no-warmup
```

From this folder, these commands reproduce raw answers in a separate directory. The score command reads the committed raw answers and verifies the station-directory hash. All cases include the original event/alert IDs, first-notice text and routes. The standalone sampling scripts accept the May first-notice source linked below.

```bash
mkdir -p /tmp/nyc-parser
python infer.py http://127.0.0.1:18443 /tmp/nyc-parser/qwen.json
python infer.py http://127.0.0.1:18443 /tmp/nyc-parser/challenge_qwen.json --cases challenge.json
python infer.py http://127.0.0.1:18443 /tmp/nyc-parser/followup_qwen.json --cases followup.json
python infer.py http://127.0.0.1:18443 /tmp/nyc-parser/followup_examples.json --cases followup.json --prompt prompt_examples.txt
python score.py /path/to/verified/stations /tmp/nyc-parser/scores
```

[May source manifest](../may_validation/sources.json) · [Fixed May forecast comparison](../may_validation/README.md)
