# Four months of NYC alert forecasts

The January–April 2026 development panel links 7,273 original subway notices and their 17,287 archived versions to scheduled station complexes, preperiod Citi Bike docks and hourly outcomes. The rules identified 34 incidents with two stations whose recognized roles did not overlap. Nineteen incidents on 18 dates across 11 calendar weeks kept such a pair after timetable, overlap and count checks. The measured constraint is the number of complete within-incident comparisons, even though the network panel contains 282,280 unique station-hours.

## One incident from notice to counts

On April 4 at 03:49 New York time, initial A-train notice 499700 named Utica Av and Hoyt-Schermerhorn Sts as the ends of an express segment and instructed riders at Nostrand Av to board on the upper-level express platform. The target ran from 04:00 to 05:00. Revision 499711 was archived at 05:57 local time. All three locations had six revenue calls in the retrospective effective timetable for that hour.

| Station complex | Extracted instruction | Assigned docks in April | Subway entries | Recorded bike departures |
|---|---|---:|---:|---:|
| Hoyt-Schermerhorn Sts (175) | Express boundary | 3 | 11 | 0 |
| Nostrand Av (179) | Boarding platform | 6 | 55 | 1 |
| Utica Av (181) | Express boundary | 11 | 41 | 1 |

The closest assigned dock at Nostrand Av was 4214.03, 71.8 m from the complex coordinate. The bike outcome sums starts at all six assigned docks. The zero at Hoyt-Schermerhorn means that the monthly trip archive recorded no start at those docks during the target hour. Dock operating status is unavailable. [MTA versions for event 244445](https://data.ny.gov/resource/7kct-peq7.json?event_id=244445) · [Dock assignments and distances](docks.csv) · [Effective calls](effective_calls.csv)

## Coverage and seasonal baseline

Counts use the first complete New York clock hour after the original notice. Each station-hour is assigned once, to the earliest eligible notice on a scheduled route. The numerical baseline repeats the same local hour and catchment seven calendar days earlier. Both modes use exactly the same rows, with both target and both lagged counts observed. Errors are mean absolute errors (MAE) in entries or departures per complex-hour.

| Month | Original notices | Unique station-hours | Both targets | Both targets and lags | Weekly subway MAE | Weekly bike MAE |
|---|---:|---:|---:|---:|---:|---:|
| January | 1,927 | 73,530 | 50,434 | 50,211 | 116.35 | 5.67 |
| February | 1,752 | 67,926 | 45,017 | 43,925 | 103.98 | 3.53 |
| March | 1,821 | 72,071 | 50,983 | 48,533 | 67.77 | 6.89 |
| April | 1,773 | 68,753 | 48,437 | 47,978 | 80.88 | 7.45 |
| Total or pooled MAE | 7,273 | 282,280 | 194,871 | 190,647 | 92.21 | 5.93 |

The continuous series adds 102 February station-hours from January 31 notices. Every one of the earlier 67,824 February station-hours matched its complex, notice ID and both outcomes, including missing values. January now supplies February's first week of lags. Two April 30 notices have May targets and were excluded from the panel. Public archives can arrive after the forecast origin. The weekly lag and fit through March assume timely operator access to counts. [Monthly denominators](monthly.csv) · [Origin and target keys](events.csv) · [All version keys](versions.csv.gz)

**Dated geography and time.** Each month's dock assignments use observed start coordinates from the last seven days of the preceding month, matched to the nearest complex within 500 m. Complex coordinates come from 08:00 and 17:00 on the preceding month's last Wednesday. We apply that same fixed map to target and lag. The four maps cover 291, 290, 292 and 294 complexes respectively. A mapped catchment with no recorded start receives zero only during a citywide hour with trip records. A subway aggregate missing from the rail archive is blank. Availability variations can therefore affect which events survive. We cannot determine passenger mode switching from these counts.

The winter base GTFS matches the [December 4 catalogued version](https://www.transit.land/data/feeds/f-dr5r-nyctsubway/versions/b3b3f88db7d48efc5c9ab52e560efca52bccc49f). The March 1 base timetable was archived February 27, SHA-1 `f51d0e0444d677217e631b0fd1744414c2185207`. We applied each seasonal base to its service days, including previous-day winter calls after midnight on March 1. Stop mapping retains the retrospective station-directory caveat. Supplementary timetables and later amendments may differ from the fixed bases. The source clock is interpreted as UTC on the basis of earlier winter and summer checks. Targets are advanced in UTC before conversion to local time. For example, a 01:44 EST origin on March 8 maps to 03:00 EDT. The missing 02:00 local hour is not imputed.

**Support for comparing roles.** Of the 34 incidents with disjoint role sets, 24 retained a pair after mapping and earliest-notice assignment. Both outcomes and weekly lags were available for 19. Allowing partially overlapping but unequal role sets gives 55 candidates, 42 after assignment and 31 fully matched on 27 dates. In separate queries of the effective timetable, every one of the 19 and 31 respectively retained a qualifying pair with scheduled revenue calls on the affected routes. The check covered 147 candidate station–incident links in 55 incidents. These retrospective calls validate scheduled service and do not enter any model. The 19 strict cases are contained in the 31 broader cases. Repeated routes, language and dates limit how much independent information they provide. [Incident-level attrition](pair_support.csv) · [Retrospective query manifest](effective_sources.json)

## April chronological forecast check

Earlier complete targets supplied 142,669 training station-hours associated with 4,774 notices. We retained the previously specified weekly lag, calendar, station, native fields, wording cue, initial-text TF-IDF and per-mode ridge regression with penalty 10. The initial-text vocabulary was fitted on distinct training notices. A 24-hour recurrence rule for repeated headers and route sets was checked against the full April inventory and removed zero training rows. April forecasts used 47,978 distinct station-hours from 1,606 notices over 30 dates. The extractor itself was developed on January–April wording. This is a development backtest.

| Forecast inputs | Network subway | Network bike | Rule-named subway | Rule-named bike |
|---|---:|---:|---:|---:|
| Weekly seasonal | 80.88 | 7.454 | 127.55 | 9.868 |
| Weekly count, station and calendar | 107.86 | 10.094 | 170.71 | 12.049 |
| Add native fields and a broad wording cue | 110.12 | 10.109 | 176.85 | 12.744 |
| Add full first-notice text | 121.14 | 9.945 | 203.36 | 12.297 |
| Add rule-named station identity | 121.16 | 9.945 | 215.75 | 12.676 |
| Add the local role labels | 121.15 | 9.945 | 213.90 | 12.700 |

The rule-named subset contains 219 station-hours from 126 incidents on 29 dates. Adding roles to the names-and-text model reduced pooled subway error by 1.84 entries and increased bike error by 0.024 departures. Omitting one date at a time changed the subway contrast from a 3.11-entry gain to a 2.33-entry loss. The seasonal baseline had lower error than every ridge arm in both modes at these locations and across the network. The February benefit from identifying the station in the text model did not recur on either April outcome. Five April incidents offered a strict within-incident comparison. A claim about Qwen's predictive value requires an actual Qwen comparison. [All April scores](april_scores.csv) · [Paired predictions for named stations](april_named_predictions.csv)

## A stronger seasonal control

After inspecting the April ridge results, we tested a separate tree model that predicts a correction to the weekly count in physical units. All arms used one histogram gradient boosting fit per mode with absolute-error loss, 100 iterations, learning rate 0.05, up to 15 leaves, at least 50 training rows per leaf, L2 penalty 1, and no early stopping. Base inputs were the lagged count, preperiod station coordinates, hour, weekday, routes, status and the coarse wording cue. Text features were 16 truncated-SVD components of train-fitted TF-IDF, computed on distinct training notices. The model already receives the station coordinates at every row. The names variant adds an indicator that this location appeared in a recognized instruction. The role variant also adds eight training-vocabulary role indicators. Neither changes the scoring set. Negative forecasts are clipped to zero.

| Inputs | Network subway | Network bike | Rule-named subway | Rule-named bike |
|---|---:|---:|---:|---:|
| Weekly seasonal | 80.882 | 7.454 | 127.548 | 9.868 |
| Boosted seasonal correction, native fields and cue | 81.040 | 7.395 | 127.852 | 9.628 |
| Add full-text features | 80.963 | 7.388 | 127.276 | 9.635 |
| Add station mention and roles | 80.963 | 7.388 | 127.276 | 9.635 |

Adding the mention indicator and then the role indicators left every saved April prediction identical at ten decimal places. This finding is specific to this tree recipe and these sparse rule features. The boosted text model reduced bike MAE by 0.88% and increased rail MAE by 0.10% versus weekly seasonal on the network. The equal-mode reduction was 0.39%. A descriptive paired 95% interval ranged from −3.19% to +4.18% using 5,000 circular seven-day block resamples, carrying all stations and both modes on each date together. Three-day blocks gave −3.19% to +4.12%. Against the boosted native/cue arm on the identical rows, adding full text yielded a 0.09% equal-mode reduction (seven-day interval −0.15% to +0.30%). These intervals describe sensitivity within one development month and omit fitting and model-selection uncertainty. [Boosted scores](boost_scores.csv) · [Named-station paired predictions](boost_named_predictions.csv) · [Complete block summaries](boost_blocks.csv)

**Decision for the next test.** Retain weekly seasonal and boosted native/text forecasts as practical comparators. Before opening May outcomes, run an outcome-blind extraction benchmark for boarding, short turns, alternatives and overlapping roles on fixed notice IDs, and freeze the models, eligibility, selection procedure and information regimes. A 7B parser can then be compared directly with the rules on the same text and forecast rows. The current results do not justify 72B training or a journal superiority claim.

## Reproduce

From the repository root, with Python 3.10+, NumPy, pandas, SciPy and scikit-learn (1.8.0 used here). The build retrieves five monthly bike files and daily subway aggregates. Source hashes pin the exact inputs. Use a fresh cache for a new build.

```bash
python research/alert_study/build_winter_spring_panel.py /tmp/nyc-inputs /tmp/nyc-panel
python research/alert_study/check_panel_role_calls.py /tmp/nyc-inputs /tmp/nyc-panel
python research/alert_study/summarize_panel.py /tmp/nyc-panel /tmp/nyc-panel
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/alert_study/april_notice_pilot.py /tmp/nyc-inputs /tmp/nyc-panel /tmp/nyc-april
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python research/alert_study/april_seasonal_boost.py /tmp/nyc-inputs /tmp/nyc-panel /tmp/nyc-boost
python research/alert_study/april_block_summary.py /tmp/nyc-boost/boost_predictions.csv.gz /tmp/nyc-boost/boost_blocks.csv
```

The April ridge and boosting commands also emit the full row predictions. A separate SQLite calculation reproduced the monthly coverage and weekly MAEs, checked every prior February station-hour and both outcomes, and verified the effective-call pair counts. All 7,273 first notices matched the complete in-window version sequences and forecast clocks. The version table links event and alert IDs to each archived timestamp. Join the target-month station file by event ID, then the dock table by month and complex ID. The event table records unmapped stops and routes without calls. [Source URLs and hashes](sources.json) · [January](2026-01.csv.gz) · [February](2026-02.csv.gz) · [March](2026-03.csv.gz) · [April](2026-04.csv.gz)
