# May validation population and frozen controls

The registered join links 1,777 original NYC subway notices to 4,231 versions archived during May. The April 29 geometry contains 424 station complexes. Using dock locations observed April 24–30, 1,544 docks were assigned to 295 complexes. The May trip archive contains 4,694,366 distinct starts and records at least one citywide start in each of the 744 hours. A zero counts recorded departures in an assigned catchment. A complex without assigned docks has a missing bike outcome. [Source URLs and SHA-256 hashes](sources.json) · [Dock assignments](docks.csv) · [Version ledger](versions.csv.gz)

| Stage | Primary station-hours | Assigned incidents | Dates |
|---|---:|---:|---:|
| Route and timetable join, earliest eligible notice per complex-hour | 66,123 | 1,611 | 31 |
| Both target outcomes observed | 46,842 | 1,593 | 31 |
| Both targets and weekly lags observed | 46,621 | 1,593 | 31 |
| Same paired rows after excluding parser-audited notices, exact inputs and recurrence groups | 44,860 | 1,534 | 31 |
| Paired rows with fault or response wording | 35,791 | 1,231 | 31 |
| Paired rows with a rule-extracted local role | 242 | 151 | 31 |

The timetable availability filter retained 1,743 original notices. Of these, 1,731 reached at least one station before competing notices were resolved. Multiple alerts overlapped on 12,650 of the 66,123 primary station-hours. The separate May 17 timetable sensitivity contains 34 originals and 832 complete station-hours assigned within those notices. Those rows have their own earliest-notice assignment. For a combined sensitivity, resolve overlapping keys across the two populations using original issue time and numeric event ID. [Primary panel](primary.csv.gz) · [Schedule sensitivity](schedule_excluded.csv.gz) · [Incident mapping](events.csv) · [Coverage](coverage.json)

## One notice with distinct station roles

Only one primary incident had complete outcomes and lags at two or more assigned stations with disjoint rule-extracted roles. On May 27 at 15:03 New York time, A-train notice 515127 reported a suspension between 168 St and Inwood–207 St and early turnbacks at 145 St or 168 St. At 15:15, revision 515131 reported delays after a train had been removed from service. The forecast uses the original text and predicts the 16:00–17:00 hour. The table gives the roles the rules actually extracted. The original text also allowed a turnback at 168 St.

| Station | Extracted role | Assigned docks | Subway entries | Bike departures |
|---|---|---:|---:|---:|
| Inwood–207 St (143) | Suspension boundary | 4 | 358 | 7 |
| 145 St (151) | Short turn | 7 | 904 | 26 |
| 168 St–Washington Hts (605) | Suspension boundary | 8 | 2,114 | 21 |

A second read of the trip archive counted unique ride starts at the 19 docks for this hour. Separately queried fare-category records reproduced all three subway counts. Revision 515131 preceded the target hour, so this trace illustrates how exposure defined at issue can differ from service during the outcome hour. [MTA version history](https://data.ny.gov/resource/7kct-peq7.json?event_id=251649) · [Station totals](../trace/stations.csv) · [Individual docks](../trace/docks.csv) · [Verification queries](../trace/sources.json)

## Forecast files

The operator comparison fitted native/cue, full-text and full-text-plus-rule controls on 190,606 paired January–April rows after purging 41 rows linked to a May recurrence group. Each mode has those three predictions and its weekly baseline on the same 46,621 validation keys. The lag-free sensitivity fitted on 146,434 January–March rows with both outcomes and predicted raw counts from native, text and rule inputs. Its comparator uses a training-only station/hour-of-week median with station and mode fallbacks. The weekly lag played no part in the lag-free training filter, features or objective. Prediction files contain identifiers and forecasts. Forecast-error comparisons follow the complete Qwen extraction and forecast freeze. [Prediction files and checksums](../controls) · [Registered comparisons](../README.md)

From the repository root, reproduce the May joins and controls with a source-cache directory holding the [winter/spring inputs](../../winter_spring/sources.json). The panel builder retrieves missing May sources. The forecast script also needs the four earlier monthly alert-version archives and preperiod coordinates listed in that manifest.

```bash
root=research/alert_study/may_validation
python "$root/build_panel.py" "$CACHE" "$root/panel"
python "$root/trace_pair.py" "$CACHE" "$root/trace"
python "$root/freeze_controls.py" "$CACHE" "$root/controls"
```
