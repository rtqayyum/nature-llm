# February development audit and weekly forecasting baseline

**24 November 2026.** All 1,752 original notices from February 1–28, New York local time, were inventoried. Across 67,824 unique station-hours linked to the winter base timetable, 44,938 had recorded targets in both modes. A weekly seasonal forecast could also obtain both lagged counts for 32,903 station-hours in February 8–28. Its mean absolute error was 93.10 subway entries and 3.61 bike departures per station-hour. The language comparison remains open.

## From 1,752 notices to 14 usable reviewed pairs

| February, local issue date | First notices | Internally confirmed different-role pairs | Both places on original affected routes | Both places retained with both counts |
|---|---:|---:|---:|---:|
| 1–7 | 443 | 6 | 5 | 2 |
| 8–14 | 455 | 9 | 9 | 5 |
| 15–21 | 410 | 2 | 2 | 1 |
| 22–28 | 444 | 14 | 13 | 6 |
| Total | 1,752 | 31 | 29 | 14 |

An outcome-blind cue screen used the header and description available in version zero. It removed separately labelled planned-work/service reminders. It selected 40 notices with at least two of four cue groups for boarding/transfer, turnbacks, bypasses and an indicated segment. One internal AI-assisted reading confirmed 29 of those. We also read 20 screen-negative notices per week, ranked by the SHA-256 of a fixed prefix and event ID. Two of those 80 described a qualifying pair. Both used alternative-service instructions. The remaining 1,632 were unreviewed, so 31 is a documented minimum. The pairs and exclusions are recorded against source IDs and the two station mentions. The review assessed one example per positive notice. Mentions used only for a train direction, an incident cause or two termini with the same role did not qualify. No independent second annotation was performed. [Reviewed cases](review.csv) · [Sampling and case-level decisions](events.csv)

Two pairs fell outside the affected-route population. Among the 29 on-map pairs, eight lost a station to an earlier overlapping notice, five lacked a modal target, and two had both problems. The 14 surviving incidents cover ten dates and seven distinct unordered station pairs. Six of the 14 contain language matched by the fault/response screen, four match maintenance/project language, and four remain unclassified. Complete cases therefore still require review for unscheduled eligibility. The base timetable and the operator's effective schedule agreed on the binary availability of both named stations for all 31 positive cases, including the two exclusions. The cross-check used revenue calls on the named routes during the target hour. It covered reviewed pairs only. The effective archive is a retrospective check and has no feature role. [Scheduled calls for each pair](effective_schedule.csv) · [MTA base and supplemental schedules](https://data.ny.gov/Transportation/MTA-Subway-Schedules-2026/g8es-h7gb)

## Forecast errors on identical station-hours

Each forecast repeats the count at that complex during the same local hour seven days earlier. All four counts must be available for the row to enter either mode's error. We weight each retained station-hour equally. Target hours during the first week supply history. The final three notices forecast March 1 and were retained in the inventory but purged from February evaluation.

| Target dates | Matched station-hours | Subway MAE, entries | Bike MAE, departures |
|---|---:|---:|---:|
| Feb 8–14 | 13,000 | 61.82 | 2.70 |
| Feb 15–21 | 11,142 | 113.06 | 3.52 |
| Feb 22–28 | 8,761 | 114.13 | 5.08 |
| Pooled | 32,903 | 93.10 | 3.61 |

A separate SQL calculation on the released rows reproduced the sample sizes and absolute-error sums of 3,063,320 entries and 118,807 departures. Two disaggregated MTA queries for complex 28 on Feb 1 and 8 at midnight reproduced 108 and 104 entries respectively. Differences between weeks are descriptive. The incident mix, demand, coverage and calendar vary. Only three target weeks contribute. The comparison assumes operator access to lagged aggregates because public monthly trip files arrive later. [Exact errors](weekly_naive.csv) · [Paired counts and forecasts, gzipped CSV](station_hours.csv.gz)

## Timing, geography and exposure

- All 1,752 originals agree with the earlier independent count by native status. Of these, 1,024 had at least one later version strictly before the next complete local hour began. Three target windows crossed the month boundary.
- Six NYC subway alerts on February 3 matched the same line, station, direction and instruction in a separately timestamped public recap. The recap was 3–5 minutes later than the MTA timestamp interpreted as UTC and converted to Eastern Standard Time. For example, alert 482609 maps from 06:22 UTC to 01:22 EST and the corresponding recap entry is 01:25. The other IDs were 482629, 482633, 482836, 482842 and 482845. This corroborates the winter hour offset. Exact first receipt, summer/winter transition hours and the timestamp definition across other archives require separate checks. [Local-time recap](https://www.reddit.com/r/nyctransitalerts/comments/1qvg4c1/alert_summary_for_232026/) · [MTA example](https://data.ny.gov/resource/7kct-peq7.json?alert_id=482609)
- The MTA winter base timetable recovered from a February 25 Internet Archive capture has SHA-1 `b3b3f88db7d48efc5c9ab52e560efca52bccc49f`. The bytes match the version Transitland recorded on December 4, before every forecast origin. Its service calendar covers December 8 through May 16. The next catalogued version begins service March 1. The audit applied service-day exceptions, cross-midnight calls and both directions; stop IDs were cross-checked against the station directory and January 28 complex geography. Ten notices contain unmapped Staten Island stop IDs. [Dated feed identity](https://www.transit.land/data/feeds/f-dr5r-nyctsubway/versions/b3b3f88db7d48efc5c9ab52e560efca52bccc49f) · [Public archive and data query hashes](sources.json)
- The 82,540 event-complex links reduce to 67,824 unique station-hours across 593 target hours and 424 complexes. The earliest original notice receives each overlapping row, with event ID breaking ties. The panel includes 22,366 rail-only rows, 255 bike-only rows and 265 with neither outcome. Missing target values are blank. A zero at assigned docks is a count of recorded starts. Independent dock operating status is unavailable.
- For a reproducible recurrence diagnostic, we joined consecutive event IDs with identical normalized initial headers and route sets, at gaps of at most 6, 24 or 72 hours. This produced 1,736, 1,718 and 1,696 groups respectively. Even under the 24-hour rule, all 14 usable role cases have separate group IDs. Seven distinct station pairs and repeated service patterns show why those group IDs alone cannot establish statistical independence.
- The primary dock map used January 25–31 origins and nearest-complex assignment within 500 m. Citi Bike reported snow recovery affecting dock access during this preperiod. Rebuilding with January 18–24 or all January increased complexes with mapped docks from 290 to 291, and paired-outcome rows from 44,938 to 45,119. All three maps retained the same number of usable role pairs. This development sensitivity measures geographic availability only. [Map sensitivity](dock_sensitivity.csv) · [Operator's February 5 recovery notice](https://citibikenyc.com/blog/winter-recovery-2026)

## Next experiment

Use the measured baseline to start an earlier-training/later-validation NYC pilot. Build the January–April development panel and run the same forecasting head with history/calendar, native alert fields plus simple text, and station names with and without roles. Keep candidate locations and scoring masks identical. Broaden the text screen to include alternative-service wording, obtain a second outcome-blind check on the labelled cases, and group related disruptions across split boundaries. Compare on May validation under operator-lag and delayed/public information regimes. Report physical-unit errors and paired time-block uncertainty before deciding whether a Qwen comparison is warranted. The current February cases alone cannot determine a journal submission or model superiority.

Rebuild using Python 3.9+ from the repository root. The second command checks the effective timetable against the rebuilt event list. The checked-in review is an input.

```bash
python3 research/alert_study/audit_february.py /tmp/nyc-feb-input /tmp/nyc-feb-results
python3 research/alert_study/check_february_schedule.py /tmp/nyc-feb-input /tmp/nyc-feb-results
```

[Alert-study design](../../MULTIMODAL_FORECASTING_DIRECTIONS_20261123.md) · [Exposure record](../EXPOSURE.md) · [Effective-schedule query hashes](effective_sources.json)
