# How many first notices support a test of station roles?

**NYC development audit, 24 November 2026.** Across three consecutive local dates, 134 distinct MTA subway events had an initial notice. An internal review of the initial text identified nine events with an explicit pair of differently instructed places. Seven pairs retained the original event and both measured outcomes at both complexes after geographic matching and overlap removal. Two of those seven notices had identical initial wording and described the same A-line switch problem under different event IDs.

| Stage | Observations | Unit and denominator |
|---|---:|---|
| First notices, August 8, 9 and 10 | 25, 40 and 69 | 134 event IDs across three dates and 44 combinations of affected route codes |
| Revised strictly before the forecast hour | 91 (67.9%) | 134 first notices. A revision at the target boundary does not count. |
| Route-linked candidate observations | 6,183 | Event × complex × first complete target hour, on the seasonal timetable |
| After keeping the earliest notice per station-hour | 5,002 | 54 distinct target hours and 424 complexes. 931 station-hours had multiple incident IDs, and 122 IDs retained at least one row. |
| Subway count available | 4,943 (98.8%) | 5,002 unique station-hours |
| Catchment with preperiod dock and bike count | 3,666 (73.3%) | 5,002 unique station-hours, including 326 with zero recorded departures |
| Both counts available | 3,627 (72.5%) | 5,002 unique station-hours |
| Two named places with distinct instructions | 9 | Single-pass internal reading of 134 original headers and descriptions |
| Both role locations on the affected-route map | 8 | Nine annotated pairs |
| Both locations retain the incident and both counts | 7 | Nine annotated pairs |

The coverage denominator includes maintenance, marine bridge openings, unresolved causes and ordinary delay language. A wording screen marked 109 first notices as containing fault or response language, 20 as maintenance or projects, four as marine bridge openings and one as unclassified. Seven contained a separately marked planned-work reminder. Confirm scheduling and causation from independent operational records before assigning the unscheduled primary cohort. Native status alone gives a weak guide to that task. The archive labels 120 of the 134 initial notices simply “delays.”

**What counts as a role pair.** A pair needs two different named complexes with different operational instructions in the same initial notice. Examples include a boarding platform versus the boundary of a service change, a bypassed station versus a reroute boundary, and a short-turn terminus versus a reroute boundary. Two endpoints with the same boundary role do not qualify on that basis alone. Direction labels, incident locations, and instructions solely in a planned-work reminder do not qualify. The [nine supporting pairs](role_pairs.csv) specify one example per positive notice. This exploratory review included three previously traced incidents and has no independent second annotation. It establishes feasibility for a text audit. Scoring a role extractor requires a fixed vocabulary, independently reconciled annotations and case-level evidence spans. The full role model could also use single-station notices.

**Two different reasons for losing a pair.** For event 262432, Crescent St had no dock in its preperiod catchment, and an earlier alert claimed the Broadway Junction station-hour. For event 262567, the J/M notice explicitly named 2 Av as a short-turn destination. The base J/M schedule excluded it. Picking stations from the language model’s own output would give it different evaluation cases. Preserve the common affected-line population for the primary analysis. Evaluate off-line destinations separately on a geography rule specified before examining future outcomes. Events 262363 and 262384 also illustrate why resampling individual event IDs would overstate independent evidence.

**Time and geography.** The wider archive request covered August 8–14 local time plus one hour of follow-up. Its last returned subway version was on August 11 at 23:51 in the archive clock. We analyzed August 8–10, the first three complete local dates before that boundary. Later dates in the request have unknown coverage. A separate count query reproduced all 134 first rows and distinct IDs. We used the [August clock check](../README.md) and selected the complete hour starting at the next clock boundary. Event eligibility and station assignments use version zero. Later versions enter only the revision-timing diagnostic. Ties on issue time use ascending event ID.

The MTA's May 26 base timetable was publicly archived before these incidents and covers the August service dates. Its SHA-1 is `ed28414bf8c061381dd39ef7806e46d4e12d48b7`. We applied weekday service and date exceptions, included preceding-service-day trips crossing midnight, and took scheduled calls in either direction during the target hour for the original affected routes. We matched stop IDs to a station directory, checked physical coordinates within 100 m, and required the complex ID in the August 6 entry geography. Every event had at least one scheduled candidate and all selected parent stops mapped. The directory crosswalk was checked retrospectively. The August 1 and August 8 timetable revisions still require a byte-level comparison, including temporary exceptions, before using this population in a confirmatory protocol. [Seasonal GTFS version](https://www.transit.land/data/feeds/f-dr5r-nyctsubway/versions/ed28414bf8c061381dd39ef7806e46d4e12d48b7) · [August 8 GTFS version](https://www.transit.land/data/feeds/f-dr5r-nyctsubway/versions/5116d5c7bc4afeddf53f34dcbc2c29363e84b2a2)

**Outcome accounting.** Hourly fare aggregates sum ridership. Dock assignments use August 1–7 origins, the nearest complex within 500 m and unique ride IDs, as in the five-trace sample. Of the 5,002 station-hours, 1,316 had rail only, 39 bike only, and 20 neither. A missing total remains blank. A recorded zero at mapped docks cannot establish their operating status. An independent SQL recount reconciled the event links, unique observations, missing counts and bike zeros. All six modal totals from the three overlapping original traces reproduced exactly. Source snapshots can change. Compare hashes on replay.

**Decision.** Keep February 2026 as development and audit all four weeks before allocating model training. A status-only query under the provisional winter UTC conversion returned 1,752 first-notice rows, including 1,435 labelled simply “delays.” Reading only the other status groups would miss some operational instructions. Verify the winter clock, obtain the applicable dated timetable, distinguish planned changes, group repeated incidents, and estimate how many independent role comparisons and complete time blocks survive. This audit measures availability and overlap. Forecast accuracy and the publication decision remain open.

[Event-level audit](events.csv) · [Station-hour counts](station_hours.csv) · [Source queries and hashes](sources.json) · [February planning inventory](february_inventory.json) · [Exposure record](../EXPOSURE.md)

To rebuild the August results, allow approximately 1.1 GB for cached source files and use Python 3.9 or later. The February inventory includes its own query URL.

```bash
python3 research/alert_study/audit_cohort.py /tmp/nyc-alert-inputs /tmp/nyc-august-cohort
```
