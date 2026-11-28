# How many May notices could support a station-role exchange?

The frozen exchange test selects at most one pair per first notice. Each station must appear in the notice text, resolve to a complex under the frozen dictionary, and have a retained station-hour. We counted every station meeting those necessary conditions before opening the Qwen forecasts. We included overlapping dictionary names and names appearing anywhere in the original header and description, including planned-work reminders. A station mention can therefore qualify even when the message gives it no operational role.

| May population | Paired notices | Paired station-hours | Station-hours with a resolvable mention | Notices with at least one | Notices with at least two | Dates with at least two in one notice |
|---|---:|---:|---:|---:|---:|---:|
| All | 1,593 | 46,621 | 1,254 | 996 | 166 | 31 |
| Removing parser-audit notices, matching texts and linked recurrence groups | 1,534 | 44,860 | 1,163 | 947 | 137 | 29 |
| Fault/response wording | 1,231 | 35,791 | 946 | 802 | 98 | 28 |

Of the 1,593 notices, 597 had zero retained complexes with a resolvable name, 830 had one, 99 had two, 52 had three, five had four and ten had five. The 166 possible exchange notices corresponded to 164 of the study's text-and-route recurrence groups. Distinct notice IDs and distinct wording groups can still refer to the same disruption. The exchange comparison can include at most 332 station-hours; actual inclusion requires two disjoint, nonempty role assignments from Qwen. These limits apply to direct station mentions and the exchange test. They do not bound differences in predictions between separately trained forecasting models.

The script matches each alias independently against normalized original text using the extraction verifier's boundaries, slash/hyphen variants and route-based disambiguation. It selects station IDs from the frozen operator rail forecast keys. An independent enumeration of text spans by dictionary-name length, joined to the lag-free bike forecast keys, reproduced all 1,593 eligible station sets. Neither calculation used forecast errors. [Per-notice support](events.csv) · [Protocol](../README.md)

Run from the repository root using the two files pinned in the [source inventory](../panel/sources.json).

```bash
root=research/alert_study/may_validation
python "$root/name_support.py" "$CACHE/may_notices.json" "$CACHE/stations" "$root/name_support/events.csv"
```

## How many mentions describe different instructions?

One internal AI-assisted reader reviewed 32 notices selected by a fixed hash ordering of the 166 candidates. The protocol was written before displaying the selected texts. The reader saw the original wording and eligible station identities. The 32 decisions were saved before joining any counts or later revisions for this audit. Five notices were already linked to earlier parser-audit exposure. The sample was drawn from an exposed development month.

| Reading of eligible stations in the first notice | Notices |
|---|---:|
| Two stations had disjoint, nonempty role sets | 3 |
| Two or more stations had roles, all sharing a role | 20 |
| Only one station had a current role | 5 |
| No station had a current role under the reminder exclusion | 4 |

Three of 32 (9.4%) supported the exchange. Two of the three belonged to the five previously flagged notices. Of the other 27, one supported an exchange. In one negative case the connection between a planned-work reminder and the reported disruption was unclear. All four stations in that reminder had the same skipped role, so either reading yields the same exchange decision. These are single-reader annotations under the study vocabulary. Qwen's actual assignments and their predictive value require the frozen comparison. [Selection and reading protocol](review_protocol.md) · [All 32 readings and IDs](review.csv)

The three qualifying notices all received a revision before the full forecast hour. The two versions in each event history explicitly described a return to at least part of the scheduled service. Minutes below are elapsed times; all times are local to New York.

| Date | Initial and revised alert IDs | Original distinction | Minutes to revision | Forecast hour begins |
|---|---|---|---:|---|
| May 2 | 507774 → 507777 | Local segment boundary at Hoyt-Schermerhorn Sts and lower-level boarding at Nostrand Av | 25 | 10:00 |
| May 23 | 514250 → 514251 | Express segment boundaries at 125 St and Grand Central-42 St and lower-level boarding at 86 St | 4 | 15:00 |
| May 27 | 515127 → 515131 | Suspension boundary at Inwood-207 St and short turn at 145 St | 12 | 16:00 |

On May 23 the revision reported local service and upper-level boarding at 14:20, forty minutes before the forecast hour. The 15:00 hour recorded 668, 1,969 and 3,814 entries at 125 St, 86 St and Grand Central-42 St, respectively. Reaggregating 23 fare-category records reproduced all three values. Recounting distinct rides in the May Citi Bike trip archive reproduced five, five and 17 departures from the nine, four and ten assigned docks at these stations. The counts describe observed use; the contrast between initial and revised instructions does not identify a causal demand change. [Version IDs, station IDs, dock IDs and counts for all three notices](review_traces.csv) · [Fare query and checksum](may23_fare_source.json)

### Replaying the observed counts and weekly inputs

Across the eight traced station-hours, 55 fare-category records and 123 distinct bike starts reproduced all 16 target values. At the same local hours a week earlier, 59 fare records and 263 distinct rides reproduced the 16 weekly inputs. The eight stations account for 51 mapped docks. The check used the fixed May dock assignment in both weeks, deduplicated by ride ID, and included mapped docks with zero starts. All 32 values agreed exactly with the frozen panel. The weekly replay uses retrospective captures; public availability at the first issue time remains subject to the separate [timing audit](../public_timing/README.md). [Station totals](raw_counts/counts.csv) · [Departures per dock](raw_counts/docks.csv) · [Six fare queries and two trip archives](raw_counts/sources.json)

```bash
root=research/alert_study/may_validation
python "$root/check_review_counts.py" "$CACHE" "$root/name_support/raw_counts"
```
