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
