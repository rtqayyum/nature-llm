# Which archive inputs could have existed when a May alert was issued?

Of the 46,621 paired May station-hours, **801 (1.72%)** passed four necessary timing checks for reconstructing the operator inputs from the public archives. All 801 belong to 34 first notices on May 6. For every paired station-hour, at least one fare row used for the weekly lag was last updated after the notice. The archives do not establish what fare total a public user would have seen at issue. The registered full-month comparison measures forecasting under an operator-access assumption.

| Source timing at the original issue | Station-hours passing | Incidents with at least one passing station-hour |
|---|---:|---:|
| April Citi Bike object used for dock locations already had its recorded modification time | 41,419 | 1,406 |
| At least one MTA row for each of the 424 complex coordinates in the May map had been created | 38,061 | 1,293 |
| Citi Bike trip object for the same hour one week earlier already had its recorded modification time | 6,225 | 216 |
| All current MTA fare rows contributing to the complex's weekly rail lag had been created | 2,627 | 99 |
| All four conditions together | 801 | 34 |
| All contributing fare rows had also reached their current last-update time | 0 | 0 |

**What the timestamps measure.** We matched the original issue instant in the frozen ledger to the 46,621 unique complex-hour keys in the operator forecast file. For each lag we subtracted seven local calendar days from the target hour. On the 31 lag dates we queried the earliest current fare-row creation. Only April 29, May 13 and May 20 had any matched origins late enough to qualify. On those dates we retrieved the last creation and update times across all fare categories for each potentially eligible complex-hour. For the two geometry observations on April 29, we verified the full set of complex IDs and coordinates against both frozen source queries. The first supporting row for every coordinate was created May 6 at 10:38 EDT. These portal fields describe the lifetime of current rows. Earlier editions, identical values from other rows and operator feeds could support a different history. A later update can leave a count unchanged. [Socrata system-field definitions](https://dev.socrata.com/docs/system-fields) · [Detailed source requests and response hashes](detail_sources.json)

**Bike inputs.** We streamed the March, April and May trip archives, matched their full SHA-256 hashes and lengths to the study manifests and recorded the modification times returned with those bytes. The current objects are dated April 3, May 4 and June 4, respectively. For forecasts from May 8 onward, the weekly bike lag comes from the May file. The timestamps support a check on object vintage. They do not recover a historical access policy or rule out earlier copies. [Archive and fare-source receipts](metadata.json) · [Citi Bike system data](https://citibikenyc.com/system-data)

**A map for earlier issue times.** The preceding study map used March 25 fare coordinates and March 25–31 bike starts. A supporting row for each of its 424 complex coordinates existed by April 8. Compared with the May map, 1,529 dock IDs appeared in both and all retained the same complex. The May map added 15 IDs and lost 17. Of the 295 May complexes with assigned docks, 269 had identical dock sets. The other 26 account for 3,499 paired station-hours (7.51%). Using the older map could therefore change which trips count toward the bike outcome. A future public-input evaluation needs a map frozen from a dated public snapshot and labels built with the same catchments. Exact historical rail values still require as-of snapshots. The registered May population, labels and forecasts keep their original map. [Daily denominators](by_date.csv) · [Aggregate checks](summary.json)

Reproduce the counts from the captured source times using Python's standard library from the repository root.

```sh
a=research/alert_study/may_validation/public_timing
python "$a/analyze.py" "$a/metadata.json" "$a" /tmp/may-timing-check
diff "$a/summary.json" /tmp/may-timing-check/summary.json
diff "$a/by_date.csv" /tmp/may-timing-check/by_date.csv
```

To repeat the source acquisition, run the commands below into a fresh directory. The first reads approximately 1.66 GB of trip archives. Later changes to a public row or object can change a receipt. The analysis reads forecast keys and source timestamps. It does not calculate model errors or use a timing filter to select the primary results.

```sh
mkdir -p /tmp/may-timing-sources
python "$a/fetch.py" /tmp/may-timing-sources/metadata.json
python "$a/fetch_detail.py" /tmp/may-timing-sources/metadata.json /tmp/may-timing-sources
python "$a/analyze.py" /tmp/may-timing-sources/metadata.json /tmp/may-timing-sources /tmp/may-timing-sources
```
