# Boston development source check

Selected May 14, 2025 (Boston local time) for a one-day feasibility check before inspecting Boston notice or demand rows. Reserve the entire Boston May 2025 block for development. Use original notices for Red, Orange or Blue Line service, exclude facility-only advisories, distinguish archived versions from repeated affected-entity and active-period rows, and retain the first complete local hour after issue. For dock geometry use only starts in the preceding seven days. Check rail and bike coverage without choosing events by the magnitude of their outcomes. Keep unknown gate status explicit.

## May 14 trace

The archive contains 963 entity/period/version rows for 254 alert IDs created on this date. Nine IDs specified a Red, Orange or Blue Line route in their earliest archived version. Six concerned facilities. All three remaining originals had a modification timestamp and a push timestamp equal to creation. They named stations in the message text and supplied no native stop ID. The seven modification records for these three alerts comprise three originals, one wording revision and three closures. Timestamp agreement supports reconstruction at issue. An independent delivery receipt was unavailable.

| Alert ID | Issue, EDT | Named station | Target hour, EDT | Assigned docks | Gated entries | Recorded bike starts |
|---|---:|---|---:|---:|---:|---:|
| 643641 | 05:41:20 | Chinatown | 06:00–07:00 | 2 | 64 | 6 |
| 643648 | 06:43:00 | Chinatown | 07:00–08:00 | 2 | 104 | 5 |
| 643879 | 19:45:20 | Central | 20:00–21:00 | 5 | 320 | 37 |

The two Chinatown alerts refer to a signal problem, with the second calling it ongoing. The Central alert describes a medical emergency. At 20:00:17 its wording changed to say service was proceeding, 17 seconds into the target hour. All three original station mentions locate the cause of delay. They provide no contrast between two stations given different passenger instructions. [Version history](versions.csv) · [Station-hour counts and weekly lags](traces.csv) · [Assigned docks](docks.csv) · [Departures per dock](dock_hours.csv) · [Half-hour gate records](half_hours.csv)

## May service-change screen

Following the one-day trace, we made an exploratory text inventory of the same development month. The archive had 95 non-facility originals with an explicit Red, Orange or Blue route ID, all with text at the creation timestamp. We screened the first header and description for operational wording and also retained every effect other than `OTHER_EFFECT`. The [screen and exact text hashes](month_screen/origins.csv) and [review of the 11 candidates](month_screen/review.csv) document this selection. Seven first notices announced future work. Four described immediate replacement or suspended service. This targeted review can miss role-bearing wording outside the screen.

The four immediate notices represented two described disruptions. The May 20 Braintree switch problem produced one notice. The May 22 flooding near Ruggles produced three IDs as the shuttle and suspension boundaries changed. We recorded named segment boundaries and explicit alternative itineraries before joining those cases to outcomes. The 12 selected station-hours all had gate counts and same-hour weekly gate lags. Ten also had bike starts and weekly bike lags. **All ten complete rows came from the May 22 flood.** Braintree and Quincy Adams had no mapped Bluebikes dock within 500 m on the May 20 map. Their bike outcomes are missing. [Station readings](month_screen/station_readings.csv) · [Counts, lags and missing values](month_screen/traces.csv)

At 20:14:29 on May 22, alert 645688 directed passengers to take the Green Line from North Station to Copley, walk to Back Bay and board a shuttle toward Jackson Square. The four stations had five, four, two and two assigned docks respectively. In the 21:00–22:00 hour they recorded 331, 469, 12 and 10 gated entries and one, one, zero and zero bike starts. Contemporary coverage independently corroborates the Green Line and replacement shuttle instructions. Gate operating status during that hour remains unknown. The 12 entries at Back Bay quantify recorded faregate interactions. They cannot establish how many passengers used the shuttle or how demand changed. [Contemporary report](https://www.boston.com/news/transportation/2025/05/22/mbta-shuttle-buses-green-line-replace-orange-line-service-due-to-noreaster-flooding/) · [Half-hour components by line](month_screen/half_hours.csv) · [Starts at each dock](month_screen/dock_hours.csv)

A revision explains why that distinction matters. At 20:44:15, alert 645688 moved the suspension endpoint from Jackson Square to Back Bay before the 21:00 target hour. It retained all 22 distinct informed-entity records and the detour effect. Its numeric severity increased from 7 to 10. The two endpoint stations both had gate and bike outcomes. An updated-forecast comparison can hold the newly available severity and other native fields constant in both model arms while comparing the previous and revised texts. The original and revised records alone cannot isolate the contribution of wording. [All four incident histories and entity hashes](month_screen/version_history.csv)

An independent reaggregation from the full source archives reproduced all 30 target and lag gate totals, all 26 observed target and lag bike counts, and all four missing bike values across the one-day and immediate-instruction traces. A separate spatial nearest-neighbor implementation also reproduced every reported catchment size.

**Decision.** Boston qualifies for an auditable source-linkage pilot. This month supplies one documented disruption with fully paired immediate-instruction examples under the screen. A replication needs more independent disruptions, explicit grouping of related IDs, a frozen out-of-development period and an assessment of gate coverage during disruptions. May 2025 belongs to Boston development. This audit produced no Boston forecast-accuracy estimate.

## Source handling and reproduction

- Use local creation dates and epoch modification times. Collapse repeated entity and active-period rows within an ID and modification time before counting versions. Freeze the first text. Passenger instructions and incident-location mentions are recorded separately. [MBTA LAMP](https://performancedata.mbta.com/) · [Source URLs, sizes and full capture hashes](sources.json)
- For each May 14, 20 and 22 map, resolve named locations to GTFS parent stations active that date. Compute each dock's modal coordinate, rounded to four decimals, from unique ride starts in the preceding seven local days. Break a modal tie by coordinates. Assign each dock to its nearest light- or heavy-rail parent within 500 m, breaking an exact distance tie by parent ID. Apply that map to both target and seven-day lag. A bike zero requires a mapped dock and observed citywide starts in that hour. The trip CSV supplies offset-free timestamps, which we interpret as Boston local time. All selected dates are outside daylight-saving transitions. The trip-derived map and weekly counts are retrospective. Their availability through public releases at the original forecast times has not been established. [Preperiod dock geometry and distances for the role cases](month_screen/docks.csv) · [Bluebikes archive](https://s3.amazonaws.com/hubway-data/index.html)
- Read both half-hour records and combine the published line components at split stations. Require the same nonempty set of reported lines in the two halves. These are retrospectively published, unscaled gated entries. MBTA documents that open gates during emergencies can cause undercounting. The outcome file provides no independent gate-state indicator for these rows. [MBTA gated entries and data dictionary](https://www.arcgis.com/home/item.html?id=7859894afb5641ce91a2bb03599fdf5b)
- The [source projections](inputs) preserve the fields needed for replay if the full MBTA downloads change. They contain the selected hours from the annual gate file, relevant dated station records, every row for the May 14 identity inventory, and the May cohort's heavy-rail first versions plus the four immediate service-change histories. Unused facility and other-mode texts are omitted. The original monthly Bluebikes file is required separately. The [projection builder](prepare_sources.py) checks the full capture hashes.

From the repository root with Python 3.9+ and PyArrow installed:

```bash
root=research/alert_study/boston_sample
cache=$(mktemp -d)
cp "$root"/inputs/{alerts.parquet,stops.parquet,gates.zip} "$cache/"
curl -fL --retry 3 https://s3.amazonaws.com/hubway-data/202505-bluebikes-tripdata.zip -o "$cache/bikes.zip"
python "$root/check_sample.py" "$cache" "$cache/day"
python "$root/check_month.py" "$cache" "$cache/month"
```
