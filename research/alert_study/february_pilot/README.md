# Does identifying a station's role improve the forecast?

In the February 2026 NYC development data, a model given both the station names and their reviewed roles averaged 144.42 entries and 5.53 bike departures in absolute error on the named comparison stations. Removing the roles gave 145.25 entries and 5.46 departures. Seven future incidents contributed these comparisons. Exchanging the roles between the two stations in each notice yielded a lower subway error again. These data give little reason to expect a consistent advantage from the specific roles. The larger multi-month experiment remains open.

## Paired results

All values are mean absolute error per station during the first complete local clock hour after the original notice. Smaller values indicate better forecasts. Each column uses the same observations for every applicable model. The pair columns comprise 14 station-hours from seven incidents, five distinct unordered station pairs and four dates. The network columns comprise 19,782 station-hours from 675 initial notices across 13 dates. All station-hours are part of the alert-exposed population. The cohort includes both maintenance and disruptions.

| Inputs | Network subway entries | Network bike departures | Named-pair subway entries | Named-pair bike departures |
|---|---:|---:|---:|---:|
| Same hour one week earlier | 113.73 | 4.219 | 191.07 | 5.929 |
| Weekly count, station and calendar | 123.58 | 3.936 | 155.49 | 5.657 |
| Add native fields and a broad wording category | 126.46 | 3.941 | 153.38 | 5.697 |
| Add which reviewed stations the notice names | 126.45 | 3.941 | 145.25 | 5.457 |
| Add their reviewed station roles | 126.45 | 3.941 | 144.42 | 5.532 |
| Exchange the two roles at prediction time | — | — | 140.92 | 5.702 |

The seasonal baseline had the lowest network subway error among these models. The history-and-calendar model had the lowest network bike error. The apparent gain of 0.83 entries from adding the roles on named stations reverses to a loss of 1.77 entries when the single February 16 incident is omitted. Four incidents occurred on February 24. Across leave-one-date-out summaries the rail improvement ranged from −1.77 to +5.39 entries, and the bike improvement ranged from −0.125 to −0.025 departures. These are sensitivity ranges. Four contributing dates provide too little evidence for a reliable uncertainty interval on the role comparison.

## Forecast design and coverage

- The first fit used targets whose hours ended by midnight at the start of February 15, New York time. It scored forecasts issued from that cutoff with completed target hours by midnight February 22. The next fold advanced both boundaries by seven days and expanded the training history. Every arm needed a valid subway and bike target and both same-hour counts from seven days earlier. The weekly lags assume an operator can access completed historical counts. February 1–7 supply history for the first eligible fit rows.
- An entire training recurrence group was excluded if it appeared in the later period's full notice inventory. Groups used the existing 24-hour rule for matching initial headers and route sets. The two fits used 13,000 and 24,127 station-hours. The resulting evaluation sizes were 11,086 and 8,696. Fifteen training rows were removed by the second fold's recurrence purge. Starting the evaluation at notice issue time accounts for the smaller denominators relative to the target-date [seasonal audit](../february/README.md).
- The numerical model was a separate ridge fit for each mode. It predicted the difference between log(1 + target) and log(1 + weekly count), using a fixed penalty of 10 and a train-fitted feature dictionary. Predictions were transformed back to counts and clipped at zero. The history arm included the log weekly count, station complex, hour, weekday and hour of week. The next arm added the first notice's route set, native status and previously released coarse cause-wording category. This is a lightweight text control. Full-text TF-IDF, gradient boosting, delayed public features and Qwen remain to be evaluated.
- Both annotation arms received the same reviewed mentioned-station indicator and its complex ID. The full arm also received the local station's role, for example boarding platform, short turn or express boundary. Each training fold included any matched annotated station, including isolated names for which the other station lost its row. The respective fits had only 12 and 14 named training rows. Scoring the role comparison required the same originating notice, both named stations, both observed modes and both weekly lags. The first evaluation week contributed one complete pair and the second contributed six. The exchange control used the fitted role model and changed only which of the two labels accompanied each station.
- An earlier internal reading provided one positive pair per qualifying reviewed notice. Most notices were unreviewed. A missing annotation means the data contain no reviewed role for that station. The experiment measures the value of those sparse annotations, under a fixed small model, on previously explored February development data. A deployable parser, independent annotation check, unscheduled-event eligibility and May validation on a larger January–April training panel are still needed. The current result alone cannot justify a 72B comparison or a journal claim about language-model accuracy.

## Reproduce and verify

Run from the repository root with Python, NumPy, pandas and scikit-learn. The measured run used NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0. The script verifies the Git blob IDs for the three [input files](../february) and checks for duplicated scoring units, invalid origin times and crossing recurrence groups.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/alert_study/february_ablation.py --output /tmp/nyc-february-ablation --keep-predictions
```

A separate SQLite calculation reconstructed the evaluation keys from the event inventory, the station-hour table and the two annotated complex IDs. It reproduced both fold sizes, both complete-pair sets and all 66 MAEs from the row predictions to within 1e-8. The command above also emits the full prediction table for inspection. [Fold counts](folds.csv) · [All 66 scores](scores.csv) · [Named-station predictions and exchanged roles](pair_predictions.csv) · [Forecast code](../february_ablation.py)
