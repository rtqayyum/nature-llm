# Which station instructions can a simple parser recover?

Across 7,273 NYC subway first notices from January through April 2026, explicit wording rules assigned at least one operational role to a station complex in 877 notices. On the existing February chronological panel, the role labels increased pooled forecast error for both subway entries and bike departures at stations the rules identified. These are development results. They motivate a broader extraction audit before a more expensive forecasting experiment.

## First-notice coverage

| NYC month | First notices | At least one resolved complex with a role | Two complexes with disjoint role sets |
|---|---:|---:|---:|
| January | 1,927 | 214 | 9 |
| February | 1,752 | 252 | 7 |
| March | 1,821 | 216 | 11 |
| April | 1,773 | 195 | 7 |

The screen uses update zero only, with the archive clock interpreted as UTC and month boundaries in America/New_York. Role patterns cover explicit boarding, transfers, skipped stops, short turns and segment boundaries. The parser scans the header and description, stopping at an embedded planned-work reminder. It keeps character offsets in the original field. A station directory supplies full names with slash/hyphen variants. Among 1,724 extracted mentions, 1,082 resolved to a unique complex citywide, 614 used the affected routes and directory's daytime lines to select among identical names, and 28 remained unresolved. The retrospective directory is a development aid. Its daytime lines are an imperfect substitute for a dated timetable, especially during reroutes. Names outside a recognized instruction receive no feature in either station-specific arm. Archive counts include maintenance, repeated wording and notices with multiple operational changes.

The strict screen found 34 notices on 27 dates with two distinct complexes whose recognized role sets had no label in common. Allowing overlapping but unequal role sets raised the count to 55. A station can be both a suspension boundary and a short turn. Among 31 previously reviewed February positives the extractor recovered 39/62 assigned station-role labels and both labels in 13/31 notices. These inspected examples informed development, so these fractions describe coverage on known examples. The prior annotations each contain one selected pair and cannot measure false positives across all extracted stations.

A separate internal reading used six March and six April notices sampled by deterministic hash from 40 and 28 candidates respectively. Each candidate had at least two earlier wording cues and failed the strict disjoint-role screen. Ten of the twelve contained two explicitly distinguishable station instructions, one described a boarding interval and a conditional terminus, and one assigned a shared boarding instruction to three stations. In four of the ten, repeated wording described local service between Hoyt-Schermerhorn Sts and Broadway Junction and lower-level boarding at Nostrand Av. This targeted sample identifies missed constructions. It cannot estimate population prevalence. The annotations were made without those months' demand outcomes. [Sample IDs](miss_sample.csv) · [Internal readings](miss_review.csv)

## February chronological forecasts

Mean absolute errors on the same February folds and common two-mode mask as the [earlier pilot](../february_pilot/README.md). Every learned arm uses the same numerical ridge head per mode. The text arm adds TF-IDF unigrams and bigrams from the complete initial notice, with document frequencies fitted on distinct training incidents (minimum two documents). The names arm adds a local indicator and complex identity for stations the rules found. The roles arm adds one indicator per corresponding role. The numeric features include the same local weekly lag under the operator-access assumption. Each head is refitted with the same ridge penalty of 10. No hyperparameters were selected using these scores.

| Inputs | Network subway | Network bike | Rule-named subway | Rule-named bike |
|---|---:|---:|---:|---:|
| Weekly seasonal | 113.73 | 4.219 | 161.52 | 5.835 |
| Calendar, weekly count, native alert fields and broad wording cue | 126.46 | 3.941 | 181.11 | 4.960 |
| Add full first-notice text | 131.07 | 3.924 | 202.02 | 5.013 |
| Add which station the rule names | 131.03 | 3.924 | 187.44 | 4.872 |
| Add the rule's station roles | 131.06 | 3.925 | 192.97 | 4.947 |

The network columns cover 19,782 station-hours from 675 incidents on 13 dates. The rule-named columns cover 103 station-hours from 63 incidents on 12 dates. Adding roles increased named-station subway MAE by 15.76 in the first fold and reduced it by 1.80 in the second. Bike MAE increased by about 0.075 departures in each fold. Only one scored incident retained stations with disjoint labels, and only three retained any unequal sets. Most of the 103 observations cannot support a within-notice role exchange. The larger named-station sample measures average error at tagged locations. It offers little evidence about which station in a single incident should receive which role.

**Decision.** Keep the weekly rail forecast and the full-text and names-only controls. Before GPU forecast training, join the remaining development months using period-appropriate timetables and preperiod dock maps, and estimate how many complete, differently instructed stations survive in each incident. Evaluate overlapping roles, enumerations, alternatives and alias resolution against outcome-blind annotations on separate notices. Use May for validation after freezing the comparison. No accuracy result for Qwen or evidence for a publication decision follows from this test.

## Reproduce

From the repository root with Python, NumPy, pandas, SciPy and scikit-learn (1.8.0 used here), run:

```bash
python research/alert_study/extract_alert_roles.py /tmp/nyc-source /tmp/nyc-roles
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/alert_study/february_notice_text.py /tmp/nyc-source /tmp/nyc-forecasts
```

The first command verifies retrieved source bytes against the [manifest](sources.json), rebuilds the [event inventory](events.csv), [source spans](mentions.csv), sample IDs and [February comparison](february_comparison.csv). The second uses the released role table and verified February panel to produce row predictions and all [60 MAEs](february_scores.csv). A separate check matched every released span to the original field, recomputed all MAEs from the saved predictions, and reproduced the earlier network weekly and native-cue baselines within 1e-8.
