# How often was a May notice revised before its target hour?

Of the 1,593 May incidents with paired station-hours, 989 had an archived revision strictly after the original issue and strictly before the target hour began (62.1%). Together they account for 29,998 of 46,621 paired station-hours (64.3%). Another 24 incidents had a revision timestamped exactly at the start and none strictly between issue and start. The frozen forecasts use the original notice throughout. These revisions describe the information that arrived while the forecast was outstanding.

The screen compares each original with its latest strictly pre-target revision. An exact match on both native affected-route and status strings remained in 732 of the 989 revised incidents. In 43 of those 732 incidents the frozen rules changed at least one resolved complex–role assignment. In 28, a change touched at least one station in the paired forecast set, covering 40 station-hours. The comparison tests the two messages' recorded fields. Intermediate versions may differ. [Event-level timing and changed role sets](events.csv) · [Counts before and after pairing](summary.json) · [Source hashes](../panel/sources.json)

## Reading the 28 selected notice pairs

One internal, AI-assisted reading of the 28 original/latest texts classified the reported changes below. Selection used rule output, native-field agreement and station-hour availability. Demand magnitudes and forecast errors were excluded from the review. The counts describe the selected pairs and cannot estimate archive-wide parser accuracy. A text change also cannot establish which trains operated.

| Change in the notice | Incidents |
|---|---:|
| New operational instruction added | 14 |
| Endpoint of an existing segment moved | 4 |
| Scheduled service explicitly resumed | 4 |
| Prior instruction disappeared without an explicit restoration statement | 3 |
| Rules treated a statement about restored local-track service as a new role | 2 |
| An existing express segment was also applied to another line | 1 |

[Version IDs and individual readings](text_review.csv)

The four endpoint changes can be traced to consecutive messages, including two cases where the update of interest was not the first revision. Times are local to New York. In all four pairs the adjacent messages retained the same native route and status strings.

| Date | Consecutive alert IDs | Time of endpoint edit | Old endpoint → new endpoint | Both stations retained on the paired mask |
|---|---|---|---|---|
| May 14 | 511148 → 511150 | 11:42 | Broadway Junction → Hoyt-Schermerhorn Sts | Yes |
| May 16 | 512354 → 512357 | 12:54 | 62 St → 9 Av | No bike outcome or lag at 62 St |
| May 28 | 515549 → 515558 | 18:38 | Crown Heights-Utica Av → Pennsylvania Av | No bike outcome or lag at Pennsylvania Av |
| May 29 | 515715 → 515738 | 05:37 | 14 St → 34 St-Penn Station | Yes |

The May 14 and May 29 cases support a matched comparison at both endpoints. Two incidents are insufficient for a reliable estimate of forecast benefit. A separate update-time evaluation would need to compare the text immediately before and after an edit at the same issue time and on the same outcome rows. Selection and review on May make these examples developmental.

To recreate the screen, use the version snapshot and station directory from the panel manifest. The program uses only the keys from the frozen operator rail forecast file. The same keys apply to both modes.

```bash
root=research/alert_study/may_validation
python "$root/revision_timing.py" "$CACHE/versions-2026-05.json" "$CACHE/stations" "$root/revision_timing"
```
