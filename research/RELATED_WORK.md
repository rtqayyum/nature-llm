# Related-work and novelty audit

Evidence cut-off: 1 November 2026. This is a targeted primary-source review,
not an exhaustive systematic review or a proof of firstness. It updates the
earlier ECTT brief and detailed protocol. Published statements below describe
the authors' methods, not independently reproduced performance.

| Work | Established mechanism | Consequence for our claim | Evidence depth |
|---|---|---|---|
| LLM-UTP, Nature Communications 2026 | Multi-domain traffic training, target fine-tuning, trend/spatiotemporal encoding and language-model backbone | A larger language model, shared traffic training or another backbone alone cannot be our contribution | Prior paper/source reproduction plus current assessment |
| TESTAM, ICLR 2024 | Temporal/static/dynamic experts with pseudo-label-based gating | Ordinary expert routing already exists | Official proceedings abstract and author repository |
| OpenCity | Heterogeneous traffic pretraining and zero-shot spatial transfer | Cross-city or zero-shot traffic transfer alone is occupied | Author arXiv abstract |
| CompactST, PVLDB 2025 | Compact multi-domain pretraining, multi-scale mixing and dataset-oriented tuning | Large backbone scale must earn its cost against compact alternatives | Official PVLDB paper abstract/intro and artifact link |
| FlexST, arXiv 2609.13878 | Multi-resolution diffusion, domain-adaptive experts and periodic encoding | Multi-resolution handling, selective sharing and negative-interference routing alone are occupied | Author manuscript abstract and relevant full-text method passages; arXiv record reports ICDM 2026 acceptance, proceedings not independently checked |
| ST-FGDIR, Sensors 2026 | Learned signed forecast-gain routing between stable and corrected candidates; robustness losses | A learned estimate of correction benefit is not new | Publisher abstract/method excerpts; full page retrieval was rate-limited |
| MURECAST, Symmetry 2026 | Frozen base, out-of-sample residual memory, utility ranking and chronological calibrated activation | Historical residual reuse plus benefit calibration is particularly close prior art | Publisher abstract/method excerpts; full page retrieval was rate-limited |
| Choe and Ramdas, Comparing Sequential Forecasters | Anytime-valid, variance-adaptive inference for conditional forecast-score differences | Sequential comparison and repeated-look validity are established statistics | Journal and author arXiv abstracts; author implementation located |
| Howard et al., confidence sequences, Annals of Statistics 2021 | Time-uniform confidence sequences and mixture boundaries | Credit the standard Gaussian-mixture construction; our instantiation is not a new concentration theorem | Author implementation/documentation located |
| Arnold et al., Sequential model confidence sets, JRSS-B 2026 | Multiple forecasters compared with time-uniform confidence sets | Multiple-candidate sequential selection alone is occupied | Journal primary abstract/intro excerpts |
| Li and Zhu, CC-SMCS, arXiv 2609.28522 | Constrained sequential forecasting-pipeline selection; delayed multi-horizon extension | Delayed labels, temporal dependence and feasibility constraints alone cannot carry novelty | Author full-text selected formulation and delayed-feedback appendix passages; preprint |
| Xu et al., Active anytime-valid RCPS, NeurIPS 2024 | Sequential risk control, label budgets, risk predictors and label-efficient verification | Label-efficient verification alone is occupied; our mechanism must alter forecast learning, not only label querying | Author abstract and full-text formulation/contribution passages |
| Basu, Transfer-Informed Betting, arXiv 2603.08907 | Source-informed betting for cross-domain selective prediction | Warm-starting a verifier with source evidence is not a sufficient new claim | Author full-text selected formulation/theorem passages; reported guarantees are not adopted without checking their assumptions; preprint |
| Liu et al., Meta Two-Sample Testing, NeurIPS 2021 | Auxiliary-task learning to obtain more powerful tests with limited target data | Learning for test power is a general existing idea; application alone does not prove novelty | Official proceedings abstract and paper intro |
| Zhou et al., conformal instance transfer, PMLR 2019 | Model-independent source-instance/feature selection to avoid negative transfer | Combining conformal ideas and negative-transfer avoidance is not a first | Official proceedings abstract |
| Tan et al., NeurIPS 2024 | Ablations question whether language components explain forecasting gains in studied models | Require compact, random-initialization and language-component controls; do not generalize their findings to every LLM | Official proceedings abstract |
| ShiftGuard-ITS | Repository describes risk-controlled forecasting under shifts/outages | Stress testing and risk-control integration alone are insufficient | Repository description only; no paper theorem or numerical result independently validated |

## Specific candidate gap

The selected traffic methods learn accurate forecasts, utility gates or calibrated
corrections. The selected statistical methods provide comparison and selection
machinery. Our candidate question joins learning and verification:

**Can source-episodic training make a transferred forecast's benefit over a
strong, frozen local comparator easier to verify from a short target prefix,
while improving complete-system forecasting after the verification budget?**

The distinction is the forecasting objective: train the numerical representation
and correction for a useful verifiable gain, then assess with the same verifier
used by ordinarily trained candidates. We do not claim to invent confidence
sequences, active risk control, fallback routing or meta-learning for testing.

This gap is an inference from the reviewed formulations. None of the selected
passages establishes the exact proposed traffic training objective and matched
evaluation. Absence of a located match is not proof that no such work exists.
The closest general testing and certification-aware learning literature remains
a mandatory related-work discussion, not something to omit in a novelty claim.

## Rejected central claims

- First Qwen traffic model, first cross-domain traffic model, or first MoE.
- First utility/benefit gate, first calibrated correction or first safe transfer.
- First anytime-valid comparison, delayed-feedback verification or label budget.
- Leakage repair as methodological novelty.
- A wrapper applying existing statistics to Qwen with no learning contribution.

## Primary links

1. LLM-UTP: https://doi.org/10.1038/s41467-026-73610-2
2. TESTAM: https://proceedings.iclr.cc/paper_files/paper/2024/hash/0cb310ed8121549488fea8e8c2056096-Abstract-Conference.html
3. OpenCity: https://arxiv.org/abs/2408.10269
4. CompactST: https://www.vldb.org/pvldb/vol18/p2149-han.pdf
5. FlexST: https://arxiv.org/abs/2609.13878
6. ST-FGDIR: https://doi.org/10.3390/s26196319
7. MURECAST: https://doi.org/10.3390/sym18091485
8. Comparing Sequential Forecasters: https://doi.org/10.1287/opre.2021.0792 and https://arxiv.org/abs/2110.00115
9. Sequential model confidence sets: https://doi.org/10.1093/jrsssb/qkag066
10. CC-SMCS: https://arxiv.org/abs/2609.28522
11. Active anytime-valid RCPS: https://arxiv.org/abs/2406.10490
12. Transfer-Informed Betting: https://arxiv.org/abs/2603.08907
13. Meta Two-Sample Testing: https://proceedings.nips.cc/paper_files/paper/2021/hash/2e6d9c6052e99fcdfa61d9b9da273ca2-Abstract.html
14. Conformal instance transfer: https://proceedings.mlr.press/v105/zhou19a.html
15. Language-model ablation study: https://proceedings.nips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html
16. ShiftGuard repository: https://github.com/shishirbit/ShiftGuard-ITS
17. Official Qwen base/model configuration: https://huggingface.co/Qwen/Qwen2.5-1.5B
18. Confidence-sequence foundation and author implementation: https://doi.org/10.1214/20-AOS1991 and https://github.com/gostevehoward/confseq
