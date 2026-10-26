# SIGSOFT empirical-review checklist

Status reflects the repository snapshot revised on 2026-09-20. `N/A` denotes
human-subject criteria that do not apply to this computational benchmark study.

| Item | Status | Evidence / justification |
|---|---|---|
| Research questions stated and used to organize results | Yes | Five RQs in `article/sections/1_introduction.tex` and RQ1--RQ5 result sections |
| Study characterized as exploratory; no retrospective confirmatory hypotheses | Yes | `Methodology / Statistical Analysis` |
| Experimental objects and full-sample rationale | Yes | `Methodology / Programming Tasks and Code Generation`; all 563 HumanEval+/MBPP+ tasks |
| Independent unit and pairing declared | Yes | `Methodology / Statistical Analysis`; task-level grouping and pairing |
| Factors, controls, and uncontrolled factors declared | Yes | `Methodology / Study Overview` and `Threats to Validity` |
| Outcomes and multilabel metrics operationalized | Yes | `Methodology / Ground Truth` and `Evaluation Metrics` |
| Statistical procedure justified by outcome/design | Yes | `artifacts/statistical_audit.md` decision matrix |
| Two-sided tests and operational nulls identified | Yes | `Methodology / Statistical Analysis` and statistical audit |
| Multiple comparisons handled | Yes | Holm families: 78 RQ1 pairs, eight RQ3 primary comparisons, available RQ4 families |
| Effects, 95% CIs, denominators, and adjusted p-values reported | Partial | Complete for reproducible comparisons; local SLM Phase-6/7 inference is blocked because the SSD raw files are unavailable, and those manuscript rows are explicitly descriptive |
| Dependence within programming tasks handled | Yes | Task-cluster bootstrap and whole-task paired permutation |
| Assumptions and benchmark-conditioned scope declared | Yes | `Methodology / Statistical Analysis`; `Threats / Conclusion Validity` |
| Class imbalance and rare subgroups addressed | Yes | Macro/per-class metrics, Wilson/task-cluster CIs, counts, and Conclusion Validity |
| Seeds and resampling counts reported | Yes | Split/Optuna seed 42; analysis seed 20260920; 10,000 resamples |
| Decoding and reasoning controls reported | Yes | `Methodology / Evaluated Models`, `Programming Tasks`, and environment table |
| Model IDs, endpoint period, tools, and software versions reported | Yes | Tables `Models evaluated` and `Recorded execution and replication environment` |
| Original CPU/RAM/GPU metadata retained | No | The artifacts do not contain the original host identifiers; Table 4 reports `NR` rather than inventing them |
| Raw-data provenance and immutable hashes provided | Yes | `artifacts/manifest.md` and `artifacts/statistics/analysis_metadata.json` |
| End-to-end analysis command provided | Yes | README, `Reproduce Statistical Analyses` |
| Figures disclose normalization, denominator, and label reduction | Yes | Confusion-matrix caption and Construct Validity |
| Participants, recruitment, incentives, assignment, attention checks | N/A | No human participants; computational study using public benchmarks and generated code |
| Personal/sensitive data and privacy concerns | N/A | No personal data; Ethical Considerations |
| Risks and responsible use discussed | Yes | `Threats / Ethical Considerations and Responsible Use` |
| Causal claims limited by design | Yes | RQ5 and External/Conclusion Validity explicitly identify scale confounds |
| Manuscript compiles and citations/cross-references resolve | Yes | `latexmk -xelatex ...`; reference checker reports zero errors/warnings |

## Residual review notes

- The JSERD template emits a frontmatter width warning, but rendered page 1 has
  no visible clipping or overlap.
- Exact replication of original runtime/performance is limited by missing
  original-host hardware metadata and possible hosted-endpoint revision.
- The local SLM Phase-6/7 point estimates should receive paired CIs only after
  the two external-SSD JSONL files are restored and the same pipeline is rerun.
