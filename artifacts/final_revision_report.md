# Final SIGSOFT-oriented revision report

Revision date: 2026-09-20 (America/Recife)  
Source snapshot analyzed: `6fdcacd02ab08bd47c1fa2013e41c77db99d918f`

## Implemented changes

- Added one reproducible statistical entry point,
  `scripts/run_statistical_analysis.py`, with schema/pairing checks, immutable
  input hashes, fixed analysis seed, Wilson intervals, task-cluster bootstrap,
  Cochran Q, exact two-sided McNemar, paired task-level permutation, and Holm
  correction.
- Added SciPy to the versioned `uv` environment and generated CSV/LaTeX results,
  metadata, a validation report, and a pipeline log under
  `artifacts/statistics/`.
- Added `artifacts/manifest.md`, `artifacts/statistical_audit.md`, the prospective
  repeated-generation protocol, and this SIGSOFT checklist.
- Added the exploratory-analysis declaration, variables/controls, sample
  rationale, statistical assumptions, resampling count, and analysis seed to
  the methodology. Operational replication and host-environment details remain
  outside the manuscript for possible later inclusion.
- Corrected RQ1 Muse Glimmer from 456 to 455 successful tasks and added Wilson
  intervals, paired effects, bootstrap CIs, and multiplicity-adjusted results.
- Rebuilt the RQ2 self-detection table from the current deterministic task split,
  replacing stale denominators and adding Wilson intervals; added task-clustered
  uncertainty to the main RQ2 interpretation.
- Added the eight paired primary RQ3 comparisons with effect sizes, CIs, and
  Holm-adjusted permutation results.
- Added standalone tool denominators/CIs and paired API-model RQ4 effects for
  tool-assisted detection and generation. Local SLM tool-assisted rows are
  explicitly descriptive because their raw SSD files are unavailable.
- Reframed RQ5 as a confounded, benchmark-conditioned synthesis rather than a
  causal scale study; aligned the abstract, introduction, results, threats, and
  conclusion with the available evidence.
- Added Conclusion Validity and Ethical Considerations/Responsible Use and
  documented that the detector and EvalPlus outcomes are not proofs of program
  correctness.
- Added end-to-end reproduction commands to the README.

## Statistical decisions by RQ

- **RQ1:** comparative; 563 paired tasks, Cochran Q, 78 exact McNemar tests with
  Holm, Wilson model intervals, paired task-bootstrap effects.
- **RQ2:** descriptive; task-clustered metric CIs and Wilson self-detection CIs;
  no post-hoc detector-ranking test.
- **RQ3:** comparative; base versus each specialized detector on 113 paired task
  clusters, bootstrap effects and two-sided task-label permutation; Holm over
  Accuracy/Macro-F1 for four models.
- **RQ4:** mixed; standalone tool recall is descriptive; Phase 6 uses McNemar and
  paired bootstrap; Phase 7 uses task-clustered bootstrap/permutation. Formal
  inference is currently available only for the two hosted models.
- **RQ5:** comparative synthesis; no new omnibus scale test because size is
  confounded and only two purposively selected large models are present.

## Verification performed

```text
uv run scripts/run_statistical_analysis.py --config config_phase3_balanced.yaml
uv run ruff check scripts/run_statistical_analysis.py
uv run python -m compileall -q src scripts/run_statistical_analysis.py
uv run scripts/check_article_references.py
cd article && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo article/main.pdf
pdftoppm -png article/main.pdf <temporary-prefix>
```

The reference checker reports 10 TeX files, 47 citation uses, 34 BibTeX entries,
zero errors, and zero warnings. The final PDF has 23 A4 pages. Representative
and changed pages were rendered and visually checked; revised tables fit without
clipping or overlap. The remaining overfull warning originates in the template's
frontmatter close and has no visible page-1 defect.

## Verifiable blockers and residual limitations

1. `/mnt/ssd/HalluCode/results/phase6_local_tools/generation_with_tools.jsonl`
   and `/mnt/ssd/HalluCode/results/phase7_local_tools/classification_with_tools.jsonl`
   are not mounted. This blocks regenerated inference for local SLM RQ4 rows.
2. Exact CPU, RAM, and GPU identifiers for the original inference/training host
   were not retained. The manuscript reports `NR` rather than fabricating them.
3. Repeated Phase-1 generations were not authorized and would require 36,595
   new model calls under the documented five-repetition plan. Consequently,
   within-model sampling stability remains unknown.
4. Hosted endpoint weights/precision are provider-controlled and can change;
   conclusions apply to persisted responses collected September 13--14, 2026.
