# Statistical audit and RQ decision matrix

The study is research-question driven and exploratory. No retrospective H0/H1 statements are introduced. Every inferential procedure below is two-sided and conditions its interpretation on the observed EvalPlus benchmark.

| RQ | Estimand | Type | Independent unit | Pairing | Analysis and decision |
|---|---|---|---|---|---|
| RQ1 | Per-model probability of a task producing a Correct solution under the recorded single generation; paired differences between models | comparative | EvalPlus task | yes, the same 563 tasks for all 13 models | Wilson CI per model; Cochran's Q global test; exact two-sided McNemar for all 78 pairs; Holm over the 78-pair family; paired task bootstrap (10,000) for differences in percentage points. Valid because the binary outcome and task alignment were verified. |
| RQ2 | Exact-set Accuracy, Macro Precision/F1/Recall and class recall; self-error detection proportion | descriptive | programming task; repeated rationale rows clustered within task | not required for the main characterization | Task-cluster bootstrap (10,000) for metric CIs; Wilson CIs for self-detection proportions. No pairwise p-value is added because RQ2 characterizes heterogeneous detectors and did not predefine a focal model contrast. |
| RQ3 | Difference between the Gemma 4 E4B base and each specialized detector on the same held-out samples | comparative | programming task | yes, matched sample indices clustered in 113 tasks | Paired task-cluster bootstrap for differences; two-sided task-level label-swap permutation. Holm is applied to the predefined family of eight primary comparisons (Accuracy and Macro-F1 for four specialized models). Remaining class metrics are descriptive with CIs. |
| RQ4 | (a) tool Runtime/Syntax recall; (b) change in detector metrics with static feedback; (c) change in generation correctness and cycle count | mixed descriptive/comparative | generated program clustered by task for detection; task for generation | yes for the same code/model without-vs-with tools and Phase-1-vs-Phase-6 generation | Static-tool recall is descriptive with proportions. Phase 6 uses exact McNemar and paired bootstrap for Correct plus median/IQR for cycles. Phase 7 uses task-cluster bootstrap and paired task-level permutation. On this machine, inferential regeneration is complete for the two API models and blocked for the eleven SLMs because their raw SSD files are unavailable. |
| RQ5 | Whether the two substantially larger models show a consistent descriptive advantage across the preceding estimands | comparative synthesis, not a causal scale experiment | inherits RQ1–RQ4 units | available within each inherited comparison | No new omnibus scale p-value: only two non-randomly selected large models are available and size is confounded with family, training, backend and precision. RQ5 synthesizes effects/CIs from RQ1–RQ4 and limits claims accordingly. |

## Audit of previously reported tests

### RQ1 Cochran's Q

- Operational null: all 13 models have equal marginal probability of Correct on the 563 paired tasks.
- Outcome: binary Correct versus any error label.
- Pairing: confirmed; every model has one unique row for the same 563 benchmark/task keys.
- Independent unit: task. Independence across benchmark tasks is an approximation; inference is conditional on EvalPlus and does not imply random sampling from all programs.
- Result regenerated: Q = 230.970, df = 12, p = 1.253e-42.
- Decision: retain and describe as an exploratory omnibus comparison.

### RQ1 post-hoc McNemar

- Operational null for each pair: the two discordant directions are equally probable.
- Test: exact two-sided McNemar, not an asymptotic or one-sided variant.
- Family: all 78 unordered pairs among 13 models, fixed before inspecting the regenerated p-values.
- Correction: Holm across all 78 raw p-values.
- Effect: paired difference in Correct percentage points with a 95% percentile CI from 10,000 paired task bootstraps.
- The published Qwen 2.5 Coder 7B versus 32B result is reproduced: raw p = 0.002618, Holm p = 0.090556; paired difference 6.22 pp in favor of 7B, 95% CI [2.31, 10.12] pp.
- Decision: retain, correct the reporting to include the effect/CI, and avoid interpreting a non-significant adjusted comparison as equivalence.

## Multiplicity families

- RQ1: 78 model-pair McNemar tests, one Holm family.
- RQ3: eight predefined primary comparisons (four specialized models × Accuracy/Macro-F1), one Holm family. Other reported metrics are descriptive.
- RQ4 Phase 6: all available model-level Phase-1/Phase-6 comparisons in one Holm family; currently two API models.
- RQ4 Phase 7: all available model/metric comparisons in one conservative Holm family; currently 16 comparisons for two API models and eight metrics.

## Assumptions and limitations

- Bootstrap resamples whole tasks, preserving all rows and labels within a task.
- The paired permutation swaps complete task clusters between conditions. This requires exchangeability of condition labels under the null and is used only for matched conditions with the same inputs.
- A single generation per task/model does not estimate within-model sampling variance. The inferential unit is task, not generation seed.
- EvalPlus tasks are treated as the finite benchmark population. CIs quantify task-to-task uncertainty within this benchmark and should not be interpreted as population-sampling guarantees for arbitrary software.
- Missing raw local Phase-6/Phase-7 artifacts are explicitly blocked; point estimates already in the manuscript are not treated as independently verified inferential evidence until those files are restored.
