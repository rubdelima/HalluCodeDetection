# HalluCodeDetection: Continuation Guide

This file is the working context for a new agent or a new machine. Read it
before modifying training, evaluation, dataset splits, or files on the SSD.

## Project overview

HalluCodeDetection evaluates whether code generated for EvalPlus tasks is
correct or contains one or more of four labels:

- `correct`
- `functional`
- `runtime`
- `syntax`

The CLI has seven practical phases:

1. **Dataset building:** generate code for HumanEval/MBPP tasks and run
   EvalPlus.
2. **Dataset judging:** use judge models to provide explanations for the
   labels.
3. **Fine-tuning:** supervised QLoRA/LoRA training with resumable Optuna.
4. **Evaluation:** ask each judge model to classify held-out generated code.
5. **Static analysis:** compare Python compilation and static checkers against
   execution-derived Syntax and Runtime labels.
6. **Tool-assisted iterative generation:** generate new code, inspect it with
   compilation and Pyright, and let the model submit or revise it for up to
   five cycles before one final EvalPlus evaluation.
7. **Tool-assisted error detection:** classify held-out original Phase-1 code
   with compilation and Pyright feedback; do not use Phase-6 generations.

These CLI phase numbers are not identical to manuscript RQ numbers. In the
article, RQ1 is code generation; RQ2 is error/self-error detection; RQ3 is
fine-tuning; RQ4 is static tools; RQ5 is static-feedback detection; RQ6 is
iterative generation; and RQ7 synthesizes SLM-versus-larger-model comparisons
across settings. Do not treat article RQ7 as synonymous with CLI `--phase7`.

The CLI entry point is [main.py](main.py). Run commands with `uv run`.

```bash
uv run main.py --config <config-file> --build_dataset
uv run main.py --config <config-file> --dataset_judge
uv run main.py --config <config-file> --train_model
uv run main.py --config <config-file> --evaluate
uv run main.py --config <config-file> --phase5
uv run main.py --config <config-file> --phase6
uv run main.py --config <config-file> --phase7
```

## Important behavior

- The phase-4 judge receives **the original programming-task description and
  the generated code**. It must return JSON labels and an explanation.
- The evaluated model does **not** receive the ground-truth labels, EvalPlus
  test output, phase-1 labels, or phase-2 explanations.
- Ground truth is used only afterward to calculate multilabel metrics.
- Evaluation errors, including CUDA OOM for a sample, are recorded and skipped
  instead of aborting the entire evaluation. Use `--retry` to retry saved
  error rows.

## Layout and key modules

| Area | Primary files |
|---|---|
| Configuration loading | `src/constants/__init__.py`, `src/constants/*.py` |
| Dataset build and judge | `src/dataset/build.py`, `src/dataset/augmentation.py` |
| Dataset split/sampling | `src/dataset/judge_dataset.py` |
| Model handlers/prompts | `src/models/` |
| QLoRA training | `src/training/runner.py`, `hyperparams.py`, `search.py`, `storage.py` |
| Phase-4 evaluation | `src/evaluations/run.py`, `metrics.py`, `ui.py` |
| Static tools / Phase 5 | `src/static_analysis/run.py`, `tools.py`, `feedback.py` |
| Iterative generation / Phase 6 | `src/phase6/run.py`, `src/schemas/phase6.py` |
| Tool-assisted detection / Phase 7 | `src/phase7/run.py`, `src/schemas/phase7.py` |
| Hosted backend | `src/models/opencode_go_handler.py` |
| Article and reproducibility checks | `article/`, `scripts/check_article_references.py`, `scripts/run_statistical_analysis.py` |
| Results notebook | `notebooks/phase_results.ipynb` |

`pyproject.toml` defines the Python dependencies. The normal local runtime is
the repository `.venv`, managed through `uv`.

## Dataset integrity and balancing

`load_hallucination_dataset()` builds records from:

- `data/results/evalplus/dataset_base.json`
- `data/results/evalplus/judge/dataset_judge.jsonl`

The split is deterministic and grouped by `task_id`. A whole task is assigned
to exactly one of train, validation, or test. The greedy split objective also
matches class frequencies as closely as possible. Do not replace it with a
row-level random split: it would leak different generated answers for the
same programming task across splits.

The balanced phase-3 option is intentionally applied **after** that split and
only to `train`:

- records containing `syntax` are excluded;
- records are bucketed by primary label `runtime`, `functional`, `correct`;
- data are downsampled without duplication to the requested ratio;
- validation and test remain unmodified and task-disjoint.

The current supported ratios are written in configuration as
`[runtime, functional, correct]`, for example `[1, 3, 3]`.

## Configuration files

### `config.yaml`

Base configuration: all Ollama/Hugging Face model definitions, dataset source
paths, and the prior standard/weighted experiments. Treat it as the shared
base; do not point it at an experiment-specific result directory unless that
is deliberate.

### `config_phase3_balanced.yaml` (current phase-4 configuration)

This is the main configuration to use for the current balanced experiment and
final phase 4. It uses `extends: config.yaml`; recursive YAML merge support is
implemented in `src/constants/__init__.py`.

Current training setup:

- base model: `google/gemma-4-E4B-it` (local HF path on the SSD);
- QLoRA; `label_weighted` loss with weight `3.0`;
- Optuna target: `macro_f1`;
- training-only syntax exclusion;
- ratios: `1:1:1` and `1:3:3`;
- `max_saved_models: 1` for the balanced Macro-F1 study;
- Optuna/results: `/mnt/ssd/HalluCode/results/phase3_balanced_no_syntax`;
- retained models: `/mnt/ssd/HalluCode/models/phase3_balanced_no_syntax`.

The evaluation list includes baselines and all selected fine-tuned models.
It is the correct configuration to finish phase 4:

```bash
uv run main.py --config config_phase3_balanced.yaml --evaluate
```

### `config_phase3_accuracy_retrain.yaml`

A one-off, isolated retraining configuration for the highest observed
validation-Accuracy candidate. It exists because Optuna will otherwise not
repeat an already completed finite-grid trial.

- one candidate, one trial;
- target: `accuracy`;
- ratio `1:3:3`;
- `r=16`, `alpha=32`, `dropout=0.1`, `lr=1e-4`, 4 epochs, `bias=all`;
- isolated artifacts under `/mnt/ssd/HalluCode/results/phase3_accuracy_retrain`
  and `/mnt/ssd/HalluCode/models/phase3_accuracy_retrain`.

## Fine-tuned model inventory

Do not delete these without explicit user approval. The external SSD is the
authoritative location for model artifacts.

| Stable ID | Purpose | Local path | Validation result |
|---|---|---|---|
| `trained/gemma-4-e4b-best` | Best original accuracy search | `/mnt/ssd/HalluCode/models/8a59793f` | Acc 66.65 |
| `trained/gemma-4-e4b-weighted-macro-f1-best` | Best prior weighted Macro-F1 search | `/mnt/ssd/HalluCode/models/macro_f1_weighted/1c41671b` | MF1 52.39 |
| `trained/gemma-4-e4b-balanced-macro-f1-best` | Best balanced Macro-F1 search | `/mnt/ssd/HalluCode/models/phase3_balanced_no_syntax/8e34f894` | Acc 67.21; MF1 43.81 |
| `trained/gemma-4-e4b-balanced-accuracy-retrained` | Repeated highest-Accuracy balanced candidate | `/mnt/ssd/HalluCode/models/phase3_accuracy_retrain/c0c9e44c` | Acc 69.70; MF1 37.65 |

The original balanced run observed an Accuracy of 71.32 for the same
hyperparameters as the last row, but it was not retained because its Macro F1
was lower. The isolated repeat was retained for phase-4 comparison.

The current `models` and `evaluation.models` sections in
`config_phase3_balanced.yaml` already define and select the two balanced
models. Do not use raw absolute model paths as public model IDs; add a stable
ID with `local_path` instead.

## Training implementation

- `src/training/search.py` uses persistent Optuna TPE backed by SQLite.
  Failures are saved as `FAIL` trials and interrupted `RUNNING` trials are
  marked failed on resumption.
- Search candidates include LoRA fields, loss strategy/weight, and class
  balance ratio. A change to any of these needs a distinct study/result
  directory if prior trials must not be reused.
- `src/training/hyperparams.py` provides `ClassificationWeightedSFTTrainer`.
  It is ordinary causal SFT cross-entropy with a higher token weight for class
  names; it does not directly optimize non-differentiable Macro F1.
- Macro F1, Macro Recall, and Accuracy are calculated on the validation split
  after training. `optuna_target` can be `accuracy`, `macro_f1`, or
  `macro_recall`.
- The trainer disables in-epoch evaluation/checkpoints to avoid GPU memory
  peaks. It uses QLoRA 4-bit, gradient checkpointing, and clears CUDA memory
  between model loads.

## Evaluation and UI conventions

Evaluation is multilabel: every class has independent TP/FP/FN counts.
Metrics are not derived from a single primary label. The notebook uses a
primary-label reduction only for its confusion matrices.

The Rich progress line, used by both validation in phase 3 and phase 4, uses:

- `Acc`: exact-set accuracy;
- `MF1`: Macro F1;
- `CR`, `FR`, `RR`, `SR`: recall for Correct, Functional, Runtime, Syntax.

All displayed metric values are percentages with one decimal and no `%` sign,
for example `Acc:54.3`. Local temporary models evaluated during phase 3 are
shown as `trained_model`, not as absolute paths. The training table has a
separate `ratio` column.

Phase-4 rows are persisted in:

```text
data/results/evalplus/evaluation_results.jsonl
```

`evaluation.overwrite: false` means completed rows are reused. The final
summary reloads persisted rows so resumed evaluation includes prior models.

## Notebook

`notebooks/phase_results.ipynb` is the report notebook. Its visible prose is
English. It contains:

1. phase-1 EvalPlus classification table;
2. train/validation/test class-distribution pie charts;
3. phase-3 training table;
4. phase-4 coverage/metric tables and confusion matrices.

The phase-4 table includes Macro Precision, Macro F1, Macro Recall, recalls by
class, and starred metrics that exclude `syntax`. Confusion matrices are
row-normalized and show both percent and `predictions / real-row total`.

After changing notebook source, execute it from the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/phase_results.ipynb --ExecutePreprocessor.timeout=180
```

## Experiment history and boundaries

- The original Phase-1 output contains the local model generations and
  EvalPlus-derived labels. RQ1 compares code generation across all 563
  HumanEval+/MBPP+ tasks. Correct is task-level pass@1; Functional and Runtime
  can overlap, while Correct and Syntax are exclusive outcomes.
- The hosted DeepSeek V4.1 Flash and MiMo V2.5 code-generation baselines use
  `config_phase1_opencode_baselines.yaml` and write only to
  `data/results/evalplus/phase1_opencode_baselines/dataset_base.json`. This
  isolates their Phase-1 comparison rows from the judged/training dataset for
  Phases 2--4. Never merge those rows into the main `dataset_base.json` just
  to make an analysis convenient. **Caution:** `config.yaml` currently also
  lists these hosted IDs under `dataset_building.models`; use the dedicated
  baseline configuration for hosted Phase-1 generation, and inspect the
  selected model list before any main dataset rebuild.
- Hosted inference uses `OPENCODE_GO_API_KEY` from `.env` or the environment.
  Never commit `.env`, log the key, or embed it in a manuscript/artifact.
  `config.yaml` currently sets `reasoning_effort: none` and 12 concurrent
  threads for each hosted model. Threaded execution is specific to the
  OpenCode Go backend. Transient provider failures are retried with backoff;
  failed rows may be rerun with `--retry`. A user interrupt should not make
  already persisted results disappear.
- The 65/15/20 split is by programming task, not generated row. The held-out
  test side contains 113 task IDs. Multiple models can generate distinct
  rows for the same task, so "113 tasks" and thousands of held-out code
  samples are both correct. Phase 7 deduplicates judge explanations and
  classifies 1,243 unique original Phase-1 code samples on those tasks.
- Do not use EvalPlus tests or execution labels in model prompts for Phase 4,
  Phase 6, or Phase 7. They are used only afterward for scoring. Phase 6
  must not perform test-guided retries or use an imagined public/private
  EvalPlus split as a repair oracle. Public `base_plus` tests do not make a
  genuinely hidden evaluation set.

### Static analysis and feedback experiments

Phase 5 compares `compile`, Ruff, Pyright, Pylint, Semgrep, and CrossHair. It
reports Runtime Recall and Syntax Recall for held-out test tasks (`T`) and
all unique generated code (`G`), plus average verification time. Its raw and
summary outputs are under `data/results/evalplus/phase5/`. Compilation and
Pyright were selected for subsequent feedback experiments: compilation gives
deterministic syntax evidence; Pyright provides incomplete, potentially
false-positive runtime evidence. Do not equate low recall with low precision:
they measure different failures. Preserve the original Phase-5 raw output
when recalculating tool mappings or summaries.

Phase 6 writes new, separate generations to
`data/results/evalplus/phase6/generation_with_tools.jsonl` (or configured
`results_dir`). Within each cycle, code is generated, checked with `compile`
then Pyright, and self-reviewed by the same model. The model decides whether
to submit or revise, up to five cycles. EvalPlus runs only once on the final
candidate. Save per-cycle records and the average cycles used. A task-level
progress counter advances only after all cycles and final evaluation for that
task finish; a zero counter can coexist with in-flight requests. Use
`--phase6_limit` and `--phase6_rounds` for short smoke tests; limited runs
write separate smoke files.

Phase 7 (CLI) writes tool-assisted classifications to
`data/results/evalplus/phase7/classification_with_tools.jsonl`. It uses the
held-out, original Phase-1 generations, not Phase-6 code. It supplies
`compile`/Pyright evidence for one classification judgment; there is no
iterative code repair. It can be restricted to hosted models on a laptop with
repeatable `--model_name` and later resumed for local fine-tuned models on
the SSD-capable machine. `--phase7_only_errors` is an analysis/smoke-test
filter, not a change to the full-test definition. Limited runs are isolated
from full results.

### Manuscript, figures, and statistical interpretation

- The manuscript is `article/main.tex`, using `article/sections/` and
  `article/refs.bib`. The `article/overleaf/` directory is a separately
  prepared copy. Compile from `article/` with `latexmk main.tex`; `latexmkrc`
  selects XeLaTeX and BibTeX. Check citations bidirectionally with
  `uv run scripts/check_article_references.py`. The checker includes
  `main.tex` and the sections, reports missing keys and uncited BibTeX
  entries, and never deletes references automatically.
- RQ1's five-part bar chart is generated by
  `article/src/generate_code_generation_plot.py` into
  `article/figures/code_generation_results.pdf`. The parts distinguish
  Correct, Functional-only, Functional+Runtime, Runtime-only, and Syntax;
  they are not four mutually exclusive independent labels. Article figures
  such as this chart and the RQ5 Venn diagram should fit one column unless
  a deliberate double-column float is requested.
- RQ2 metrics are multilabel. `Acc` is exact-set Accuracy, `MF1` averages
  per-class F1 over all four labels, and `MF1(NS)` averages the three
  non-Syntax F1 scores. A high Syntax recall does not imply high Syntax F1:
  false positives can depress precision. The self-detection table uses the
  generator's own erroneous test-set code, not all generated code.
- For the RQ3 paired bootstrap, compare each fine-tuned detector with the
  Gemma 4 E4B base on exactly the same valid samples, clustered by task.
  There are 3,729 persisted rows per detector but the same three CUDA-OOM
  indices are invalid for all five compared conditions; the paired analysis
  uses 3,726 valid rows. Standalone RQ2 values include the failed rows as
  empty predictions, explaining small differences from RQ3 after rounding.
- The RQ6 comparison is *iterative generation* versus original RQ1 code
  generation, with paired task outcomes and exact two-sided McNemar tests.
  Do not import RQ6-cycle claims into the RQ7 subsection about initial code
  generation. For RQ7's initial-generation comparison, pair each of 11 SLMs
  with each of the two hosted baselines on the same 563 tasks. Correct is the
  primary outcome; the other error classes are complementary. If adjusting
  22 pairwise tests within each class, Muse Glimmer, GPT-OSS, and Gemma 4 31B
  significantly exceed both hosted baselines in Correct; under the stricter
  all-88-tests Holm family, only Muse Glimmer retains that result. State the
  multiplicity family and never infer that model size *caused* a difference
  from these purposively selected models.
- `scripts/run_statistical_analysis.py` and `artifacts/statistics/` contain
  the reproducible statistical outputs. Regenerate and validate paired raw
  data before updating manuscript statistics; do not derive tests from
  rounded percentages in tables.

## SSD and portability

The original environment mounts an external SSD at `/mnt/ssd`. Expected data:

```text
/mnt/ssd/HalluCode/hf-models/       # downloaded Hugging Face Gemma models
/mnt/ssd/HalluCode/models/          # merged fine-tuned models
/mnt/ssd/HalluCode/checkpoints/     # temporary Trainer checkpoints
/mnt/ssd/HalluCode/results/         # isolated Optuna/JSONL experiment state
```

On a new device, mount or copy this hierarchy before running the phase-3 YAML,
or update all `local_path`, `model_output_dir`, `checkpoints_path`, and
`results_dir` values. Never move or delete Hugging Face cache/model files
unless the user explicitly requests it. Check free space before downloads;
Gemma weights are large.

## Working rules for future agents

- Preserve existing user changes in this dirty repository. Do not reset or
  revert unrelated files.
- Use `apply_patch` for repository file edits.
- Use `rg`/`rg --files` for fast searching.
- Treat deletion of model/checkpoint directories as destructive: first list
  exact paths and request/confirm explicit approval.
- Keep experiments isolated by result/model/checkpoint directories and Optuna
  study name. Never let a new retention policy evict models from a different
  experiment.
- If a derived YAML overrides a list (`models.gemma` or `evaluation.models`),
  it replaces that list; re-declare inherited items that must remain.
- Run at least `uv run python -m compileall -q src` and parse the relevant YAML
  after code/configuration changes. Do not start expensive training or phase-4
  evaluation unless requested.
