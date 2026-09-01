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

The project has four practical phases:

1. **Dataset building:** generate code for HumanEval/MBPP tasks and run
   EvalPlus.
2. **Dataset judging:** use judge models to provide explanations for the
   labels.
3. **Fine-tuning:** supervised QLoRA/LoRA training with resumable Optuna.
4. **Evaluation:** ask each judge model to classify held-out generated code.

The CLI entry point is [main.py](main.py). Run commands with `uv run`.

```bash
uv run main.py --config <config-file> --build_dataset
uv run main.py --config <config-file> --dataset_judge
uv run main.py --config <config-file> --train_model
uv run main.py --config <config-file> --evaluate
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
