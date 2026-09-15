HalluCodeDetection
=================

HalluCodeDetection builds and evaluates datasets for hallucination detection in code-generation models.
The workflow is organized into independent phases that can be run from `main.py`.

Requirements
------------

- Ollama must be installed and running (current models are executed through Ollama).
- Python dependencies are managed with `uv`.

Install Dependencies
--------------------

```bash
uv sync
```

Project Phases
--------------

Phase 1 - Base Dataset Build
- Loads a percentage of the training split (configured in `config.yaml`).
- Generates model outputs for each benchmark item.
- Runs the provided tests and classifies results as:
	- `correct`
	- `functional_error`
	- `runtime_error`
	- `syntax_error`
- Writes results to `dataset_base.json` in the configured `results_dir`.
- Saves checkpoints based on `checkpoint_interval`.

Run Phase 1:

```bash
uv run main.py --build_dataset
```

Isolated OpenCode Go baseline (DeepSeek V4.1 Flash and MiMo-V2.5, generation
only, without a tool cycle):

```bash
uv run main.py --config config_phase1_opencode_baselines.yaml --build_dataset
```

This command writes to
`data/results/evalplus/phase1_opencode_baselines/dataset_base.json`. The
dedicated configuration disables judge models and uses empty Phase-4, Phase-6,
and Phase-7 model lists. Consequently, these baseline rows are not added to
the dataset consumed by Phases 2, 3, or 4. Rerunning the command resumes by
model and task without regenerating completed rows.

Phase 2 - Judge Augmentation
- A judge model reviews each result from Phase 1.
- For every sample, it receives the prompt, model output, level, and error.
- It produces a JSON explanation of why the output is correct or incorrect.
- Results are stored in `dataset_judge.jsonl` with:
	- `benchmark`
	- `benchmark_id`
	- `response_model`
	- `judge_model`
	- `explanation`

Run Phase 2:

```bash
uv run main.py --dataset_judge
```

Phase 3 - Model Training
- Trains a LoRA adapter to improve hallucination detection using the judged dataset.
- Uses Optuna's TPE sampler and maximizes validation accuracy only; the test split is not used to choose hyperparameters.
- Trial state, metrics, and failures are persisted in `results_dir/optuna_trials.db`; rerunning the command resumes the same study.
- `training.max_trials` is a total budget (completed plus failed trials), including trials from prior runs.

Run Phase 3:

```bash
uv run main.py --train_model
```

To pre-download all configured training checkpoints and reset a study
containing only failed/interrupted trials:

```bash
./scripts/prepare_phase3.sh --reset-search
```

Append `--train` to begin Phase 3 in the same environment.

Phase 4 - Evaluation
- Compares base models vs fine-tuned models.
- Reports how well the adaptation improved detection.
- Loads models by type, closing the current handler before switching to the next model.
- Streams each model answer in the spinner and refreshes a live summary table after every model.

Run Phase 4:

```bash
uv run main.py --evaluate
```

Phase 5 - Static Analysis
- Compares Python compilation, Ruff, Pyright, Pylint, Semgrep, and CrossHair.
- Reports Runtime Recall (`RR`) and Syntax Recall (`RS`) both on the held-out
  test tasks (`T`) and all unique Phase-1 generations (`G`).
- `Avg T` is the average wall-clock time in seconds per checked generation.
- The live panel is refreshed after every resumable batch. CrossHair is marked
  experimental because its symbolic analysis is most effective on code with
  assertions, type annotations, or contracts.

Run the complete Phase 5 using the same split as the balanced Phase 4:

```bash
uv run main.py --config config_phase3_balanced.yaml --phase5
```

Run one or more tools, retry failed batches, or make a small smoke test:

```bash
uv run main.py --config config_phase3_balanced.yaml --phase5 --phase5_tool ruff --phase5_tool pyright
uv run main.py --config config_phase3_balanced.yaml --phase5 --retry
uv run main.py --config config_phase3_balanced.yaml --phase5 --phase5_limit 100
```

Phase-5 outputs are written under `results_dir/phase5/`:

- `static_analysis_summary.csv`: the requested comparison table;
- `static_analysis_results.jsonl`: per-sample predictions, diagnostics, tool
  versions, timing, and status used for audit and automatic resumption.

Phase 6 - Tool-assisted Code Generation
- Produces fresh generations; it does not reuse Phase-1 code.
- Each candidate receives feedback only from Python `compile` and Pyright.
- The same model self-reviews the candidate and can submit it or revise it for
  up to five rounds.
- EvalPlus is run only once on the final candidate. Test results are never
  returned to the model, so there is no test-driven retry.
- Results are isolated in `phase6/generation_with_tools.jsonl`.

Run Phase 6:

```bash
uv run main.py --config config_phase3_balanced.yaml --phase6
```

Phase 7 - Tool-assisted Error Classification
- Reads original Phase-1 code and labels directly from `dataset_base.json`.
- Uses the same held-out task IDs as Phase 4, but deduplicates the Phase-2
  judge explanations, yielding 1,243 unique code samples from 113 tasks.
- Gives `compile` and Pyright diagnostics to the classification model in a
  single judgment call; there is no repair cycle.
- Never reads Phase-6 results and writes only to
  `phase7/classification_with_tools.jsonl`.

Run Phase 7:

```bash
uv run main.py --config config_phase3_balanced.yaml --phase7
```

Short, resumable checks with OpenCode Go:

```bash
uv run main.py --config config_phase3_balanced.yaml --phase6 \
  --model_name deepseek-v4.1-flash --phase6_limit 1 --phase6_rounds 2
uv run main.py --config config_phase3_balanced.yaml --phase7 \
  --model_name deepseek-v4.1-flash --phase7_only_errors --phase7_limit 1
```

Commands with a `--phase6_limit` or `--phase7_limit` write to a separate
`*_smoke.jsonl` file, so short validation runs cannot contaminate full results.
Completed rows are reused on later runs. `--retry` retries only calls that
ended in an infrastructure/API error; it does not expose EvalPlus failures to
the model or initiate a test-guided repair.

Configuration
-------------

Edit `config.yaml` to control:
- dataset percentage (`dataset_load`)
- models list (`models`)
- judge model (`judge_model`)
- test timeout (`tests_timeout`)
- checkpoint interval (`checkpoint_interval`)
- results directory (`results_dir`)
- Optuna trial budget (`training.max_trials`) and startup trials (`training.optuna_startup_trials`)

Interactive Menu
----------------

```bash
uv run main.py
```

Use a custom config file:

```bash
uv run main.py --build_dataset --config config.yaml
```

Outputs
-------

- Base dataset results: `results_dir/dataset_base.json`
- Judge dataset results: `results_dir/dataset_judge.jsonl`

Notes
-----

- `--view_dataset` launches a Textual TUI to explore results.
- The dataset build phase runs models in model-first order to minimize model reloads.

Run the TUI:

```bash
uv run main.py --view_dataset
```
