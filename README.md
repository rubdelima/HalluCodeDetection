HalluCodeDetection
=================

HalluCodeDetection builds and evaluates datasets for hallucination detection in code-generation models.
The workflow is organized into four phases that can be run independently from `main.py`.

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
