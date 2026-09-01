from __future__ import annotations

import gc
import time
from pathlib import Path
from uuid import uuid4

import torch
from transformers import BitsAndBytesConfig

from src.constants import HalluCodeDetectionConfig
from src.constants.models import ModelInfo
from src.constants.training import TrainingHyperparameters
from src.core import ui as core_ui
from src.dataset.judge_dataset import balance_training_split, load_hallucination_dataset
from src.evaluations import evaluate_model
from src.schemas.training import TrainingResult
from src.training.hyperparams import get_trainer
from src.training.search import ask_next_trial, complete_trial, completed_trial_count, create_study, fail_trial
from src.training.state import (
    build_training_status_rows,
    display_hyperparameters_for_run,
    hyperparameters_key,
    load_training_results,
    write_training_results,
)
from src.training.storage import apply_saved_model_policy, remove_model_dir, save_merged_model
from src.training.ui import render_training_status_table
from src.models.loading import load_model, load_text_tokenizer


def train_model(
    run_path: Path,
    merged_model_path: Path,
    hyperparameters: TrainingHyperparameters,
    dataset,
) -> TrainingResult:
    dtype = _training_dtype()
    quantization_config = _quantization_config(hyperparameters, dtype)

    try:
        model, processor = _load_training_model(hyperparameters, dtype, quantization_config)
        trainer = get_trainer(hyperparameters, model, str(run_path), processor, dataset)

        start_time = time.time()
        trainer.train()
        trainer.save_model()
        training_time = time.time() - start_time

        del model
        del trainer
        _clear_memory()

        save_merged_model(
            base_model_id=hyperparameters.model_name.local_path or hyperparameters.model_name.id,
            adapter_path=run_path,
            merged_model_path=merged_model_path,
            dtype=dtype,
        )
        validation_result = _evaluate_validation_model(
            dataset,
            hyperparameters,
            merged_model_path,
        )

        return TrainingResult(
            **hyperparameters.model_dump(),
            run_path=str(run_path),
            model_path=str(merged_model_path),
            saved_model=True,
            training_time=training_time,
            train_acc=None,
            val_acc=validation_result.overall_accuracy,
            val_macro_f1=validation_result.macro_f1,
            val_macro_recall=validation_result.macro_recall,
            test_acc=None,
        )
    finally:
        _clear_training_locals(locals())
        _clear_memory()


def train_models(config: HalluCodeDetectionConfig) -> None:
    paths = _training_paths(config)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["training_runs_path"].mkdir(parents=True, exist_ok=True)

    previous_results = load_training_results(paths["trained_models_path"])
    all_hyperparameters = config.training_config.hyperparameters
    study = create_study(config.training_config, paths["optuna_storage_path"], previous_results)

    _render_start_status(config, paths, all_hyperparameters, previous_results, completed_trial_count(study))
    dataset = _load_training_dataset(config, paths["results_dir"])
    models_path = Path(config.training_config.models_path)
    models_path.mkdir(parents=True, exist_ok=True)

    while completed_trial_count(study) < config.training_config.max_trials:
        next_trial = ask_next_trial(study, all_hyperparameters)
        if next_trial is None:
            core_ui.console.print("[green]No pending hyperparameters left in the search space.[/]")
            break
        trial, hyperparameters = next_trial

        try:
            result = _train_next_candidate(
                hyperparameters,
                config,
                dataset,
                models_path,
                paths["training_runs_path"],
                previous_results,
            )
            previous_results.append(result)
            write_training_results(paths["trained_models_path"], previous_results)
            complete_trial(trial, result, config.training_config.optuna_target)
        except Exception as error:
            core_ui.console.print(f"[red]Error training model {hyperparameters.model_name.id}: {error}[/]")
            fail_trial(trial, error)

        _render_status_table(
            [hyperparameters],
            previous_results,
            all_hyperparameters,
            config.training_config.optuna_target,
        )

    core_ui.console.print(
        f"[cyan]Optuna trials: {completed_trial_count(study)}/{config.training_config.max_trials}[/]"
    )


def _training_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _quantization_config(
    hyperparameters: TrainingHyperparameters,
    dtype: torch.dtype,
) -> BitsAndBytesConfig | None:
    if not hyperparameters.use_qlora:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_storage=dtype,
    )


def _load_training_model(
    hyperparameters: TrainingHyperparameters,
    dtype: torch.dtype,
    quantization_config: BitsAndBytesConfig | None,
):
    model = load_model(
        hyperparameters.model_name.local_path or hyperparameters.model_name.id,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    # Required for gradient checkpointing and avoids retaining generation KV
    # cache during supervised fine-tuning.
    model.config.use_cache = False

    tokenizer = load_text_tokenizer(hyperparameters.model_name.local_path or hyperparameters.model_name.id)
    return model, tokenizer


def _evaluate_validation_model(
    dataset,
    hyperparameters: TrainingHyperparameters,
    merged_model_path: Path,
):
    model_info = ModelInfo(
        name=merged_model_path.name,
        id=str(merged_model_path),
        type=hyperparameters.model_name.type,
        size=hyperparameters.model_name.size,
        quantization="4-bit" if hyperparameters.use_qlora else None,
    )
    return evaluate_model(dataset_split=dataset["validation"], model_info=model_info)


def _training_paths(config: HalluCodeDetectionConfig) -> dict[str, Path]:
    results_dir = Path(config.training_config.results_dir or config.dataset_building_config.results_dir)
    checkpoints_path = config.training_config.checkpoints_path
    return {
        "results_dir": results_dir,
        "trained_models_path": results_dir / "trained_models.jsonl",
        "training_runs_path": Path(checkpoints_path) if checkpoints_path else results_dir / "training_runs",
        "optuna_storage_path": results_dir / "optuna_trials.db",
    }


def _render_start_status(
    config: HalluCodeDetectionConfig,
    paths: dict[str, Path],
    all_hyperparameters: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
    trial_count: int,
) -> None:
    configured_keys = {hyperparameters_key(hyperparameters) for hyperparameters in all_hyperparameters}
    json_only = len(
        [
            result
            for result in previous_results
            if hyperparameters_key(result) not in configured_keys
        ]
    )
    core_ui.console.print(
        "[cyan]"
        f"Search: {config.training_config.search_strategy} | "
        f"space={len(all_hyperparameters)} | "
        "next=Optuna TPE | "
        f"executed={trial_count} | "
        f"json-only={json_only}"
        "[/]"
    )
    core_ui.console.print(
        f"[cyan]Optuna TPE: validation target={config.training_config.optuna_target}; "
        f"limit={config.training_config.max_trials}.[/]"
    )


def _load_training_dataset(config: HalluCodeDetectionConfig, results_dir: Path):
    # Use exactly the same sampling and split policy as evaluation.  In
    # particular, never let records for one benchmark task cross a split.
    del results_dir
    return load_hallucination_dataset(config)


def _train_next_candidate(
    hyperparameters: TrainingHyperparameters,
    config: HalluCodeDetectionConfig,
    dataset,
    models_path: Path,
    training_runs_path: Path,
    previous_results: list[TrainingResult],
) -> TrainingResult:
    run_id = uuid4().hex[:8]
    run_path = training_runs_path / run_id
    merged_model_path = models_path / run_id

    candidate_dataset = dataset
    if config.training_config.class_balance_enabled:
        candidate_dataset = balance_training_split(
            dataset=dataset,
            ratio=hyperparameters.class_balance_ratio,
            exclude_syntax=config.training_config.exclude_syntax_from_training,
            seed=config.training_config.random_seed,
        )

    try:
        result = train_model(run_path, merged_model_path, hyperparameters, candidate_dataset)
    except Exception:
        remove_model_dir(str(run_path))
        remove_model_dir(str(merged_model_path))
        raise

    result = apply_saved_model_policy(
        result=result,
        results=previous_results,
        max_saved_models=config.training_config.max_saved_models,
        target=config.training_config.optuna_target,
    )
    remove_model_dir(str(run_path))
    return result


def _render_status_table(
    selected_hyperparameters: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
    all_hyperparameters: list[TrainingHyperparameters],
    target,
) -> None:
    core_ui.console.print(
        render_training_status_table(
            build_training_status_rows(
                display_hyperparameters_for_run(selected_hyperparameters, previous_results),
                previous_results,
                known_hyperparameters=all_hyperparameters,
                target=target,
            )
        )
    )


def _clear_training_locals(local_values: dict[str, object]) -> None:
    for name in ("trainer", "model", "processor"):
        if name in local_values:
            del local_values[name]


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
