from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from src.constants import HalluCodeDetectionConfig
from src.constants.models import ModelInfo
from src.constants.training import TrainingHyperparameters
from src.core import ui as core_ui
from src.dataset.judge_dataset import build_dataset, select_records, stratified_split
from src.dataset.utils import load_jsonl
from src.evaluations import evaluate_model
from src.schemas.dataset import BaseResultRow, JudgeResultRow
from src.schemas.training import TrainingResult
from src.training.hyperparams import get_trainer
from src.training.search import select_pending_hyperparameters
from src.training.state import (
    build_training_status_rows,
    display_hyperparameters_for_run,
    hyperparameters_key,
    load_training_results,
    write_training_results,
)
from src.training.storage import apply_saved_model_policy, remove_model_dir, save_merged_model
from src.training.ui import render_training_status_table


def train_model(
    run_path: Path,
    merged_model_path: Path,
    hyperparameters: TrainingHyperparameters,
    dataset,
) -> Optional[TrainingResult]:
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
            base_model_id=hyperparameters.model_name.id,
            adapter_path=run_path,
            merged_model_path=merged_model_path,
            dtype=dtype,
        )
        train_result, validation_result, test_result = _evaluate_trained_model(
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
            train_acc=train_result.overall_accuracy,
            val_acc=validation_result.overall_accuracy,
            test_acc=test_result.overall_accuracy,
        )
    except Exception as error:
        core_ui.console.print(f"[red]Error training model {hyperparameters.model_name.id}: {error}[/]")
        return None
    finally:
        _clear_training_locals(locals())
        _clear_memory()


def train_models(config: HalluCodeDetectionConfig) -> None:
    paths = _training_paths(config)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    paths["training_runs_path"].mkdir(parents=True, exist_ok=True)

    previous_results = load_training_results(paths["trained_models_path"])
    all_hyperparameters = config.training_config.hyperparameters
    initial_selection = _select_next_candidates(config, all_hyperparameters, previous_results)

    _render_start_status(config, paths, all_hyperparameters, initial_selection, previous_results)
    if not initial_selection:
        core_ui.console.print("[yellow]No pending hyperparameters to train.[/]")
        return

    dataset = _load_training_dataset(config, paths["results_dir"])
    models_path = Path(config.training_config.models_path)
    models_path.mkdir(parents=True, exist_ok=True)

    while True:
        next_selection = _select_next_candidates(config, all_hyperparameters, previous_results)
        if not next_selection:
            core_ui.console.print("[green]No pending hyperparameters left in the search space.[/]")
            break

        result = _train_next_candidate(
            next_selection[0],
            config,
            dataset,
            models_path,
            paths["training_runs_path"],
            previous_results,
        )
        if result is not None:
            previous_results.append(result)
            write_training_results(paths["trained_models_path"], previous_results)

        _render_status_table(next_selection, previous_results, all_hyperparameters)


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
    model = AutoModelForImageTextToText.from_pretrained(
        hyperparameters.model_name.id,
        dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    processor = AutoProcessor.from_pretrained(hyperparameters.model_name.id)
    return model, processor


def _evaluate_trained_model(
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
    results = []
    for split_name in ("train", "validation", "test"):
        result = evaluate_model(dataset_split=dataset[split_name], model_info=model_info)
        results.append(result)

    return tuple(results)


def _training_paths(config: HalluCodeDetectionConfig) -> dict[str, Path]:
    results_dir = Path(config.dataset_building_config.results_dir)
    return {
        "results_dir": results_dir,
        "trained_models_path": results_dir / "trained_models.jsonl",
        "training_runs_path": results_dir / "training_runs",
    }


def _select_next_candidates(
    config: HalluCodeDetectionConfig,
    all_hyperparameters: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
) -> list[TrainingHyperparameters]:
    limit = 1 if config.training_config.search_strategy == "bayesian" else None
    return select_pending_hyperparameters(
        all_hyperparameters,
        previous_results,
        config.training_config,
        limit=limit,
    )


def _render_start_status(
    config: HalluCodeDetectionConfig,
    paths: dict[str, Path],
    all_hyperparameters: list[TrainingHyperparameters],
    initial_selection: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
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
        f"next={len(initial_selection)} | "
        f"executed={len(previous_results)} | "
        f"json-only={json_only}"
        "[/]"
    )
    _render_status_table(initial_selection, previous_results, all_hyperparameters)
    if initial_selection:
        core_ui.console.print("[cyan]Bayesian search will pick the next candidate after each result.[/]")


def _load_training_dataset(config: HalluCodeDetectionConfig, results_dir: Path):
    base_results = load_jsonl(results_dir / "dataset_base.json", BaseResultRow)
    judge_results = load_jsonl(results_dir / "dataset_judge.jsonl", JudgeResultRow)
    records = select_records(base_results, judge_results)
    dataset = build_dataset(records)
    return stratified_split(dataset, validation_size=0.1, test_size=0.2, seed=42)


def _train_next_candidate(
    hyperparameters: TrainingHyperparameters,
    config: HalluCodeDetectionConfig,
    dataset,
    models_path: Path,
    training_runs_path: Path,
    previous_results: list[TrainingResult],
) -> TrainingResult | None:
    run_id = uuid4().hex[:8]
    run_path = training_runs_path / run_id
    merged_model_path = models_path / run_id

    result = train_model(run_path, merged_model_path, hyperparameters, dataset)
    if result is None:
        remove_model_dir(str(run_path))
        remove_model_dir(str(merged_model_path))
        return None

    result = apply_saved_model_policy(
        result=result,
        results=previous_results,
        max_saved_models=config.training_config.max_saved_models,
    )
    remove_model_dir(str(run_path))
    return result


def _render_status_table(
    selected_hyperparameters: list[TrainingHyperparameters],
    previous_results: list[TrainingResult],
    all_hyperparameters: list[TrainingHyperparameters],
) -> None:
    core_ui.console.print(
        render_training_status_table(
            build_training_status_rows(
                display_hyperparameters_for_run(selected_hyperparameters, previous_results),
                previous_results,
                known_hyperparameters=all_hyperparameters,
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
