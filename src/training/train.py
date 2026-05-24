from __future__ import annotations

import itertools
from typing import cast

from src.core import ui
from src.dataset.types import BaseResultRow, JudgeResultRow
from src.training.config import parse_training_settings
from src.training.data import (
    apply_sampling,
    build_dataset,
    load_jsonl,
    select_records,
    stratified_split,
    to_conversation,
)
from src.training.runner import run_grid_search


def train_model(config: dict[str, object], model_name: str | None = None) -> None:
    settings, paths = parse_training_settings(config, model_name)

    if settings.validation_size <= 0 or settings.test_size <= 0:
        raise ValueError("validation_size and test_size must be > 0")
    if settings.validation_size + settings.test_size >= 1:
        raise ValueError("validation_size + test_size must be < 1")

    base_results = [cast(BaseResultRow, item) for item in load_jsonl(str(paths.base_path))]
    judge_results = [cast(JudgeResultRow, item) for item in load_jsonl(str(paths.judge_path))]

    if not base_results:
        ui.console.print("No base results found. Run --build_dataset first.")
        return
    if not judge_results:
        ui.console.print("No judge results found. Run --dataset_judge first.")
        return

    records = select_records(base_results, judge_results, paths.judge_model)
    records = apply_sampling(records, settings.dataset_load, settings.correct_size, settings.random_seed)
    if not records:
        ui.console.print("No training records after filtering.")
        return
    if len(records) < 3:
        ui.console.print("Not enough training records to split. Add more data.")
        return

    raw_dataset = build_dataset(records)
    split_raw = stratified_split(
        raw_dataset,
        settings.validation_size,
        settings.test_size,
        settings.random_seed,
    )

    train_dataset = to_conversation(split_raw["train"])
    validation_dataset = split_raw["validation"]
    test_dataset = split_raw["test"]

    grid = list(
        itertools.product(
            settings.lora_r_list,
            settings.lora_alpha_list,
            settings.lora_dropout_list,
            settings.learning_rate_list,
        )
    )

    run_grid_search(
        settings,
        paths,
        train_dataset,
        validation_dataset,
        grid,
    )
