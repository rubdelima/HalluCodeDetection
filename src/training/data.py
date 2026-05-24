
from __future__ import annotations

import json
import math
import random
from typing import Iterable, cast

from datasets import Dataset, DatasetDict

from src.dataset.load import load_mbpp_split
from src.dataset.types import BaseResultRow, JudgeResultRow
from src.training.messages import create_conversation


def load_jsonl(path: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(cast(dict[str, object], json.loads(line)))
    except FileNotFoundError:
        return []
    return items


def _to_int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def select_records(
    base_results: Iterable[BaseResultRow],
    judge_results: Iterable[JudgeResultRow],
    judge_model: str | None,
) -> list[dict[str, str]]:
    judge_by_key: dict[tuple[str, int, str], JudgeResultRow] = {}
    for item in judge_results:
        if judge_model and item.get("judge_model") != judge_model:
            continue
        bench_id = _to_int_or_none(item.get("benchmark_id"))
        if bench_id is None:
            continue
        key = (
            str(item.get("benchmark")),
            bench_id,
            str(item.get("response_model")),
        )
        judge_by_key[key] = item

    mbpp_train = load_mbpp_split("train")
    mbpp_by_id = {ex.benchmak_id: ex for ex in mbpp_train}

    records: list[dict[str, str]] = []
    for base_item in base_results:
        bench_id = _to_int_or_none(base_item.get("benchmark_id"))
        if bench_id is None:
            continue
        key = (
            str(base_item.get("benchmark")),
            bench_id,
            str(base_item.get("model")),
        )
        judge_item = judge_by_key.get(key)
        if judge_item is None:
            continue
        example = mbpp_by_id.get(bench_id)
        if example is None:
            continue
        records.append(
            {
                "problem_description": example.prompt,
                "generated_code": str(base_item.get("code", "")),
                "level": str(base_item.get("level", "")),
                "explanation": str(judge_item.get("explanation", "")),
            }
        )
    return records


def apply_sampling(
    records: list[dict[str, str]],
    dataset_load: float,
    correct_size: float,
    seed: int,
) -> list[dict[str, str]]:
    if not records:
        return []

    rng = random.Random(seed)
    correct = [item for item in records if item.get("level") == "correct"]
    other = [item for item in records if item.get("level") != "correct"]

    rng.shuffle(correct)
    if correct_size < 0 or correct_size > 1:
        raise ValueError("correct_size must be in [0, 1]")
    keep_correct = int(math.floor(len(correct) * correct_size))
    selected = other + correct[:keep_correct]

    rng.shuffle(selected)
    if dataset_load <= 0 or dataset_load > 1:
        raise ValueError("dataset_load must be in (0, 1]")
    count = max(1, math.ceil(len(selected) * dataset_load))
    return selected[:count]


def build_dataset(records: list[dict[str, str]]) -> Dataset:
    return Dataset.from_list(records)


def stratified_split(
    dataset: Dataset,
    validation_size: float,
    test_size: float,
    seed: int,
) -> DatasetDict:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be > 0")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be < 1")

    holdout = validation_size + test_size
    stratify_dataset = dataset.map(
        lambda sample: {"stratify_level": sample["level"]},
        batched=False,
    )
    
    stratify_dataset = stratify_dataset.class_encode_column("stratify_level")

    first_split = stratify_dataset.train_test_split(
        test_size=holdout,
        seed=seed,
        stratify_by_column="stratify_level",
    )
    temp = first_split["test"]

    test_ratio = test_size / holdout
    second_split = temp.train_test_split(
        test_size=test_ratio,
        seed=seed,
        stratify_by_column="stratify_level",
    )

    train_split = first_split["train"].remove_columns("stratify_level")
    validation_split = second_split["train"].remove_columns("stratify_level")
    test_split = second_split["test"].remove_columns("stratify_level")

    return DatasetDict(
        {
            "train": train_split,
            "validation": validation_split,
            "test": test_split,
        }
    )


def to_conversation(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda sample: create_conversation(
            sample["problem_description"],
            sample["generated_code"],
            sample["level"],
            sample["explanation"],
        ),
        remove_columns=dataset.column_names,
        batched=False,
    )
