
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable, List
from pydantic import BaseModel
from pathlib import Path

from datasets import Dataset, DatasetDict
from src.dataset.load import load_all_examples
from src.schemas.dataset import BaseResultRow, JudgeResultRow, JudgeExplanation
from src.schemas.evalplus import EvalPlusExample
from src.core import ui
from src.constants import HalluCodeDetectionConfig
from src.dataset.utils import load_jsonl

class Record(BaseModel):
    benchmark: str
    benchmark_id: int
    task_id: str
    problem_description: str
    generated_code: str
    level: str
    levels: list[str]
    explanations: list[JudgeExplanation]
    
def select_records(
    base_results: Iterable[BaseResultRow],
    judge_results: Iterable[JudgeResultRow],
    examples_by_key: dict[tuple[str, int], EvalPlusExample],
) -> List[Record]:
    
    base_by_key : dict[tuple[str, int, str], BaseResultRow] = {
            (base_item.benchmark, base_item.benchmark_id, base_item.model): base_item
            for base_item in base_results
    }
    
    records = []
    for judge_item in judge_results:
        base_item = base_by_key.get(
            (judge_item.benchmark, judge_item.benchmark_id, judge_item.response_model)
        )
        if not base_item:
            continue
        
        example = examples_by_key.get((judge_item.benchmark, judge_item.benchmark_id))
        if not example:
            continue
        
        records.append(Record(
            benchmark=base_item.benchmark,
            benchmark_id=base_item.benchmark_id,
            task_id=base_item.task_id or f"{base_item.benchmark}/{base_item.benchmark_id}",
            problem_description=example.prompt,
            generated_code=base_item.code,
            level=base_item.primary_level_name,
            levels=[lv.level_name for lv in base_item.levels],
            explanations=judge_item.explanations,
        ))
        
    return records


def apply_sampling(
    records: List[Record],
    dataset_load: float,
    correct_size: float,
    seed: int,
) -> list[Record]:
    
    if not records:
        return []

    rng = random.Random(seed)
    correct = [item for item in records if item.level == "correct"]
    other = [item for item in records if item.level != "correct"]

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


def build_dataset(records: list[Record]) -> Dataset:
    return Dataset.from_list([record.model_dump() for record in records])


def balance_training_split(
    dataset: DatasetDict,
    ratio: tuple[int, int, int],
    exclude_syntax: bool,
    seed: int,
) -> DatasetDict:
    """Downsample the training split to a runtime:functional:correct ratio.

    Splitting occurs before sampling, so this operation cannot move a task
    between train, validation, and test. Validation and test stay untouched.
    """
    labels = ("runtime", "functional", "correct")
    if len(ratio) != len(labels) or any(value <= 0 for value in ratio):
        raise ValueError("Class-balance ratio must contain three positive values: runtime, functional, correct.")

    train = dataset["train"]
    buckets: dict[str, list[int]] = {label: [] for label in labels}
    for index, record in enumerate(train):
        record_labels = set(record.get("levels", []))
        if exclude_syntax and "syntax" in record_labels:
            continue
        primary_label = record["level"]
        if primary_label in buckets:
            buckets[primary_label].append(index)

    scale = min(len(buckets[label]) // weight for label, weight in zip(labels, ratio))
    if scale < 1:
        available = ", ".join(f"{label}={len(buckets[label])}" for label in labels)
        raise ValueError(
            "Unable to create the requested balanced training split; "
            f"available syntax-free primary labels: {available}."
        )

    rng = random.Random(seed)
    selected: list[int] = []
    for label, weight in zip(labels, ratio):
        indices = buckets[label].copy()
        rng.shuffle(indices)
        selected.extend(indices[: scale * weight])
    rng.shuffle(selected)

    counts = ", ".join(f"{label}={scale * weight}" for label, weight in zip(labels, ratio))
    ui.console.print(
        "[cyan]Training sampling: "
        f"runtime:functional:correct={':'.join(map(str, ratio))}; "
        f"syntax excluded={exclude_syntax}; {counts}.[/]"
    )
    return DatasetDict({
        "train": train.select(selected),
        "validation": dataset["validation"],
        "test": dataset["test"],
    })

def grouped_split(
    dataset: Dataset,
    validation_size: float,
    test_size: float,
    seed: int,
) -> DatasetDict:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be > 0")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be < 1")

    task_ids = dataset["task_id"]
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        by_task[task_id].append(index)

    unique_tasks = list(by_task)
    total_tasks = len(unique_tasks)
    n_test = max(1, int(round(total_tasks * test_size)))
    n_val = max(1, int(round(total_tasks * validation_size)))
    if n_test + n_val >= total_tasks:
        n_test = 1
        n_val = 1 if total_tasks > 2 else 0

    # Assign whole tasks, not individual rows.  The greedy objective matches
    # each split's class histogram and its record count to the requested
    # fractions, while the task quotas make the split sizes reproducible.
    split_names = ("train", "validation", "test")
    fractions = {"train": 1 - validation_size - test_size, "validation": validation_size, "test": test_size}
    task_targets = {"train": total_tasks - n_val - n_test, "validation": n_val, "test": n_test}
    labels = sorted(set(dataset["level"]))
    total_labels = Counter(dataset["level"])
    target_labels = {
        split: {label: total_labels[label] * fractions[split] for label in labels}
        for split in split_names
    }
    target_records = {split: len(dataset) * fractions[split] for split in split_names}
    split_labels = {split: Counter() for split in split_names}
    split_records = Counter()
    split_tasks = Counter()

    rng = random.Random(seed)
    rng.shuffle(unique_tasks)
    unique_tasks.sort(key=lambda task: len(by_task[task]), reverse=True)

    split_of: dict[str, str] = {}
    for task in unique_tasks:
        group_labels = Counter(dataset[index]["level"] for index in by_task[task])
        candidates = [split for split in split_names if split_tasks[split] < task_targets[split]]

        def score(split: str) -> float:
            class_score = 0.0
            for name in split_names:
                for label in labels:
                    observed = split_labels[name][label] + (group_labels[label] if name == split else 0)
                    target = max(target_labels[name][label], 1.0)
                    class_score += ((observed - target) ** 2) / target
            size_score = 0.0
            for name in split_names:
                observed = split_records[name] + (len(by_task[task]) if name == split else 0)
                target = max(target_records[name], 1.0)
                size_score += ((observed - target) ** 2) / target
            return class_score + size_score

        selected = min(candidates, key=score)
        split_of[task] = selected
        split_labels[selected].update(group_labels)
        split_records[selected] += len(by_task[task])
        split_tasks[selected] += 1

    train_idx = [i for i, t in enumerate(task_ids) if split_of[t] == "train"]
    val_idx = [i for i, t in enumerate(task_ids) if split_of[t] == "validation"]
    test_idx = [i for i, t in enumerate(task_ids) if split_of[t] == "test"]

    return DatasetDict(
        {
            "train": dataset.select(train_idx),
            "validation": dataset.select(val_idx),
            "test": dataset.select(test_idx),
        }
    )



def load_hallucination_dataset(
    config: HalluCodeDetectionConfig,
    correct_size: float | None = None,
)->DatasetDict:
    results_dir = Path(config.dataset_building_config.results_dir)
    judge_dir = Path(config.dataset_building_config.judge_results_dir)
    base_path = results_dir / "dataset_base.json"
    judge_path = judge_dir / "dataset_judge.jsonl"
    
    empty_split = DatasetDict({
        "train": Dataset.from_list([]),
        "validation": Dataset.from_list([]),
        "test": Dataset.from_list([]),
    })
    
    if not base_path.exists():
        ui.console.print("No base results found. Run --build_dataset first.")
        return empty_split
    
    if not judge_path.exists():
        ui.console.print("No judge results found. Run --dataset_judge first.")
        return empty_split
    
    base_results = load_jsonl(base_path, BaseResultRow)
    if not base_results:
        ui.console.print("No base results found to evaluate.")
        return empty_split
    
    judge_results = load_jsonl(judge_path, JudgeResultRow)
    if not judge_results:
        ui.console.print("No judge results found to evaluate.")
        return empty_split

    examples_by_key = load_all_examples(config.dataset_building_config.datasets_base)

    records = select_records(base_results, judge_results, examples_by_key)
    records = apply_sampling(
        records=records, 
        dataset_load=config.dataset_config.load_size, 
        correct_size=config.dataset_config.correct_size if correct_size is None else correct_size,
        seed=config.dataset_config.random_seed
    )
    
    if not records:
        ui.console.print("No evaluation records after filtering.")
        return empty_split

    return grouped_split(
        dataset=build_dataset(records), 
        validation_size=config.dataset_config.validation_size, 
        test_size=config.dataset_config.test_size, 
        seed=config.dataset_config.random_seed
    )
