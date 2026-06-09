
from __future__ import annotations

import math
import random
from typing import Iterable, List
from pydantic import BaseModel
from pathlib import Path

from datasets import Dataset, DatasetDict
from src.dataset.load import load_mbpp_split
from src.schemas.dataset import BaseResultRow, JudgeResultRow
from src.core import ui
from datasets import DatasetDict, Dataset
from src.constants import HalluCodeDetectionConfig
from src.dataset.utils import load_jsonl

class Record(BaseModel):
    problem_description: str
    generated_code: str
    level: str
    explanation: str
    
def select_records(
    base_results: Iterable[BaseResultRow],
    judge_results: Iterable[JudgeResultRow],
    mbpp_split: str = "train",
) -> List[Record]:
    
    mbpp_items = load_mbpp_split(mbpp_split)
    mbpp_items_by_id = {item.benchmark_id: item for item in mbpp_items}
    
    base_by_id : dict[tuple[int, str], BaseResultRow] = {
            (base_item.benchmark_id, base_item.model): base_item
            for base_item in base_results
    }
    
    records = []
    for judge_item in judge_results:
        key = (judge_item.benchmark_id, judge_item.response_model)
        
        base_item = base_by_id.get(key)
        if not base_item:
            continue
        
        mbpp_item = mbpp_items_by_id.get(judge_item.benchmark_id)
        if not mbpp_item:
            continue
        
        records.append(Record(
            problem_description=mbpp_item.prompt,
            generated_code=base_item.code,
            level=base_item.level,
            explanation=judge_item.explanation,
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



def load_hallucination_dataset(config: HalluCodeDetectionConfig)->DatasetDict:
    results_dir = Path(config.dataset_building_config.results_dir)
    base_path = results_dir / "dataset_base.json"
    judge_path = results_dir / "dataset_judge.jsonl"
    
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

    records = select_records(base_results, judge_results)
    records = apply_sampling(
        records=records, 
        dataset_load=config.dataset_config.load_size, 
        correct_size=config.dataset_config.correct_size, 
        seed=config.dataset_config.random_seed
    )
    
    if not records:
        ui.console.print("No evaluation records after filtering.")
        return empty_split

    return stratified_split(
        dataset=build_dataset(records), 
        validation_size=config.dataset_config.validation_size, 
        test_size=config.dataset_config.test_size, 
        seed=config.dataset_config.random_seed
    )
