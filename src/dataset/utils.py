import json
from typing import TypeVar, cast, List
from pathlib import Path
from datasets import Dataset
from src.schemas.dataset import BaseResultRow, JudgeResultRow
from src.schemas.mbpp import MBPPExample
from src.models.prompts import analyse_hallucination_prompt, get_target_response

JSONL_OBJECT = TypeVar("JSONL_OBJECT")

def load_jsonl(
    path: Path | str,
    return_type: type[JSONL_OBJECT],
    ) -> list[JSONL_OBJECT]:
    path_ = Path(path) if isinstance(path, str) else path
    
    items: list[JSONL_OBJECT] = []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    line = line.strip()
                    if not line:
                        continue

                    json_line = cast(dict[str, object], json.loads(line))

                    if return_type is dict:
                        items.append(cast(JSONL_OBJECT, json_line))
                    else:
                        item = cast(JSONL_OBJECT, return_type(**json_line))
                        items.append(item)
                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        return []

    return items

def append_jsonl(path: Path | str, items: List[dict[str, object]]) -> None:
    path_ = Path(path) if isinstance(path, str) else path
    
    with path_.open("a", encoding="utf-8") as handle:
        for item in items:
            json_line = json.dumps(item, ensure_ascii=True)
            handle.write(json_line + "\n")

def get_pending_tasks(
    mbpp_by_id: dict[int,MBPPExample],
    existing : List[JudgeResultRow], 
    base_results : List[BaseResultRow], 
    judge_model: str
    ) -> List[tuple[BaseResultRow, int]]:
    pending: List[tuple[BaseResultRow, int]] = []
    
    existing_keys: set[tuple[str, int, str, str]] = set()
    
    for existing_item in existing:
        bench_id = existing_item.benchmark_id
        if bench_id is None:
            continue
        existing_keys.add(
            (
                existing_item.benchmark,
                bench_id,
                existing_item.response_model,
                existing_item.judge_model,
            )
        )
        
    for base_item in base_results:
        bench_id = base_item.benchmark_id
        if bench_id is None:
            continue
        if bench_id not in mbpp_by_id:
            continue
        key = (
            base_item.benchmark,
            bench_id,
            base_item.model,
            judge_model,
        )
        
        if key not in existing_keys:
            pending.append((base_item, bench_id))
    
    return pending


def create_conversation(
    problem_description: str,
    generated_code: str,
    level: str,
    explanation: str,
) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "system",
                "content": analyse_hallucination_prompt.format(
                    problem_description=problem_description
                ),
            },
            {"role": "user", "content": generated_code},
            {"role": "assistant", "content": get_target_response(level, explanation)},
        ]
    }

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