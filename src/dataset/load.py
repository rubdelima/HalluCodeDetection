from __future__ import annotations

import math
import random
from typing import Iterable, TypeVar

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from src.schemas.evalplus import EvalPlusExample

T = TypeVar("T")

EVALPLUS_LOADERS = {
    "humaneval": get_human_eval_plus,
    "mbpp": get_mbpp_plus,
}


def load_evalplus(benchmark_name: str) -> list[EvalPlusExample]:
    loader = EVALPLUS_LOADERS.get(benchmark_name)
    if loader is None:
        raise ValueError(f"Unsupported EvalPlus dataset: {benchmark_name}")
    dataset = loader()
    return [
        EvalPlusExample.from_evalplus(benchmark_name, task_id, problem)
        for task_id, problem in dataset.items()
    ]


def load_all_examples(benchmarks: Iterable[str]) -> dict[tuple[str, int], EvalPlusExample]:
    by_key: dict[tuple[str, int], EvalPlusExample] = {}
    for benchmark in benchmarks:
        for example in load_evalplus(benchmark):
            by_key[(example.benchmark_name, example.benchmark_id)] = example
    return by_key


def sample_examples(
    examples: Iterable[T],
    fraction: float,
    seed: int | None = None,
) -> list[T]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    examples_list = list(examples)
    if not examples_list:
        return []
    count = max(1, math.ceil(len(examples_list) * fraction))
    rng = random.Random(seed)
    rng.shuffle(examples_list)
    return examples_list[:count]
