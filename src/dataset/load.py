from datasets import load_dataset
from typing import Iterable, List

import math
import random

from src.schemas.mbpp import MBPPExample

def load_mbpp_split(split: str) -> List[MBPPExample]:
    dataset = load_dataset("mbpp", "sanitized")
    split = dataset.get(split)
    if not split:
        raise ValueError(f"Split '{split}' not found in the dataset.")
    return [MBPPExample.from_dataset(example) for example in split]

def sample_examples(
    examples: Iterable[MBPPExample],
    fraction: float,
    seed: int | None = None,
) -> List[MBPPExample]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    examples_list = list(examples)
    if not examples_list:
        return []
    count = max(1, math.ceil(len(examples_list) * fraction))
    rng = random.Random(seed)
    rng.shuffle(examples_list)
    return examples_list[:count]