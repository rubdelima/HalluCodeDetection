from datasets import load_dataset
from pydantic import BaseModel
from typing import Iterable, List

import math
import random
import re


class MBPPExample(BaseModel):
    benchmak_id: int
    prompt: str
    tests: list[str]
    function_name: str | None
    function_definition: str | None
    function_signature: str | None
    
    @classmethod
    def from_dataset(cls, example)-> 'MBPPExample':
        code = example['code']
        match = re.search(r'(def\s+\w+\s*\(.*?\)\s*:)', code, re.DOTALL)
        function_definition = match.group(1).strip() if match else None

        name_match = re.search(r'def\s+(\w+)\s*\(', function_definition or "")
        function_name = name_match.group(1) if name_match else None

        # Keep only the signature line (single-line form) for prompts
        signature_match = re.search(r'(def\s+\w+\s*\(.*?\)\s*:)', function_definition or "")
        function_signature = signature_match.group(1).replace("\n", " ").strip() if signature_match else None
        
        if not function_name:
            print(f"Could not extract function name from code: {code}")
        
        return cls(
            benchmak_id=example['task_id'],
            prompt=example['prompt'],
            tests=example['test_list'],
            function_name=function_name,
            function_definition=function_definition,
            function_signature=function_signature
        )
    
    @classmethod
    def from_split(cls, split)->List['MBPPExample']:
        dataset = load_dataset("mbpp", "sanitized")
        split = dataset.get(split)
        if not split:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        return [cls.from_dataset(example) for example in split]
    
    @property
    def benchmark_name(self)->str:
        return f"mbpp"


def load_mbpp_split(split: str) -> List[MBPPExample]:
    return MBPPExample.from_split(split)


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