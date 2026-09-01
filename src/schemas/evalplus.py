from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel


def extract_signature(code: str) -> str | None:
    match = re.search(
        r"def\s+\w+\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:",
        code,
        re.DOTALL,
    )
    if not match:
        return None
    return " ".join(match.group(0).split())


class EvalPlusExample(BaseModel):
    benchmark_name: str
    benchmark_id: int
    task_id: str
    prompt: str
    entry_point: str
    function_signature: str | None
    is_completion: bool
    canonical_solution: str
    base_input: list[Any]
    plus_input: list[Any]
    atol: float
    base_expected: list[Any] = []
    plus_expected: list[Any] = []

    @classmethod
    def from_evalplus(
        cls,
        benchmark_name: str,
        task_id: str,
        problem: dict[str, Any],
    ) -> "EvalPlusExample":
        is_completion = benchmark_name == "humaneval"

        canonical_solution = problem["prompt"] + problem["canonical_solution"]
        signature_source = problem["prompt"] if is_completion else problem["canonical_solution"]

        return cls(
            benchmark_name=benchmark_name,
            benchmark_id=int(task_id.split("/")[1]),
            task_id=task_id,
            prompt=problem["prompt"],
            entry_point=problem["entry_point"],
            function_signature=extract_signature(signature_source),
            is_completion=is_completion,
            canonical_solution=canonical_solution,
            base_input=list(problem.get("base_input", [])),
            plus_input=list(problem.get("plus_input", [])),
            atol=float(problem.get("atol", 0)),
        )
