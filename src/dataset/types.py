from __future__ import annotations

from typing import TypedDict


class BaseResultRow(TypedDict):
    benchmark: str
    benchmark_id: int
    model: str
    level: str
    code: str
    error: str


class JudgeResultRow(TypedDict):
    benchmark: str
    benchmark_id: int
    response_model: str
    judge_model: str
    explanation: str
