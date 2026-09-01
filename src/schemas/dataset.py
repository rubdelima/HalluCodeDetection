from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Level:
    level_name: str
    evidences: list[str] = field(default_factory=list)


@dataclass
class JudgeExplanation:
    level: str
    explanation: str


class BaseResultRow(BaseModel):
    benchmark: str
    benchmark_id: int
    task_id: str | None = None
    model: str
    levels: list[Level]
    tc_ok: int
    tc_fail: int
    code: str = ""
    base_passed: bool | None = None
    plus_passed: bool | None = None
    exception_type: str | None = None
    failure_stage: str | None = None

    @property
    def primary_level_name(self) -> str:
        return self.levels[0].level_name if self.levels else "correct"


class JudgeResultRow(BaseModel):
    benchmark: str
    benchmark_id: int
    response_model: str
    judge_model: str
    explanations: list[JudgeExplanation]
