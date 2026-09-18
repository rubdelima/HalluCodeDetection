from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.dataset import Level
from src.static_analysis.feedback import StaticToolFeedback


class Phase6Round(BaseModel):
    round_number: int
    code: str
    tool_feedback: StaticToolFeedback
    decision: str | None = None
    assessment: str = ""
    review_raw_response: str = ""
    review_thoughts: str | None = None
    critique: str = ""
    critique_raw_response: str = ""


class Phase6ResultRow(BaseModel):
    benchmark: str
    benchmark_id: int
    task_id: str
    model: str
    rounds: list[Phase6Round] = Field(default_factory=list)
    final_code: str = ""
    levels: list[Level] = Field(default_factory=list)
    tc_ok: int = 0
    tc_fail: int = 0
    base_passed: bool | None = None
    plus_passed: bool | None = None
    exception_type: str | None = None
    failure_stage: str | None = None
    error: str | None = None
