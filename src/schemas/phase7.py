from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.judge import JudgeResponse
from src.static_analysis.feedback import StaticToolFeedback


class Phase7ResultRow(BaseModel):
    model_id: str
    kind: str
    sample_id: str
    benchmark: str
    benchmark_id: int
    task_id: str
    response_model: str
    expected_levels: list[str]
    predicted_levels: list[str] = Field(default_factory=list)
    correct: bool | None = None
    tool_feedback: StaticToolFeedback
    judge_response: JudgeResponse
    error: str | None = None
