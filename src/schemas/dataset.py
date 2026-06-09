from pydantic import BaseModel

class BaseResultRow(BaseModel):
    benchmark: str
    benchmark_id: int
    model: str
    level: str
    code: str
    error: str

class JudgeResultRow(BaseModel):
    benchmark: str
    benchmark_id: int
    response_model: str
    judge_model: str
    explanation: str