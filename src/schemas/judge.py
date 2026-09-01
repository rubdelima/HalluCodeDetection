from pydantic import BaseModel, Field
from typing import Literal, Optional

from src.schemas.dataset import JudgeExplanation

LEVEL_TYPE = Literal["syntax", "runtime", "functional", "correct"]

LEVEL_DESCRIPTIONS = """The level of error in the code.

- syntax: When the provided code has a syntax error in the Python language, such as incorrect or incomplete code.

- runtime: When the code is "compilable" but has an error when executed, such as a call to an invalid function, an undefined variable, etc.

- functional: When the code is executable, but it has some deviation from the correct functioning of what was requested.

- correct: When the code is well-formed and has no functional errors.
"""


class JudgeAnalysis(BaseModel):
    levels: list[JudgeExplanation] = Field(default_factory=list, description="List of predicted error levels with a general explanation for each.")

class JudgeResponse(BaseModel):
    analysis: Optional[JudgeAnalysis] = Field(None, description="The explanation of the analysis, including the levels of error and a detailed explanation.")
    raw_response: str = Field(..., description="The raw JSON response as returned by the model, for debugging purposes.")
    thoughts: Optional[str] = Field(None, description="Any additional thoughts or insights about the analysis, if applicable.")
