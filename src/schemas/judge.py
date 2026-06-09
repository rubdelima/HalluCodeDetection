from pydantic import BaseModel, Field
from typing import Literal, Optional
import json

LEVEL_TYPE = Literal["syntax_error", "runtime_error", "functional_error", "correct"]

LEVEL_DESCRIPTIONS = """The level of error in the code.

- syntax_error: When the provided code has a syntax error in the Python language, such as incorrect or incomplete code.

- runtime_error: When the code is "compilable" but has an error when executed, such as a call to an invalid function, an undefined variable, etc.

- functional_error: When the code is executable, but it has some deviation from the correct functioning of what was requested.

- correct: When the code is well-formed and has no functional errors.
"""


class JudgeAnalysis(BaseModel):
    level: LEVEL_TYPE = Field(..., description=LEVEL_DESCRIPTIONS)
    analysis: str  = Field(..., description="A detailed explanation of the analysis, including why the code is correct or incorrect, and any errors if applicable.")

class JudgeResponse(BaseModel):
    analysis: Optional[JudgeAnalysis] = Field(None, description="The explanation of the analysis, including the level of error and a detailed explanation.")
    raw_response: str = Field(..., description="The raw JSON response as returned by the model, for debugging purposes.")
    thoughts: Optional[str] = Field(None, description="Any additional thoughts or insights about the analysis, if applicable.")
