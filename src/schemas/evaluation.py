from pydantic import BaseModel, Field

from src.schemas.judge import LEVEL_TYPE, JudgeResponse
from typing import Optional, Dict

class EvaluationSummaryRow(BaseModel):
    model_id : str = Field(..., description="The identifier of the model being evaluated.")
    kind : str = Field(..., description="The kind of llm provider (like ollama, gemma, etc).")
    sample_index : int = Field(..., description="The index of the sample in the dataset.")
    expected_level : LEVEL_TYPE = Field(..., description="The expected level of error for the given sample, as determined by the dataset or ground truth.")
    predicted_level : Optional[LEVEL_TYPE] = Field(None, description="The predicted level of error for the given sample.")
    correct : Optional[bool] = Field(None, description="Whether the predicted level matches the expected level, indicating a correct prediction.")
    judge_response : JudgeResponse = Field(..., description="The detailed response from the judge model, including the analysis and explanation of the prediction.")

BASE_LEVEL_DICT = {
    "syntax_error": 0,
    "runtime_error": 0,
    "functional_error": 0,
    "correct": 0
}

class EvaluationResume(BaseModel):
    parsed_responses : int = Field(0, description="The number of responses that were successfully parsed and included in the evaluation.")
    total_responses : int = Field(0, description="The total number of responses that were evaluated, including those that could not be parsed.")
    overall_accuracy : float = Field(0.0, description="The overall accuracy of the model across all evaluated samples, calculated as the number of correct predictions divided by the total number of parsed responses.")
    corrects_by_level : Dict[LEVEL_TYPE, int] = Field(BASE_LEVEL_DICT, description="A breakdown of accuracy by error level, showing the accuracy for each specific level of error.")
    evaluations : list[EvaluationSummaryRow] = Field([], description="A list of individual evaluation results for each sample, providing detailed information about the model's performance on each case.")