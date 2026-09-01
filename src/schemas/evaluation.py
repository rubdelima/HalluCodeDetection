from pydantic import BaseModel, Field

from src.schemas.judge import LEVEL_TYPE, JudgeResponse
from typing import Optional, Dict

class EvaluationSummaryRow(BaseModel):
    model_id : str = Field(..., description="The identifier of the model being evaluated.")
    kind : str = Field(..., description="The kind of llm provider (like ollama, gemma, etc).")
    sample_index : int = Field(..., description="The index of the sample in the dataset.")
    expected_levels : list[str] = Field(..., description="The expected error levels for the given sample, as determined by the dataset or ground truth.")
    predicted_levels : list[str] = Field(default_factory=list, description="The predicted error levels for the given sample.")
    correct : Optional[bool] = Field(None, description="Whether the set of predicted levels matches the expected levels.")
    judge_response : JudgeResponse = Field(..., description="The detailed response from the judge model, including the analysis and explanation of the prediction.")
    error : Optional[str] = Field(None, description="The error that prevented this sample from being evaluated, if any.")

BASE_LEVEL_DICT = {
    "syntax": 0,
    "runtime": 0,
    "functional": 0,
    "correct": 0
}

class EvaluationResume(BaseModel):
    parsed_responses : int = Field(0, description="The number of responses that were successfully parsed and included in the evaluation.")
    total_responses : int = Field(0, description="The total number of responses that were evaluated, including those that could not be parsed.")
    skipped_responses : int = Field(0, description="The number of samples skipped because their evaluation raised an exception.")
    overall_accuracy : float = Field(0.0, description="The overall accuracy of the model across all evaluated samples, calculated as the number of correct predictions divided by the total number of parsed responses.")
    macro_f1: float = Field(0.0, description="Macro F1 across correct, functional, runtime, and syntax labels.")
    macro_recall: float = Field(0.0, description="Macro recall across correct, functional, runtime, and syntax labels.")
    corrects_by_level : Dict[LEVEL_TYPE, int] = Field(default_factory=lambda: BASE_LEVEL_DICT.copy(), description="A breakdown of accuracy by error level, showing the accuracy for each specific level of error.")
    true_positives: Dict[LEVEL_TYPE, int] = Field(default_factory=lambda: BASE_LEVEL_DICT.copy())
    false_positives: Dict[LEVEL_TYPE, int] = Field(default_factory=lambda: BASE_LEVEL_DICT.copy())
    false_negatives: Dict[LEVEL_TYPE, int] = Field(default_factory=lambda: BASE_LEVEL_DICT.copy())
    evaluations : list[EvaluationSummaryRow] = Field(default_factory=list, description="A list of individual evaluation results for each sample, providing detailed information about the model's performance on each case.")
