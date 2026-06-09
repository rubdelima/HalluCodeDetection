from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Tuple

from src.constants.models import ModelInfo, get_models_options

class EvaluationConfig(BaseModel):
    checkpoint_interval: int = Field(10, description="Interval (in number of samples) at which to save intermediate evaluation results.")
    models: List[ModelInfo] = Field(..., description="List of model configurations to evaluate.")
    model_temperature: float = Field(0.0, description="Temperature to use for judge model generation.")
    overwrite: bool = Field(False, description="Whether to overwrite existing evaluation results.")
    
    @classmethod
    def from_config(cls, config: dict, models_options: Optional[Dict[str, ModelInfo]] = None) -> "EvaluationConfig":
        models_options = get_models_options(config) if models_options is not None else {}
        models_id = config.get("evaluation", {}).get("models", [])
        models = [model for model_id, model in models_options.items() if model_id in models_id]
        
        return cls(
            checkpoint_interval=config.get("evaluation", {}).get("checkpoint_interval", 10),
            models=models,
            model_temperature=config.get("evaluation", {}).get("model_temperature", 0.0),
            overwrite=config.get("evaluation", {}).get("overwrite", False),
        )