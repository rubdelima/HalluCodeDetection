from pydantic import BaseModel, Field
from typing import Literal, List, Dict

MODELS_TYPE = Literal["ollama", "gemma"]

class ModelInfo(BaseModel):
    name: str = Field(..., description="Name of the model.")
    id: str = Field(..., description="ID of the model, if applicable.")
    type: MODELS_TYPE = Field(..., description="Type of the model.")
    size: float = Field(..., description="Number of parameters in the model, in billions.")
    quantization : str | None = Field(None, description="Quantization method to use for the model, if applicable")
    
def get_models_options(config: dict)->Dict[str, ModelInfo]:
    models = {}
    for model_type in ["ollama", "gemma"]:
        models_data = config.get("models", {}).get(model_type, [])
        for model in models_data:
            models[model["id"]] = ModelInfo(type=model_type, **model) #type: ignore
    return models