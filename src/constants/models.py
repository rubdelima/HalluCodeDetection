from pydantic import BaseModel, Field
from typing import Any, Literal, Dict

MODELS_TYPE = Literal["ollama", "gemma", "opencode_go"]

class ModelInfo(BaseModel):
    name: str = Field(..., description="Name of the model.")
    id: str = Field(..., description="ID of the model, if applicable.")
    type: MODELS_TYPE = Field(..., description="Type of the model.")
    size: float = Field(..., description="Number of parameters in the model, in billions.")
    local_path: str | None = Field(None, description="Optional local checkpoint directory used instead of downloading from the Hub.")
    quantization : str | None = Field(None, description="Quantization method to use for the model, if applicable")
    num_ctx : int | None = Field(None, description="Maximum context window size (used by Ollama).")
    think : bool | None = Field(None, description="Whether to enable the reasoning/thinking mode (Ollama).")
    max_tokens: int = Field(4096, gt=0, description="Maximum number of generated tokens (used by remote APIs).")
    request_timeout: int = Field(120, gt=0, description="Remote API request timeout in seconds.")
    threads: int = Field(1, gt=0, description="Concurrent requests per model (used only by OpenCode Go).")
    max_retries: int = Field(3, ge=0, description="Retries for transient OpenCode Go failures.")
    retry_base_delay: float = Field(2.0, gt=0, description="Initial OpenCode Go retry delay in seconds.")
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = Field(
        None,
        description="OpenAI-compatible reasoning effort override used by OpenCode Go.",
    )
    
def get_models_options(config: dict[str, Any])->Dict[str, ModelInfo]:
    models = {}
    for model_type in ["ollama", "gemma", "opencode_go"]:
        models_data = config.get("models", {}).get(model_type, [])
        for model in models_data:
            models[model["id"]] = ModelInfo(type=model_type, **model)
    return models
