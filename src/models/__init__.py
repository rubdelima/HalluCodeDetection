from .base import BaseModelHandler

from src.constants.models import ModelInfo, get_models_options

def get_model_handler(model: ModelInfo) -> BaseModelHandler:
    if model.type == "ollama":
        from .ollama_handler import OllamaHandler

        return OllamaHandler(
            model.id,
            num_ctx=model.num_ctx,
            max_tokens=model.max_tokens,
            think=model.think,
        )
    if model.type == "gemma":
        from .gemma import GemmaHandler

        return GemmaHandler(model.local_path or model.id)
    if model.type == "opencode_go":
        from .opencode_go_handler import OpenCodeGoHandler

        return OpenCodeGoHandler(
            model.id,
            max_tokens=model.max_tokens,
            request_timeout=model.request_timeout,
            max_retries=model.max_retries,
            retry_base_delay=model.retry_base_delay,
            reasoning_effort=model.reasoning_effort,
        )
    raise ValueError(f"Unsupported model type: {model.type}")
