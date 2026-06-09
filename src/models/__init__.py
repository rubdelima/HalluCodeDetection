from .base import BaseModelHandler
from .ollama_handler import OllamaHandler
from .gemma import GemmaHandler

from src.constants.models import ModelInfo, get_models_options

def get_model_handler(model: ModelInfo) -> BaseModelHandler:
    if model.type == "ollama":
        return OllamaHandler(model.id)
    elif model.type == "gemma":
        return GemmaHandler(model.id)
    raise ValueError(f"Unsupported model type: {model.type}")