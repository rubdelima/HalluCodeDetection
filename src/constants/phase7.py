from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.constants.models import ModelInfo, get_models_options


class Phase7Config(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
    num_ctx: int | None = Field(None, gt=0, description="Ollama context window used by Phase 7.")
    max_tokens: int | None = Field(None, gt=0, description="Maximum generated tokens per Ollama call in Phase 7.")
    model_temperature: float = 0.0
    checkpoint_interval: int = Field(1, ge=1)
    overwrite: bool = False
    results_dir: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Phase7Config:
        section = config.get("phase7", {})
        options = get_models_options(config)
        model_ids = section.get("models", [])
        return cls(
            models=[options[model_id] for model_id in model_ids if model_id in options],
            num_ctx=section.get("num_ctx"),
            max_tokens=section.get("max_tokens"),
            model_temperature=section.get("model_temperature", 0.0),
            checkpoint_interval=section.get("checkpoint_interval", 1),
            overwrite=section.get("overwrite", False),
            results_dir=section.get("results_dir"),
        )
