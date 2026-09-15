from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.constants.models import ModelInfo, get_models_options


class Phase7Config(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
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
            model_temperature=section.get("model_temperature", 0.0),
            checkpoint_interval=section.get("checkpoint_interval", 1),
            overwrite=section.get("overwrite", False),
            results_dir=section.get("results_dir"),
        )
