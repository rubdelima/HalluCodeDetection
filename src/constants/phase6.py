from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.constants.models import ModelInfo, get_models_options


class Phase6Config(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
    num_ctx: int | None = Field(None, gt=0, description="Ollama context window used by Phase 6.")
    max_tokens: int | None = Field(None, gt=0, description="Maximum generated tokens per Ollama call in Phase 6.")
    interaction_mode: str = Field(
        "direct",
        pattern="^(direct|agentic)$",
        description="How generation and review agents interact in Phase 6.",
    )
    max_rounds: int = Field(5, ge=1, le=20)
    generation_temperature: float = 0.0
    review_temperature: float = 0.0
    dataset_load: float = Field(1.0, gt=0.0, le=1.0)
    random_seed: int = 42
    checkpoint_interval: int = Field(1, ge=1)
    overwrite: bool = False
    results_dir: str | None = None
    phase1_results_path: str = (
        "./data/results/evalplus/phase1_opencode_baselines/dataset_base.json"
    )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Phase6Config:
        section = config.get("phase6", {})
        options = get_models_options(config)
        model_ids = section.get("models", [])
        return cls(
            models=[options[model_id] for model_id in model_ids if model_id in options],
            num_ctx=section.get("num_ctx"),
            max_tokens=section.get("max_tokens"),
            interaction_mode=section.get("interaction_mode", "direct"),
            max_rounds=section.get("max_rounds", 5),
            generation_temperature=section.get("generation_temperature", 0.0),
            review_temperature=section.get("review_temperature", 0.0),
            dataset_load=section.get("dataset_load", 1.0),
            random_seed=section.get("random_seed", 42),
            checkpoint_interval=section.get("checkpoint_interval", 1),
            overwrite=section.get("overwrite", False),
            results_dir=section.get("results_dir"),
            phase1_results_path=section.get(
                "phase1_results_path",
                "./data/results/evalplus/phase1_opencode_baselines/dataset_base.json",
            ),
        )
