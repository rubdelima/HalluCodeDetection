from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from src.constants.models import ModelInfo, get_models_options
import yaml
from pathlib import Path


class DatasetBuildingConfig(BaseModel):
    dataset_load : float = Field(1.0, description="Fraction of the dataset to load for generate results. Must be in (0, 1].")
    datasets_base : List[str] = Field(default_factory=lambda: ["mbpp"], description="List of datasets to use for training. Supported values: mbpp.")
    model_temperature : float = Field(1.0, description="Temperature to use for model generation when building the dataset.")
    judge_temperature : float = Field(0.0, description="Temperature to use for the judge model when generating explanations.")
    models: List[ModelInfo] = Field(...,  description="List of model configurations to use for training and evaluation.")
    tests_timeout : int = Field(30, description="Timeout for each test case in seconds.")
    checkpoint_interval : int = Field(100, description="Interval (in number of examples) to save intermediate results when building the dataset.")
    overwrite : bool = Field(False, description="Whether to overwrite existing dataset files when building the dataset.") 
    results_dir : str = Field("results", description="Directory to save the generated results when building the dataset.")
    judge_results_dir : str = Field("results/judge", description="Directory to save the judge results (explanations) from Phase 2.")
    judge_models : List[ModelInfo] = Field(default_factory=list, description="Model configurations to use for judging the generated code.")

    @classmethod
    def from_config_file(cls, path: str) -> "DatasetBuildingConfig":
        path_ = Path(path)
        if not path_.is_file():
            raise FileNotFoundError(f"Dataset building configuration file not found at {path}")
        with path_.open("r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
        
        return cls.from_config(config_data)
    
    @classmethod
    def from_config(cls, config: Dict, models_options: Optional[Dict[str, ModelInfo]] = None) -> "DatasetBuildingConfig":
        models_options = get_models_options(config) if models_options is None else models_options
        building = config.get("dataset_building", {})
        judge_models_raw = building.get("judge_models", [])
        if not judge_models_raw:
            single = building.get("judge_model", "")
            judge_models_raw = [single] if single else []
        judge_models: List[ModelInfo] = []
        for entry in judge_models_raw:
            if isinstance(entry, str):
                model_id, num_ctx, think = entry, None, None
            elif isinstance(entry, dict):
                model_id = entry.get("id")
                num_ctx = entry.get("num_ctx")
                think = entry.get("think")
            else:
                continue
            model = models_options.get(model_id)
            if model is None:
                continue
            judge_models.append(model.model_copy(update={"num_ctx": num_ctx, "think": think}))
        models_str = building.get("models", [])
        models = [model for model_id, model in models_options.items() if model_id in models_str]
        datasets_base = building.get("datasets_base", building.get("dataset_base", ["mbpp"]))

        return cls(
            dataset_load=building.get("dataset_load", 1.0),
            datasets_base=datasets_base,
            model_temperature=building.get("model_temperature", 1.0),
            judge_temperature=building.get("judge_temperature", 0.0),
            models=models,
            tests_timeout=building.get("tests_timeout", 30),
            checkpoint_interval=building.get("checkpoint_interval", 100),
            overwrite=building.get("overwrite", building.get("overwrite_results", False)),
            results_dir=building.get("results_dir", "results"),
            judge_results_dir=building.get("judge_results_dir", "results/judge"),
            judge_models=judge_models,
        )

class DatasetConfig(BaseModel):
    load_size : float = Field(1.0, description="Fraction of the dataset to load for training. Must be in (0, 1].")
    correct_size : float = Field(1.0, description="Fraction of the 'correct' examples to include in the training set. Must be in (0, 1].")
    random_seed : int = Field(42, description="Random seed for dataset shuffling and splitting.")
    validation_size : float = Field(0.1, description="Fraction of the dataset to use for validation. Must be in (0, 1).")
    test_size : float = Field(0.1, description="Fraction of the dataset to use for testing. Must be in (0, 1).")
    
    @classmethod
    def from_config(cls, config: dict) -> "DatasetConfig":
        return cls(
            load_size=config.get("dataset", {}).get("load_size", 1.0),
            correct_size=config.get("dataset", {}).get("correct_size", 1.0),
            random_seed=config.get("dataset", {}).get("random_seed", 42),
            validation_size=config.get("dataset", {}).get("validation_size", 0.1),
            test_size=config.get("dataset", {}).get("test_size", 0.1),
        )