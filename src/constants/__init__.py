import yaml
from pathlib import Path
from dataclasses import dataclass

from src.constants.dataset import DatasetBuildingConfig, DatasetConfig
from src.constants.evaluation import EvaluationConfig
from src.constants.models import ModelInfo
from src.constants.training import TrainingConfig
from src.constants.ui import UIConfig


def _merge_config(base: dict, override: dict) -> dict:
    """Recursively merge a derived YAML configuration over its base."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}
    base_config = config_data.pop("extends", None)
    if base_config is None:
        return config_data

    base_path = (config_path.parent / base_config).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base configuration file not found at {base_path}")
    return _merge_config(_load_config(base_path), config_data)


class HalluCodeDetectionConfig:
    def __init__(self, config_path: str):
        config_path_ = Path(config_path)
        if not config_path_.is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        config_data = _load_config(config_path_)
        
        self.dataset_building_config = DatasetBuildingConfig.from_config(config_data)
        self.dataset_config = DatasetConfig.from_config(config_data)
        self.evaluation_config = EvaluationConfig.from_config(config_data)
        self.training_config = TrainingConfig.from_config(config_data)
        self.ui_config = UIConfig.from_config(config_data)
