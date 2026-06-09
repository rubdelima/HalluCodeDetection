import yaml
from pathlib import Path
from dataclasses import dataclass

from src.constants.dataset import DatasetBuildingConfig, DatasetConfig
from src.constants.evaluation import EvaluationConfig
from src.constants.models import ModelInfo
from src.constants.training import TrainingConfig
from src.constants.ui import UIConfig


class HalluCodeDetectionConfig:
    def __init__(self, config_path: str):
        config_path_ = Path(config_path)
        if not config_path_.is_file():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with config_path_.open("r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
        
        self.dataset_building_config = DatasetBuildingConfig.from_config(config_data)
        self.dataset_config = DatasetConfig.from_config(config_data)
        self.evaluation_config = EvaluationConfig.from_config(config_data)
        self.training_config = TrainingConfig.from_config(config_data)
        self.ui_config = UIConfig.from_config(config_data)
        