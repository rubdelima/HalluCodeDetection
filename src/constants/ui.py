from pydantic import BaseModel, Field
from pathlib import Path
import yaml

class UIConfig(BaseModel):
    show_progress_bar : bool = Field(True, description="Whether to show a progress bar during dataset loading and training.")
    spinner_length : int = Field(10, description="Number of characters in the spinner animation.")
    
    @classmethod
    def from_config(cls, config:dict) -> "UIConfig":
        return cls(**config.get("ui", {}))
    
    @classmethod
    def from_config_file(cls, path: Path) -> "UIConfig":
        if not path.is_file():
            raise FileNotFoundError(f"UI configuration file not found at {path}")
        path = path.resolve()
        
        with path.open("r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
        
        return cls.from_config(config_data)