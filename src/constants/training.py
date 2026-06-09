from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Tuple

from src.constants.models import ModelInfo, get_models_options
from itertools import product

BIAS_OPTIONS = Literal["none", "all", "lora_only"]

class TrainingHyperparameters(BaseModel):
    model_name: ModelInfo = Field(..., description="Model configuration to use for training.")
    use_qlora: bool = Field(True, description="Whether to use 4-bit QLoRA for training (if false, uses full precision LoRA).")
    lora_r: int = Field(8, description="LoRA rank (r) hyperparameter for training.")
    lora_alpha: int = Field(16, description="LoRA alpha hyperparameter for training.")
    lora_dropout: float = Field(0.05, description="LoRA dropout probability for training.")
    learning_rate: float = Field(5e-5, description="Learning rate for training.")
    num_epochs : int = Field(..., description="Number of epochs to train for.")
    bias : BIAS_OPTIONS = Field("none", description="Which parameters to apply LoRA to. Options: 'none' (only weight matrices), 'all' (all parameters), 'lora_only' (only parameters in the LoRA adapter).")
    optimizer: str = Field("adamw_torch_fused", description="Optimizer to use for training. Options: 'adamw_torch_fused', 'adamw_torch', 'adamw_apex_fused', 'adamw_apex', 'adamw_hf_fused', 'adamw_hf', 'c' (for 4-bit QLoRA).")
    
class TrainingConfig(BaseModel):
    hyperparameters : List[TrainingHyperparameters] = Field(..., description="List of training hyperparameter configurations to use for training")
    models_path : str = Field("models", description="Path to save trained models.")
    max_saved_models : int = Field(5, description="Maximum number of trained models to save.")
    
    @classmethod
    def from_config(cls, config, models_options: Optional[Dict[str, ModelInfo]] = None) -> "TrainingConfig":
        models_options = get_models_options(config) if models_options is not None else {}
        models_id = config.get("training", {}).get("models", [])
        models = [model for model_id, model in models_options.items() if model_id in models_id]
        use_qlora_options = config.get("training", {}).get("use_qlora", [True])
        lora_r_options = config.get("training", {}).get("lora_r", [8])
        lora_alpha_options = config.get("training", {}).get("lora_alpha", [16])
        lora_dropout_options = config.get("training", {}).get("lora_dropout", [0.05])
        learning_rate_options = config.get("training", {}).get("learning_rate", [5e-5])
        num_epochs_options = config.get("training", {}).get("num_epochs", [3])
        bias_options = config.get("training", {}).get("bias", ["none"])
        optimizer_options = config.get("training", {}).get("optimizer", ["adamw_torch_fused"])
        
        hyperparameters = []
        
        for model, use_qlora, lora_r, lora_alpha, lora, learning_rate, num_epochs, bias, optimizer in product(
            models, use_qlora_options, lora_r_options, lora_alpha_options, lora_dropout_options, learning_rate_options, num_epochs_options, bias_options, optimizer_options
        ):
            hyperparameters.append(TrainingHyperparameters(
                model_name=model,
                use_qlora=use_qlora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                bias=bias,
                optimizer=optimizer,
            ))
        
        return cls(
            hyperparameters=hyperparameters,
            models_path=config.get("training", {}).get("models_path", "models"),
            max_saved_models=config.get("training", {}).get("max_saved_models", 5),
        )

class TrainingResult(TrainingHyperparameters):
    model_path: str = Field(..., description="Path where the trained model is saved.")
    train_acc : float = Field(..., description="Training accuracy of the model.")
    val_acc : float = Field(..., description="Validation accuracy of the model.")
    train_time : float = Field(..., description="Total training time in seconds.")
    