from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal

from src.constants.models import ModelInfo, get_models_options
from itertools import product

BIAS_OPTIONS = Literal["none", "all", "lora_only"]
SEARCH_STRATEGY = Literal["optuna"]
OPTUNA_TARGET = Literal["accuracy", "macro_f1", "macro_recall"]
CLASSIFICATION_LOSS_STRATEGY = Literal["standard", "label_weighted"]
ClassBalanceRatio = tuple[int, int, int]

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
    classification_loss_strategy: CLASSIFICATION_LOSS_STRATEGY = Field("standard", description="Loss strategy used for this run.")
    classification_loss_weight: float = Field(3.0, ge=1.0, description="Class-token weight for label_weighted loss.")
    class_balance_ratio: ClassBalanceRatio = Field(
        (1, 1, 1),
        description="Target runtime:functional:correct ratio used only to sample the training split.",
    )
    
class TrainingConfig(BaseModel):
    hyperparameters : List[TrainingHyperparameters] = Field(..., description="List of training hyperparameter configurations to use for training")
    models_path : str = Field("models", description="Path to save trained models.")
    checkpoints_path: str | None = Field(None, description="Path for temporary Trainer checkpoints; defaults to results_dir/training_runs.")
    max_saved_models : int = Field(5, description="Maximum number of trained models to save.")
    search_strategy: SEARCH_STRATEGY = Field("optuna", description="Persistent Optuna TPE search strategy.")
    random_seed: int = Field(42, description="Seed used by the Optuna sampler.")
    max_trials: int = Field(40, gt=0, description="Maximum number of completed or failed Optuna trials.")
    optuna_startup_trials: int = Field(10, ge=0, description="Random Optuna trials before TPE sampling starts.")
    optuna_study_name: str = Field("hallucination_detection", min_length=1, description="Persistent Optuna study name.")
    optuna_target: OPTUNA_TARGET = Field("accuracy", description="Validation metric optimized by Optuna.")
    classification_loss_strategy: CLASSIFICATION_LOSS_STRATEGY = Field(
        "standard",
        description="Whether to use normal SFT loss or upweight class-label tokens.",
    )
    classification_loss_weight: float = Field(
        3.0,
        ge=1.0,
        description="Weight applied to class-label tokens when classification_loss_strategy is label_weighted.",
    )
    class_balance_ratios: List[ClassBalanceRatio] = Field(
        default_factory=lambda: [(1, 1, 1)],
        description="Candidate runtime:functional:correct ratios for syntax-free training.",
    )
    exclude_syntax_from_training: bool = Field(
        False,
        description="Exclude records that contain the syntax label from the training split.",
    )
    class_balance_enabled: bool = Field(
        False,
        description="Apply class_balance_ratios to the training split; validation and test remain unchanged.",
    )
    results_dir: str | None = Field(
        None,
        description="Directory for training JSONL and Optuna storage; defaults to dataset-building results_dir.",
    )
    
    @classmethod
    def from_config(cls, config, models_options: Optional[Dict[str, ModelInfo]] = None) -> "TrainingConfig":
        models_options = get_models_options(config) if models_options is None else models_options
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
        class_balance_ratios = config.get("training", {}).get("class_balance_ratios", [[1, 1, 1]])
        normalized_ratios: list[ClassBalanceRatio] = []
        for ratio in class_balance_ratios:
            if not isinstance(ratio, (list, tuple)) or len(ratio) != 3:
                raise ValueError("training.class_balance_ratios entries must be [runtime, functional, correct].")
            normalized = tuple(int(value) for value in ratio)
            if any(value <= 0 for value in normalized):
                raise ValueError("training.class_balance_ratios values must be positive integers.")
            normalized_ratios.append(normalized)
        
        hyperparameters = []
        
        for model, use_qlora, lora_r, lora_alpha, lora, learning_rate, num_epochs, bias, optimizer, class_balance_ratio in product(
            models, use_qlora_options, lora_r_options, lora_alpha_options, lora_dropout_options, learning_rate_options, num_epochs_options, bias_options, optimizer_options, normalized_ratios
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
                classification_loss_strategy=config.get("training", {}).get("classification_loss_strategy", "standard"),
                classification_loss_weight=config.get("training", {}).get("classification_loss_weight", 3.0),
                class_balance_ratio=class_balance_ratio,
            ))
        
        return cls(
            hyperparameters=hyperparameters,
            models_path=config.get("training", {}).get(
                "models_path",
                config.get("training", {}).get("model_output_dir", "models"),
            ),
            checkpoints_path=config.get("training", {}).get("checkpoints_path"),
            max_saved_models=config.get("training", {}).get(
                "max_saved_models",
                config.get("training", {}).get("max_salved_models", 5),
            ),
            search_strategy=config.get("training", {}).get("search_strategy", "optuna"),
            random_seed=config.get("training", {}).get("random_seed", 42),
            max_trials=config.get("training", {}).get("max_trials", 40),
            optuna_startup_trials=config.get("training", {}).get("optuna_startup_trials", 10),
            optuna_study_name=config.get("training", {}).get("optuna_study_name", "hallucination_detection"),
            optuna_target=config.get("training", {}).get("optuna_target", "accuracy"),
            classification_loss_strategy=config.get("training", {}).get("classification_loss_strategy", "standard"),
            classification_loss_weight=config.get("training", {}).get("classification_loss_weight", 3.0),
            class_balance_ratios=normalized_ratios,
            exclude_syntax_from_training=config.get("training", {}).get("exclude_syntax_from_training", False),
            class_balance_enabled=config.get("training", {}).get("class_balance_enabled", False),
            results_dir=config.get("training", {}).get("results_dir"),
        )

class TrainingResult(TrainingHyperparameters):
    model_path: str = Field(..., description="Path where the trained model is saved.")
    train_acc : float = Field(..., description="Training accuracy of the model.")
    val_acc : float = Field(..., description="Validation accuracy of the model.")
    train_time : float = Field(..., description="Total training time in seconds.")
    
