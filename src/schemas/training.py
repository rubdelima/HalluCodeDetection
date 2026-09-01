from pydantic import Field

from src.constants.training import TrainingHyperparameters

class TrainingResult(TrainingHyperparameters):
    run_path: str = Field("", description="Temporary training output path for the run.")
    model_path: str | None = Field(None, description="Path where the merged model is saved, when selected among the best models.")
    saved_model: bool = Field(False, description="Whether this result was kept as a saved merged model.")
    training_time: float = Field(..., description="Total training time in seconds.")
    train_acc : float | None = Field(None, description="Training accuracy, when explicitly evaluated.")
    val_acc : float = Field(..., description="Validation accuracy of the model.")
    val_macro_f1: float | None = Field(None, description="Validation macro F1 over the four hallucination labels.")
    val_macro_recall: float | None = Field(None, description="Validation macro recall over the four hallucination labels.")
    test_acc : float | None = Field(None, description="Test accuracy, when explicitly evaluated.")
    
