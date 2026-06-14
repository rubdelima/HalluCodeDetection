from pydantic import Field

from src.constants.training import TrainingHyperparameters

class TrainingResult(TrainingHyperparameters):
    run_path: str = Field("", description="Temporary training output path for the run.")
    model_path: str | None = Field(None, description="Path where the merged model is saved, when selected among the best models.")
    saved_model: bool = Field(False, description="Whether this result was kept as a saved merged model.")
    training_time: float = Field(..., description="Total training time in seconds.")
    train_acc : float = Field(..., description="Training accuracy of the model.")
    val_acc : float = Field(..., description="Validation accuracy of the model.")
    test_acc : float = Field(..., description="Test accuracy of the model.")
    
