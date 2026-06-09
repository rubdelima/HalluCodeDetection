from src.constants.training import TrainingHyperparameters, Field

class TrainingResult(TrainingHyperparameters):
    training_time: float = Field(..., description="Total training time in seconds.")
    train_acc : float = Field(..., description="Training accuracy of the model.")
    val_acc : float = Field(..., description="Validation accuracy of the model.")
    test_acc : float = Field(..., description="Test accuracy of the model.")
    