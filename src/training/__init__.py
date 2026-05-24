from src.training.messages import create_conversation

__all__ = ["create_conversation", "train_model"]


def __getattr__(name: str):
	if name == "train_model":
		from src.training.train import train_model

		return train_model
	raise AttributeError(f"module 'src.training' has no attribute '{name}'")

