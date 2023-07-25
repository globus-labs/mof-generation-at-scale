"""Functions pertaining to training and running the generative model"""
from pathlib import Path


def train_generator(
        starting_model: str | Path,
        examples: list[str],
        num_epochs: int
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        starting_model: Path to the starting weights of the model
        examples: Seed examples of linkers used to train generator
        num_epochs: Number of training epochs
    Returns:
        Path to the new model weights
    """
    raise NotImplementedError()
