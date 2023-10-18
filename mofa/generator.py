"""Functions pertaining to training and running the generative model"""
from pathlib import Path

import ase

from model import MOFRecord
from difflinker_sample import sample_from_sdf

def train_generator(
        starting_model: str | Path,
        examples: list[MOFRecord],
        num_epochs: int
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        starting_model: Path to the starting weights of the model
        examples: Seed examples of linkers (data model TBD)
        num_epochs: Number of training epochs
    Returns:
        Path to the new model weights
    """
    raise NotImplementedError()


def run_generator(
        node: str='CuCu', 
        n_atoms: int|str=8, 
        input_path: str|Path=f"mofa/data/fragments_all/CuCu/hMOF_frag_frag.sdf", 
        model: str|Path="mofa/models/geom_difflinker.ckpt",
        n_samples: int=1,
        n_steps: int=None
) -> list[ase.Atoms]:
    """
    Args:
        model: Path to the starting weights
        n_atoms: Number of heavy atoms in the linker molecules to generate
        n_samples: Number of samples of molecules to generate
    Returns:
        3D geometries of the generated linkers
    """
    assert node in input_path, "node must be in input_path name"
    sample_from_sdf(node=node, 
                    n_atoms=n_atoms, 
                    input_path=input_path, 
                    model=model,
                    n_samples=n_samples,
                    n_steps=n_steps
                    )   

if __name__ == "__main__":
    run_generator()
