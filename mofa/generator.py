"""Functions pertaining to training and running the generative model"""
from pathlib import Path

import ase

from mofa.model import MOFRecord
from mofa.difflinker_sample import sample_from_sdf
from mofa.difflinker_train import get_args, main
import yaml
import os

def train_generator(
        # starting_model: str | Path,
        examples: str|list[MOFRecord]="../argonne_gnn_gitlab/DiffLinker/data/geom/datasets",
        num_epochs: int=10
) -> Path:
    """Retrain a generative model for MOFs

    Args:
        # starting_model: Path to the starting weights of the model
        examples: Seed examples of linkers (data model TBD)
        num_epochs: Number of training epochs
    Returns:
        Path to the new model weights
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
    args = get_args()
        
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list) and key != 'normalize_factors':
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
            
    args.n_epochs = num_epochs
    main(args=args)

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
        node: Which node to use: ['CuCu','ZnZn','ZnOZnZnZn'] to reproduce paper results 
        n_atoms: Number of heavy atoms in the linker molecules to generate
        input_path: Path to MOF linker fragment containing SDF file
        model: Path to the starting weights
        n_samples: Number of samples of molecules to generate
        n_steps: Number of denoising steps; if None, this value is 1,000 by default
        
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
    print("Saved XYZ files in mofa/output directory!")

if __name__ == "__main__":
    # run_generator()
    train_generator()
        
