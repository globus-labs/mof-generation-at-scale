from pathlib import Path

from pytest import fixture, mark
import torch

from mofa.utils.src.lightning import DDPM
from mofa.utils.src.linker_size_lightning import SizeClassifier
from mofa.generator import train_generator, run_generator


@fixture
def file_dir():
    return Path(__file__).parent / 'files' / 'difflinker'


@fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@fixture()
def load_denoising_model(device, file_dir):
    model = file_dir / "geom_difflinker.ckpt"
    return DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)


@fixture()
def load_size_gnn_model(device, file_dir):
    model = file_dir / "geom_size_gnn.ckpt"
    return SizeClassifier.load_from_checkpoint(model, map_location=device).eval().to(device)


def test_load_model(load_denoising_model, load_size_gnn_model):
    assert load_denoising_model.__class__.__name__ == 'DDPM'
    assert load_size_gnn_model.__class__.__name__ == 'SizeClassifier'


@mark.slow
@mark.xfail
def test_training(file_dir):
    train_generator(
        starting_model=None,
        config_path=file_dir / 'config.yaml',
        examples=file_dir / 'hMOF_frag_table.csv',
        num_epochs=1
    )


@mark.parametrize('n_atoms', [3])
@mark.parametrize('node', ['CuCu'])
@mark.parametrize('n_samples', [1, 3])
def test_sampling_num_atoms(n_atoms, node, n_samples, file_dir, tmp_path):
    samples = run_generator(
        model=file_dir / 'geom_difflinker.ckpt',
        n_atoms=n_atoms,
        input_path=str(file_dir / "hMOF_frag_frag.sdf"),
        n_samples=n_samples
    )
    assert len(samples) == 4 * n_samples
