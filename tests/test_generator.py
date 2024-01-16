from pathlib import Path

from pytest import fixture, mark
import torch

from mofa.model import LigandTemplate
from mofa.utils.src.lightning import DDPM
from mofa.utils.src.linker_size_lightning import SizeClassifier
from mofa.generator import train_generator, run_generator


@fixture
def file_dir(file_path):
    return file_path / 'difflinker'


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


@fixture()
def example_template(file_dir):
    return LigandTemplate.from_yaml(file_dir / 'templates' / 'template_COO.yml')


def test_load_model(load_denoising_model, load_size_gnn_model):
    assert load_denoising_model.__class__.__name__ == 'DDPM'
    assert load_size_gnn_model.__class__.__name__ == 'SizeClassifier'


def test_training(file_dir, tmpdir):
    new_model = train_generator(
        starting_model=None,
        run_directory=Path(tmpdir),
        config_path=file_dir / 'config.yaml',
        examples=file_dir / 'datasets/fragments_all/CuCu',
        num_epochs=1
    )
    assert new_model.is_file()


@mark.parametrize('n_atoms', [8])
@mark.parametrize('n_samples', [1, 3])
def test_sampling_num_atoms(n_atoms, example_template, n_samples, file_dir, tmp_path):
    samples = run_generator(
        model=file_dir / 'geom_difflinker.ckpt',
        templates=[example_template],
        n_atoms=n_atoms,
        n_samples=n_samples,
        n_steps=64,
    )

    # Make sure we created molecules of the correct size
    assert len(samples) == n_samples
    anchor_count = sum(len(a) for a in example_template.anchors)
    for sample in samples:
        assert len(sample.atoms) == anchor_count + n_atoms
