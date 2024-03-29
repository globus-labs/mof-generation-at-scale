from pathlib import Path

from pytest import fixture, mark

from mofa.model import LigandTemplate
from mofa.utils.src.lightning import DDPM
from mofa.utils.src.linker_size_lightning import SizeClassifier
from mofa.generator import train_generator, run_generator


@fixture
def file_dir(file_path):
    return file_path / 'difflinker'


@fixture
def device():
    return "cpu"


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


@mark.parametrize('filename', ['geom_difflinker.ckpt', 'geom_difflinker_given_anchors.ckpt'])
@mark.parametrize('n_atoms', [8])
@mark.parametrize('n_samples', [1, 3])
def test_sampling_num_atoms(n_atoms, example_template, n_samples, file_dir, filename, tmp_path):
    # prompts = "1,5" if "prompts" in filename else None ####Give 1st and 5th atom of FRAGMENT-only molecule
    # (i.e., we can keep track of atom indices in fragment SDF/XYZ etc files)
    samples = list(run_generator(
        model=file_dir / filename,
        templates=[example_template],
        n_atoms=n_atoms,
        n_samples=n_samples,
        n_steps=64,
    ))

    # Make sure we created molecules of the correct size
    assert len(samples) == n_samples
    anchor_count = sum(len(a) for a in example_template.prompts)
    for sample in samples:
        assert len(sample.atoms) > anchor_count + n_atoms  # There will be more atoms once H's are added
