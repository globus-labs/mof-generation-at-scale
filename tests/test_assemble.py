from pathlib import Path

from rdkit import Chem
from pytest import mark
from ase.io import read
from six import StringIO

from mofa.assemble import assemble_pillaredPaddleWheel_pcuMOF
from mofa.preprocess_linkers import clean_linker

_files_dir = Path(__file__).parent / 'files' / 'assemble'


@mark.parametrize('smiles', ['C=Cc1ccccc1C=C'])
def test_prepare_linker(smiles):
    mol = Chem.MolFromSmiles(smiles)
    linkers = clean_linker(mol)
    assert len(linkers) == 3


def test_paddlewheel_pcu():
    # Find a set of linkers by pulling from different folders
    chosen_folders = list(_files_dir.glob('linkers/molGAN-*'))[:3]
    coo_linkers = [f
                   for d in chosen_folders[:2]
                   for f in d.glob("linker-COO*.xyz")]
    pillar_linker = next(f for f in chosen_folders[2].glob("*.xyz") if 'COO' not in f.name)
    node = _files_dir / 'nodes/zinc_paddle_pillar.xyz'

    cif = assemble_pillaredPaddleWheel_pcuMOF(node, coo_linkers, pillar_linker)
    atoms = read(StringIO(cif), format='cif')
    assert len(atoms) > 0
