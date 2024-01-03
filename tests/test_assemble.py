from pathlib import Path
from io import StringIO

from rdkit import Chem
from pytest import mark
from ase.io import read

from mofa.assembly.assemble import assemble_pillaredPaddleWheel_pcuMOF, assemble_mof
from mofa.model import NodeDescription, LigandDescription
from mofa.assembly.preprocess_linkers import clean_linker

_files_dir = Path(__file__).parent / 'files' / 'assemble'


@mark.parametrize('smiles', ['C=Cc1ccccc1C=C'])
def test_prepare_linker(smiles):
    mol = Chem.MolFromSmiles(smiles)
    linkers = clean_linker(mol)
    assert len(linkers) == 3
    for xyz in linkers.values():
        read(StringIO(xyz), format='xyz')  # Make sure it parses as an XYZ


def test_paddlewheel_pcu():
    # Find a set of linkers by pulling from different folders
    chosen_folders = list(_files_dir.glob('linkers/molGAN-*'))[:3]
    coo_linkers = [f
                   for d in chosen_folders[:2]
                   for f in d.glob("linker-COO*.xyz")]
    pillar_linker = next(f for f in chosen_folders[2].glob("*.xyz") if 'COO' not in f.name)
    node = _files_dir / 'nodes/zinc_paddle_pillar.xyz'

    vasp = assemble_pillaredPaddleWheel_pcuMOF(node, coo_linkers, pillar_linker)
    atoms = read(StringIO(vasp), format='vasp')
    assert len(atoms) > 0


@mark.parametrize('node_name,topology,ligand_count', [('zinc_paddle_pillar', 'pcu', 3)])
def test_assemble(node_name, topology, ligand_count, linker_smiles='C=Cc1ccccc1C=C'):
    """Test the full integration"""

    # Format the node and linker descriptions
    node_path = _files_dir / f'nodes/{node_name}.xyz'
    node = NodeDescription(
        smiles='NA',
        xyz=node_path.read_text()
    )

    ligands = [LigandDescription(smiles=linker_smiles) for _ in range(ligand_count)]

    # Run the assembly code
    mof_record = assemble_mof([node], ligands, topology)
    assert len(mof_record.atoms) > 0  # Make sure a full MOF is available
    for ligand in mof_record.ligands:
        assert ligand.xyz is not None
    assert mof_record.name is not None
