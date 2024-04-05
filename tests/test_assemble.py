from pathlib import Path
from io import StringIO

from pytest import mark
from ase.io import read

from mofa.assembly.assemble import assemble_pillaredPaddleWheel_pcuMOF, assemble_mof
from mofa.model import NodeDescription, LigandDescription

_files_dir = Path(__file__).parent / 'files' / 'assemble'


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


@mark.parametrize('node_name,topology,ligand_counts', [
    ('zinc_paddle_pillar', 'pcu', {
        'COO': 2,
        'cyano': 1
    })
])
def test_assemble(node_name, topology, ligand_counts, file_path):
    """Test the full integration

    Args:
        node_name: Name of the node, which should map to an XYZ in the `files/assemble/nodes` directory
        topology: Name of the topology
        ligand_counts: Map of the name of a ligand description in `files/difflinker/templates/` to number required
    """

    # Format the node and linker descriptions
    node_path = _files_dir / f'nodes/{node_name}.xyz'
    node = NodeDescription(
        smiles='NA',
        xyz=node_path.read_text()
    )

    # Load the ligand examples
    ligands = {}
    for name, count in ligand_counts.items():
        ligand_desc = LigandDescription.from_yaml(file_path / 'difflinker' / 'templates' / f'description_{name}.yml')
        ligands[name] = [ligand_desc] * count

    # Run the assembly code
    mof_record = assemble_mof([node], ligands, topology)
    assert len(mof_record.atoms) > 0  # Make sure a full MOF is available
    for ligand in mof_record.ligands:
        assert ligand.xyz is not None
    assert mof_record.name is not None
