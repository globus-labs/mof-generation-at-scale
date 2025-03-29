from ase.io.vasp import write_vasp
from pytest import raises
from ase import Atoms
import numpy as np
from six import StringIO

from mofa.scoring.geometry import MinimumDistance, LatticeParameterChange


def test_distance(example_record):
    # Make a 2x2x2 simple cubic cell
    atoms = Atoms(positions=[[0, 0, 0]], cell=[1., 1., 1.])
    atoms *= [2, 2, 2]

    assert MinimumDistance()(atoms) == 1.
    assert MinimumDistance().score_mof(example_record) > 0


def test_strain(example_record):
    scorer = LatticeParameterChange()

    # Ensure it throws an error
    with raises(ValueError):
        scorer.score_mof(example_record)

    # Make a fake MD trajectory with no change
    example_record.md_trajectory['uff'] = [(0, example_record.structure),
                                           (1000, example_record.structure)]
    assert np.isclose(scorer.score_mof(example_record), 0)

    # Make sure it compute stresses correctly if the volume shears
    final_atoms = example_record.atoms
    final_atoms.set_cell(final_atoms.cell.lengths().tolist() + [80, 90, 90])
    sio = StringIO()
    write_vasp(sio, final_atoms)
    example_record.md_trajectory['uff'][1] = (1000, sio.getvalue())

    max_strain = scorer.score_mof(example_record)
    assert np.isclose(max_strain, 0.09647)  # Checked against https://www.cryst.ehu.es/cryst/strain.html
