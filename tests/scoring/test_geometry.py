from ase import Atoms

from mofa.scoring.geometry import MinimumDistance


def test_distance(example_record):
    # Make a 2x2x2 simple cubic cell
    atoms = Atoms(positions=[[0, 0, 0]], cell=[1., 1., 1.])
    atoms *= [2, 2, 2]

    assert MinimumDistance()(atoms) == 1.
    assert MinimumDistance().score_mof(example_record) > 0
