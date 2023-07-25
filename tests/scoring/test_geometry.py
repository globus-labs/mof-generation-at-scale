from ase import Atoms

from mofa.scoring.geometry import get_closest_atomic_distance


def test_distance():
    # Make a 2x2x2 simple cubic cell
    atoms = Atoms(positions=[[0, 0, 0]], cell=[1., 1., 1.])
    atoms *= [2, 2, 2]

    assert get_closest_atomic_distance(atoms) == 1.
