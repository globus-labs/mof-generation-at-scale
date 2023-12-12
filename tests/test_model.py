from math import isclose

from mofa.model import MOFRecord


def test_create(example_cif):
    mof = MOFRecord.from_file(example_cif, identifiers={'local': 'test'})
    assert mof.identifiers['local'] == 'test'
    assert isclose(mof.atoms.cell.lengths()[0], 39.87968858)


def test_name(example_cif):
    # Same CIF, same name
    mof_1 = MOFRecord.from_file(example_cif)
    mof_2 = MOFRecord.from_file(example_cif)
    assert mof_2.name == mof_1.name

    # No structure, random name
    mof_3 = MOFRecord()
    mof_4 = MOFRecord()
    assert mof_3.name != mof_4.name
