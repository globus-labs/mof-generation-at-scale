from textwrap import dedent


def water(last_atom_ID, last_bond_ID, last_angle_ID, model='TIP4P_cutoff'):

    ID_O = last_atom_ID + 1
    ID_H = ID_O + 1

    BT = last_bond_ID + 1
    AT = last_angle_ID + 1

    charge_dict = {
        'TIP4P_cutoff': (-1.0400, 0.5200),
        'TIP4P_2005': (-1.1128, 0.5564),
        'TIP4P_long': (-1.0484, 0.5242),
        'TIP3P_long': (-0.8300, 0.4150)
    }

    M_site_dist_dict = {
        'TIP4P_cutoff': 0.1500,
        'TIP4P_2005': 0.1546,
        'TIP4P_long': 0.1250,
        'TIP3P_long': None
    }

    LJ_dict = {
        # LAMMPS has a special TIP4P pair_style that automatically adds the M
        # site
        'TIP4P_cutoff': {ID_O: ('lj/cut/tip4p/cut', 0.15500, 3.15360),
                         ID_H: ('lj/cut/tip4p/cut', 0.0, 0.0),
                         'style': 'lj/cut/tip4p/cut',
                         'comments': {ID_O: ['O_water', 'O_water'],
                                      ID_H: ['H_water', 'H_water']}},
        'TIP4P_2005': {ID_O: ('lj/cut/tip4p/long', 0.18520, 3.15890),
                       ID_H: ('lj/cut/tip4p/long', 0.0, 0.0),
                       'style': 'lj/cut/tip4p/long',
                       'comments': {ID_O: ['O_water', 'O_water'],
                                    ID_H: ['H_water', 'H_water']}},
        'TIP4P_long': {ID_O: ('lj/cut/tip4p/long', 0.16275, 3.16435),
                       ID_H: ('lj/cut/tip4p/long', 0.0, 0.0),
                       'style': 'lj/cut/tip4p/long',
                       'comments': {ID_O: ['O_water', 'O_water'],
                                    ID_H: ['H_water', 'H_water']}},
        'TIP3P_long': {ID_O: ('lj/cut/coul/long', 0.10200, 3.18800),
                       ID_H: ('lj/cut/coul/long', 0.0, 0.0),
                       'style': 'lj/cut/coul/long',
                       'comments': {ID_O: ['O_water', 'O_water'],
                                    ID_H: ['H_water', 'H_water']}}
    }

    bond_dict = {
        # TIP4P is a rigid model (use fix shake), force constants should just be reasonable values
        # TIP3P has force constants if a flexible model is desired
        'TIP4P_cutoff': {BT: {'style': 'harmonic', 'params': (100.0, 0.9572), 'comments': '  # O_water H_water'}},
        'TIP4P_2005': {BT: {'style': 'harmonic', 'params': (100.0, 0.9572), 'comments': '  # O_water H_water'}},
        'TIP4P_long': {BT: {'style': 'harmonic', 'params': (100.0, 0.9572), 'comments': '  # O_water H_water'}},
        'TIP3P_long': {BT: {'style': 'harmonic', 'params': (450.0, 0.9572), 'comments': '  # O_water H_water'}}
    }

    angle_dict = {
        # TIP4P is a rigid model (use fix shake), force constants should just be reasonable values
        # TIP3P has force constants if a flexible model is desired
        'TIP4P_cutoff': {AT: {'style': 'harmonic', 'params': (50.0, 104.52), 'comments': '  # H_water O_water H_water'}},
        'TIP4P_2005': {AT: {'style': 'harmonic', 'params': (50.0, 104.52), 'comments': '  # H_water O_water H_water'}},
        'TIP4P_long': {AT: {'style': 'harmonic', 'params': (50.0, 104.52), 'comments': '  # H_water O_water H_water'}},
        'TIP3P_long': {AT: {'style': 'harmonic', 'params': (55.0, 104.52), 'comments': '  # H_water O_water H_water'}}
    }

    qO, qH = charge_dict[model]
    LJ_params = LJ_dict[model]
    bond_params = bond_dict[model]
    angle_params = angle_dict[model]

    if 'TIP4P' in model:

        molfile = dedent(
            """  # Water molecule. useable for TIP3P or TIP4P in LAMMPS.

3 atoms
2 bonds
1 angles

Coords

1    1.12456   0.09298   1.27452
2    1.53683   0.75606   1.89928
3    0.49482   0.56390   0.65678

Types

1    {ID_O}
2    {ID_H}
3    {ID_H}

Charges

1    {qO}
2    {qH}
3    {qH}

Bonds

1    {BT} 1 2
2    {BT} 1 3

Angles

1    {AT} 2 1 3

Shake Flags

1 1
2 1
3 1

Shake Atoms

1 1 2 3
2 1 2 3
3 1 2 3

Shake Bond Types

1 {BT} {BT} {AT}
2 {BT} {BT} {AT}
3 {BT} {BT} {AT}""".format(
                **locals())).strip()

    if 'TIP3P' in model:

        molfile = dedent(
            """  # Water molecule. useable for TIP3P or TIP4P in LAMMPS.

3 atoms
2 bonds
1 angles

Coords

1    1.12456   0.09298   1.27452
2    1.53683   0.75606   1.89928
3    0.49482   0.56390   0.65678

Types

1    {ID_O}
2    {ID_H}
3    {ID_H}

Charges

1    {qO}
2    {qH}
3    {qH}

Bonds

1    {BT} 1 2
2    {BT} 1 3

Angles

1    {AT} 2 1 3""".format(
                **locals())).strip()

    mass_dict = {ID_O: 15.9994, ID_H: 1.00794}
    molnames = ('H2O_mol', 'H2O.txt')

    extra_types = (2, 1, 1, None, None)

    return molfile, LJ_params, bond_params, angle_params, molnames, mass_dict, M_site_dist_dict[
        model], extra_types
