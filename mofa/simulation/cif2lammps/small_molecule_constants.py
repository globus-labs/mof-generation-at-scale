TraPPE = {
    'O2': {
        'pair': {'style': 'lj/cut/coul/long', 'vdW': {'O_O2': (0.0974, 3.02), 'O_com': (0.0, 0.0)}, 'charges': {'O_O2': -0.113, 'O_com': 0.226}},
        # molecule should be kept rigid, force constants don't matter
        'bonds': {('O_O2', 'O_com'): ('harmonic', 100.0, 0.604)},
        # molecule should be kept rigid, force constants don't matter
        'angles': {('O_O2', 'O_com', 'O_O2'): ('harmonic', 100.0, 180.0)},
        'dihedrals': None,
        'impropers': None
    },
    'N2': {
        'pair': {},
        'bonds': {},
        'angles': {},
        'dihedrals': None,
        'impropers': None
    },
    'H2O1': {
        'pair': {},
        'bonds': {},
        'angles': {},
        'dihedrals': None,
        'impropers': None
    }
}

TIP4P_2005_long = {
    # this is TIP4P/2005 water, should be used with long-range electrostatics with 8.5 Å cutoff and fix/shake
    # keep in mind that using any long pair_style in lammps will include long-range electrostatics FOR ALL ATOMS in the simulation
    'H2O1': {
        'pair': {'style': 'lj/cut/tip4p/long', 'vdW': {'H_w': (0.0, 1.0), 'O_w': (0.1852, 3.1589)}, 'charges': {'H_w': 0.5564, 'O_w': -1.1128}},
        'bonds': {('H_w', 'O_w'): ('harmonic', 450.0, 0.9572)},
        'angles': {('H_w', 'O_w', 'H_w'): ('harmonic', 55.0, 104.52)},
        'dihedrals': None,
        'impropers': None
    },
    'Cl1': {
        'pair': {'style': 'lj/cut/tip4p/long', 'vdW': {'Cl_Cl1': (0.22700, 3.51638)}, 'charges': {'Cl_Cl1': -1.0}},
        'bonds': None,
        'angles': None,
        'dihedrals': None,
        'impropers': None
    }
}

TIP4P_2005_cutoff = {
    # this is TIP4P/2005 water but with no long range electrostatics
    'H2O1': {
        'pair': {'style': 'lj/cut/tip4p/cut', 'vdW': {'H_w': (0.0, 0.0), 'O_w': (0.1852, 3.1589)}, 'charges': {'H_w': 0.5564, 'O_w': -1.1128}},
        'bonds': {('H_w', 'O_w'): ('harmonic', 450.0, 0.9572)},
        'angles': {('H_w', 'O_w', 'H_w'): ('harmonic', 55.0, 104.52)},
        'dihedrals': None,
        'impropers': None
    },
    'Cl1': {
        'pair': {'style': 'lj/cut/tip4p/long', 'vdW': {'Cl_Cl1': (0.22700, 3.51638)}, 'charges': {'Cl_Cl1': -1.0}},
        'bonds': None,
        'angles': None,
        'dihedrals': None,
        'impropers': None
    }
}

TIP4P_cutoff = {
    # this is the original TIP4P water model
    'H2O1': {
        'pair': {'style': 'lj/cut/tip4p/cut', 'vdW': {'H_w': (0.0, 0.0), 'O_w': (0.1550, 3.1536)}, 'charges': {'H_w': 0.5200, 'O_w': -1.040}},
        'bonds': {('H_w', 'O_w'): ('harmonic', 450.0, 0.9572)},
        'angles': {('H_w', 'O_w', 'H_w'): ('harmonic', 55.0, 104.52)},
        'dihedrals': None,
        'impropers': None
    },
    'Cl1': {
        'pair': {'style': 'lj/cut/tip4p/cut', 'vdW': {'Cl_Cl1': (0.22700, 3.51638)}, 'charges': {'Cl_Cl1': -1.0}},
        'bonds': None,
        'angles': None,
        'dihedrals': None,
        'impropers': None
    }
}

Ions = {
    'Cl1': {
        'pair': {
            'style': 'lj/cut/coul/long',
            'vdW': {
                'Cl_Cl1': (
                    0.22700,
                    3.51638)},
            'charges': {
                'Cl_Cl1': -1.0}},
        'bonds': None,
        'angles': None,
        'dihedrals': None,
        'impropers': None}}
