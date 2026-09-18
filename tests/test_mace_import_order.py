"""Exercise model loading in a fresh process so native import order is tested."""
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MACEImportOrderTests(unittest.TestCase):
    def test_mofa_model_before_mace_loads_and_evaluates(self):
        # Importing MACERunner in the test runner first would mask this crash.
        self.check_import_order('from mofa.model import MOFRecord')

    def test_pybel_before_mace_loads_and_evaluates(self):
        self.check_import_order('from openbabel import pybel')

    def check_import_order(self, first_import):
        result = subprocess.run(
            [sys.executable, '-X', 'faulthandler', '-c', first_import + '\n' + '''
from mofa.simulation.mace import load_model
from ase import Atoms
import numpy as np
atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
atoms.calc = load_model('cpu')
assert np.isfinite(atoms.get_potential_energy())
assert np.isfinite(atoms.get_forces()).all()
print('MACE energy and forces: OK')
'''],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('MACE energy and forces: OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
