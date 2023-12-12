from pathlib import Path

from mofa.assemble import assemble_pillaredPaddleWheel_pcuMOF

_files_dir = Path(__file__).parent / 'files' / 'assemble'


def test_paddlewheel_pcu():
    # Find a set of linkers by pulling from different folders
    chosen_folders = list(_files_dir.glob('linkers/molGAN-*'))[:3]
    coo_linkers = [f
                   for d in chosen_folders[:2]
                   for f in d.glob("linker-COO*.xyz")]
    pillar_linker = next(f for f in chosen_folders[2].glob("*.xyz") if 'COO' not in f.name)
    node = _files_dir / 'nodes/zinc_paddle_pillar.xyz'

    name = Path('test_mof.cif')
    assemble_pillaredPaddleWheel_pcuMOF(node, coo_linkers, pillar_linker, name)
