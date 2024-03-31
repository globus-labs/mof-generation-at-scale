from mofa.model import MOFRecord, NodeDescription
from dataclasses import asdict
from pathlib import Path
from shutil import copy
import json

node_path = '../../tests/files/assemble/nodes/zinc_paddle_pillar.xyz'
node = NodeDescription(
    xyz=Path(node_path).read_text(),
    smiles='[Zn][O]([Zn])([Zn])[Zn]',  # Is this right?
)
with open('node.json', 'w') as fp:
    json.dump(asdict(node), fp, indent=2)

for name in ['geom_difflinker.ckpt', 'hMOF_frag_frag.sdf']:
    copy(f'../../tests/files/difflinker/{name}', name)
