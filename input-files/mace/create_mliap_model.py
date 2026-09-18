"""Convert a MACE model for LAMMPS ML-IAP without CuEquivariance.

MACE 0.3.13's command-line converter enables CuEquivariance unconditionally
for ML-IAP models. The Polaris MOFA environment uses Torch 2.1, which is not
compatible with the CuEquivariance 0.4 Torch operations. The underlying MACE
model already implements the edge-force interface required by ML-IAP, so it
can be wrapped directly.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch

from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    model = torch.load(args.model, map_location="cpu")
    model = model.float() if args.dtype == "float32" else model.double()
    model.lammps_mliap = True

    wrapped = LAMMPS_MLIAP_MACE(model)
    output = args.model.with_name(args.model.name + "-mliap_lammps.pt")
    torch.save(wrapped, output)
    print(output)


if __name__ == "__main__":
    main()
