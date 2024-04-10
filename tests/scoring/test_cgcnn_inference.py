from pytest import mark
import os
from mofa.scoring.cgcnn_inference import run_cgcnn_pred_wrapper_serial
from ase.io import read
from pathlib import Path


@mark.parametrize('extxyz_name', ['test-zn'])
def test_run_cgcnn_pred_wrapper_serial(extxyz_name, cif_dir, tmpdir):
    my_ase_mofs = [read(Path(cif_dir) / x, format="cif") for x in os.listdir(cif_dir) if x.endswith(".cif")]
    pred, std = run_cgcnn_pred_wrapper_serial(my_ase_mofs, manual_batch_size=7, ncpus_to_load_data=1)
    assert len(pred) == len(my_ase_mofs)
    assert len(std) == len(my_ase_mofs)
