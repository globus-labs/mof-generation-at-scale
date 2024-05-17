import argparse
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import TQDMProgressBar

from mofa.utils.src.const import NUMBER_OF_ATOM_TYPES, GEOM_NUMBER_OF_ATOM_TYPES
from mofa.utils.src.lightning import DDPM
from mofa.utils.src.utils import disable_rdkit_logging


def _intel_on_train_start(trainer: Trainer):
    """Hook for optimizing the model and optimizer before training"""
    import intel_extension_for_pytorch as ipex
    assert len(trainer.optimizers) == 1, 'We only support one optimizer for now'
    trainer.model, trainer.optimizers[0] = ipex.optimize(trainer.model, optimizer=trainer.optimizers[0])


# Placeholder until XPU support merged: https://github.com/Lightning-AI/pytorch-lightning/pull/17700
class XPUAccelerator(Accelerator):
    """Shim for XPU support"""

    # See: https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html#create-a-custom-accelerator

    def setup(self, trainer: Trainer) -> None:
        pass

    def setup_device(self, device: torch.device) -> None:
        return

    def teardown(self) -> None:
        return

    @staticmethod
    def parse_devices(devices: Any) -> Any:
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return True

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:
        # Return optional device statistics for loggers
        return {}


def get_args(args: list[str]) -> argparse.Namespace:
    """Assemble arguments form model training

    Args:
        args: Arguments to use to configure
    Returns:
        A Namespace object containing a mix of the default arguments and those provided by the user
    """
    p = argparse.ArgumentParser(description='E3Diffusion')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    p.add_argument('--data', action='store', type=str, default="datasets")
    p.add_argument('--train_data_prefix', action='store', type=str, default='zinc_final_train')
    p.add_argument('--val_data_prefix', action='store', type=str, default='zinc_final_val')
    p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--device', action='store', type=str, default='cpu')
    p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
    p.add_argument('--log_iterations', action='store', type=str, default=20)

    p.add_argument('--exp_name', type=str, default='YourName')
    p.add_argument('--model', type=str, default='egnn_dynamics', help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics |gnn_dynamics')
    p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5, )
    p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')

    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--brute_force', type=eval, default=False, help='True | False')
    p.add_argument('--actnorm', type=eval, default=True, help='True | False')
    p.add_argument('--break_train_epoch', type=eval, default=False, help='True | False')
    p.add_argument('--dp', type=eval, default=True, help='True | False')
    p.add_argument('--condition_time', type=eval, default=True, help='True | False')
    p.add_argument('--clip_grad', type=eval, default=True, help='True | False')
    p.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
    # EGNN args -->
    p.add_argument('--n_layers', type=int, default=6, help='number of layers')
    p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
    p.add_argument('--nf', type=int, default=128, help='number of layers')
    p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
    p.add_argument('--norm_constant', type=float, default=1, help='diff/(|diff| + norm_constant)')
    p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    p.add_argument('--ode_regularization', type=float, default=1e-3)
    p.add_argument('--dataset', type=str, default='qm9', help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
    p.add_argument('--datadir', type=str, default='qm9/temp', help='qm9 directory')
    p.add_argument('--filter_n_atoms', type=int, default=None, help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    p.add_argument('--dequantization', type=str, default='argmax_variational', help='uniform | variational | argmax_variational | deterministic')
    p.add_argument('--n_report_steps', type=int, default=1)
    p.add_argument('--wandb_usr', type=str)
    p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
    p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    p.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    p.add_argument('--save_model', type=eval, default=True, help='save model')
    p.add_argument('--generate_epochs', type=int, default=1, help='save model')
    p.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    p.add_argument('--test_epochs', type=int, default=1)
    p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--start_epoch', type=int, default=0, help='')
    p.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    p.add_argument('--augment_noise', type=float, default=0)
    p.add_argument('--n_stability_samples', type=int, default=500, help='Number of samples to compute the stability')
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help='normalize factors for [x, categorical, integer]')
    p.add_argument('--remove_h', action='store_true')
    p.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
    p.add_argument('--visualize_every_batch', type=int, default=1e8, help="Can be used to visualize multiple times per epoch")
    p.add_argument('--normalization_factor', type=float, default=1, help="Normalize the sum aggregation of EGNN")
    p.add_argument('--aggregation_method', type=str, default='sum', help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
    p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
    p.add_argument('--center_of_mass', type=str, default='fragments', help='Where to center the data: fragments | prompts')
    p.add_argument('--inpainting', action='store_true', default=False, help='Inpainting mode (full generation)')
    p.add_argument('--remove_anchors_context', action='store_true', default=False, help='Remove anchors context')
    p.add_argument('--dataset_override', type=str, default="", help="Dataset override flag - set to MOFA for retraining")
    disable_rdkit_logging()

    return p.parse_args(args)


def find_last_checkpoint(checkpoints_dir):
    epoch2fname = [
        (int(fname.split('=')[1].split('.')[0]), fname)
        for fname in os.listdir(checkpoints_dir)
        if fname.endswith('.ckpt') and "-v" not in fname
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(checkpoints_dir, latest_fname)


def main(
        args,
        run_directory: Path
):
    """Run model training

    Args:
        args: Arguments from `get_args` and the configuration file
        run_directory: Directory in which to write output files
    """

    # TODO (wardlt): I trimmed off Hyun's code for organizing experimental data. We should put it back if
    #  we'd want to use this codebase for experimenting with DiffLinker as well as using it production

    # Create a new directory and start logging
    log_directory = run_directory / 'logs'
    checkpoints_dir = run_directory / 'chkpt'
    for path in [log_directory, checkpoints_dir]:
        path.mkdir(exist_ok=True, parents=True)

    with (run_directory / 'stdout.txt').open('w') as fo, (run_directory / 'stderr.txt').open('w') as fe:
        with redirect_stderr(fe), redirect_stdout(fo):
            # Determine the number of atom types
            is_geom = ('geom' in args.train_data_prefix) or ('MOAD' in args.train_data_prefix) or (args.dataset_override == "MOFA")
            number_of_atoms = GEOM_NUMBER_OF_ATOM_TYPES if is_geom else NUMBER_OF_ATOM_TYPES
            in_node_nf = number_of_atoms + args.include_charges
            anchors_context = not args.remove_anchors_context
            context_node_nf = 2 if anchors_context else 1
            if '.' in args.train_data_prefix:
                context_node_nf += 1

            # Make an XPU acceleator, if needed
            if 'xpu' in args.device:
                pl_device = XPUAccelerator()
                devices = [0]
            else:
                pl_device = args.device
                devices = "auto"

            checkpoint_callback = [callbacks.ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename='difflinker_{epoch:02d}',
                monitor='loss/val',
                save_top_k=10),
                TQDMProgressBar()
            ]
            trainer = Trainer(
                default_root_dir=log_directory,
                max_epochs=args.n_epochs,
                callbacks=checkpoint_callback,
                accelerator=pl_device,
                devices=devices,
                num_sanity_val_steps=0,
                enable_progress_bar=args.enable_progress_bar,
            )

            # Add a callback for fit setup
            if args.device == "xpu":
                trainer.on_train_start = _intel_on_train_start

            # Get the model
            if args.resume is None:
                # Create the model from scratch
                ddpm = DDPM(
                    data_path=args.data,
                    train_data_prefix=args.train_data_prefix,
                    val_data_prefix=args.val_data_prefix,
                    in_node_nf=in_node_nf,
                    n_dims=3,
                    context_node_nf=context_node_nf,
                    hidden_nf=args.nf,
                    activation=args.activation,
                    n_layers=args.n_layers,
                    attention=args.attention,
                    tanh=args.tanh,
                    norm_constant=args.norm_constant,
                    inv_sublayers=args.inv_sublayers,
                    sin_embedding=args.sin_embedding,
                    normalization_factor=args.normalization_factor,
                    aggregation_method=args.aggregation_method,
                    diffusion_steps=args.diffusion_steps,
                    diffusion_noise_schedule=args.diffusion_noise_schedule,
                    diffusion_noise_precision=args.diffusion_noise_precision,
                    diffusion_loss_type=args.diffusion_loss_type,
                    normalize_factors=args.normalize_factors,
                    include_charges=args.include_charges,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    torch_device=args.device,
                    model=args.model,
                    test_epochs=args.test_epochs,
                    n_stability_samples=args.n_stability_samples,
                    normalization=args.normalization,
                    log_iterations=args.log_iterations,
                    samples_dir=None,
                    data_augmentation=args.data_augmentation,
                    center_of_mass=args.center_of_mass,
                    inpainting=args.inpainting,
                    anchors_context=anchors_context,
                    dataset_override=args.dataset_override
                )
            else:
                last_checkpoint = find_last_checkpoint(checkpoints_dir)
                ddpm = DDPM.load_from_checkpoint(
                    last_checkpoint,
                    map_location=args.device,
                    strict=False,
                    data_path=args.data,
                    train_data_prefix=args.train_data_prefix,
                    val_data_prefix=args.val_data_prefix,
                    in_node_nf=in_node_nf,
                    n_dims=3,
                    context_node_nf=context_node_nf,
                    hidden_nf=args.nf,
                    activation=args.activation,
                    n_layers=args.n_layers,
                    attention=args.attention,
                    tanh=args.tanh,
                    norm_constant=args.norm_constant,
                    inv_sublayers=args.inv_sublayers,
                    sin_embedding=args.sin_embedding,
                    normalization_factor=args.normalization_factor,
                    aggregation_method=args.aggregation_method,
                    diffusion_steps=args.diffusion_steps,
                    diffusion_noise_schedule=args.diffusion_noise_schedule,
                    diffusion_noise_precision=args.diffusion_noise_precision,
                    diffusion_loss_type=args.diffusion_loss_type,
                    normalize_factors=args.normalize_factors,
                    include_charges=args.include_charges,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    torch_device=args.device,
                    model=args.model,
                    test_epochs=args.test_epochs,
                    n_stability_samples=args.n_stability_samples,
                    normalization=args.normalization,
                    log_iterations=args.log_iterations,
                    samples_dir=None,
                    data_augmentation=args.data_augmentation,
                    center_of_mass=args.center_of_mass,
                    inpainting=args.inpainting,
                    anchors_context=anchors_context,
                    dataset_override=args.dataset_override)

            # Force loading of the dataset now before we start distributed training
            #  There might be issues in each training rank writing to disk at the same time
            # TODO (wardlt): Separate the data loader from the model code so it's clearer how to set up
            ddpm.setup('fit')
            trainer.fit(model=ddpm)

            # Save the last model
            trained_path = run_directory / 'model.ckpt'
            trainer.save_checkpoint(trained_path)
            return trained_path
