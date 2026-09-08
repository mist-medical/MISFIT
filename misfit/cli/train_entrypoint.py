"""CLI entrypoint for misfit_train — MAE pretraining.

Single-GPU::

    misfit_train --index index.parquet --results /runs/exp1 --model swinunetr-base

Multi-GPU (single node)::

    torchrun --nproc_per_node=4 -m misfit.cli.train_entrypoint \\
        --index index.parquet --results /runs/exp1 --model swinunetr-base

Multi-node (4 nodes × 8 GPUs)::

    torchrun --nproc_per_node=8 --nnodes=4 \\
             --node_rank=<rank> --master_addr=<addr> --master_port=29500 \\
             -m misfit.cli.train_entrypoint \\
        --index index.parquet --results /runs/exp1 --model swinunetr-base

``torchrun -m misfit.cli.train_entrypoint`` needs nothing on ``PATH`` and no
shell, so it is the form to use in a container image or a Kubernetes manifest
(``$(which misfit_train)`` only works when a shell evaluates it). ``torchrun
$(which misfit_train)`` still works from an interactive shell.
"""
from argparse import ArgumentDefaultsHelpFormatter

from misfit.cli.args import ArgParser, add_train_args
from misfit.training.trainers.mae_trainer import MAETrainer


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_train",
        description="MISFIT MAE pretraining — single-GPU to multi-node.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_train_args(parser)
    return parser.parse_args(args)


def train_entry(args=None) -> None:
    """Entrypoint for the misfit_train CLI command."""
    ns = _parse_args(args)
    trainer = MAETrainer(ns)
    trainer.train()


if __name__ == "__main__":
    train_entry()
