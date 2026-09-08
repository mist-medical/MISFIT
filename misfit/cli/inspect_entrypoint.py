"""CLI entrypoint for misfit_inspect — MAE reconstruction quality assessment.

Runs the full MAE forward pass (encode → decode) over a NIfTI index and saves
reconstructions as ``.nii.gz`` files.  Use this to visually assess whether
pretraining has produced a useful encoder — plausible reconstructions indicate
that the model has learned meaningful representations.

Each volume is tiled into non-overlapping patches matching the model's native
patch size, reconstructed patch-by-patch, and stitched back into a full-
resolution NIfTI file.

Usage::

    misfit_inspect \\
        --checkpoint best_model.pt \\
        --config     /runs/exp1/config.json \\
        --index      index.parquet \\
        --output-dir /runs/exp1/reconstructions
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

from misfit.cli.args import ArgParser, add_inspect_args
from misfit.inference.inference_runners import reconstruct
from misfit.utils.console import print_error
from misfit.utils.io import read_json_file


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_inspect",
        description=(
            "Reconstruct volumes from a NIfTI index using a pretrained MISFIT "
            "checkpoint.  Reconstructions are saved as .nii.gz files and are "
            "useful for qualitative assessment of pretraining quality."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_inspect_args(parser)
    return parser.parse_args(args)


def inspect_entry(args=None) -> None:
    """Entrypoint for the misfit_inspect CLI command."""
    ns = _parse_args(args)

    config_path = Path(ns.config)
    if not config_path.exists():
        print_error(f"--config '{config_path}' does not exist.")
        sys.exit(1)

    config = read_json_file(config_path)
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    reconstruct(
        index_path=Path(ns.index),
        checkpoint_path=Path(ns.checkpoint),
        output_dir=Path(ns.output_dir),
        model_config=model_config,
        training_config=training_config,
        device=ns.device,
        split=ns.split or None,
    )


if __name__ == "__main__":
    inspect_entry()
