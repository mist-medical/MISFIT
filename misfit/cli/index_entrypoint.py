"""CLI entrypoint for misfit_index — builds the MISFIT metadata index.

Usage::

    misfit_index --input paths.csv    --output index.parquet
    misfit_index --input paths.parquet --output index.parquet --num-workers 64

The ``--input`` file must contain a ``path`` column with absolute paths to
NIfTI files. Both CSV and Parquet formats are accepted.

A ``<output_stem>_config.json`` file is written alongside the Parquet index
the first time the command is run. It records the split ratios and seed used
to assign the ``split`` column. Edit this file and re-run ``misfit_index`` to
obtain a different train/val/test split.
"""
import json
import sys
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd

from misfit.cli.args import ArgParser, add_index_args
from misfit.preprocessing.indexer import DEFAULT_SPLIT_RATIOS, build_index
from misfit.utils.console import print_error, print_info


def _split_config_path(output_path: Path) -> Path:
    """Return the companion split-config path for *output_path*.

    Example: ``/data/index.parquet`` → ``/data/index_config.json``.
    """
    return output_path.parent / (output_path.stem + "_config.json")


def _read_or_create_split_config(output_path: Path) -> dict:
    """Load the split config if it exists, otherwise create it with defaults.

    The config file lives at ``<output_stem>_config.json`` next to the parquet.
    Edit the file and re-run ``misfit_index`` to change split ratios.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` (floats) and
        ``"seed"`` (int).
    """
    config_path = _split_config_path(output_path)
    if config_path.exists():
        with open(config_path) as fh:
            config = json.load(fh)
        print_info(f"Using split config from [bold]{config_path}[/bold]")
    else:
        config = {**DEFAULT_SPLIT_RATIOS, "seed": 42}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)
        print_info(
            f"Split config written to [bold]{config_path}[/bold]  "
            "(edit and re-run to change ratios)"
        )
    return config


def _load_input_paths(input_path: Path) -> list[Path]:
    """Load NIfTI paths from a CSV or Parquet input file.

    The file must contain a ``path`` column. Both ``.csv`` and ``.parquet``
    extensions are supported; any other extension is tried as Parquet first
    and falls back to CSV.

    Args:
        input_path: Path to the input file.

    Returns:
        List of :class:`Path` objects from the ``path`` column.

    Raises:
        SystemExit: If the file does not exist, cannot be read, or lacks a
            ``path`` column.
    """
    if not input_path.exists():
        print_error(f"--input '{input_path}' does not exist.")
        sys.exit(1)

    suffix = input_path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(input_path)
        else:
            # .parquet or unknown — treat as Parquet.
            df = pd.read_parquet(input_path)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Could not read --input '{input_path}': {exc}")
        sys.exit(1)

    if "path" not in df.columns:
        print_error(f"--input '{input_path}' must contain a 'path' column.")
        sys.exit(1)

    return [Path(p) for p in df["path"].tolist()]


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_index",
        description=(
            "Build the MISFIT metadata index from a collection of NIfTI files.\n\n"
            "Computes per-volume statistics (spacing, foreground bbox, intensity\n"
            "percentiles, normalization constants) in parallel and writes the\n"
            "results to a Parquet file. Run this once before training."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_index_args(parser, input_required=True)
    return parser.parse_args(args)


def index_entry(args=None) -> None:
    """Entrypoint for the misfit_index CLI command."""
    ns = _parse_args(args)

    # --- Collect paths ---
    input_path = Path(ns.input)
    nifti_paths = _load_input_paths(input_path)
    if not nifti_paths:
        print_error(f"--input '{input_path}' contains no paths.")
        sys.exit(1)
    print_info(
        f"Loaded [bold]{len(nifti_paths):,}[/bold] NIfTI paths "
        f"from '{input_path}'."
    )

    # --- Split config ---
    output_path = Path(ns.output)
    split_cfg = _read_or_create_split_config(output_path)
    split_ratios = {k: split_cfg[k] for k in ("train", "val", "test")}
    split_seed = int(split_cfg.get("seed", 42))

    # --- Build index ---
    _, errors = build_index(
        nifti_paths=nifti_paths,
        output_path=output_path,
        num_workers=ns.num_workers_index,
        split_ratios=split_ratios,
        split_seed=split_seed,
    )

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    index_entry()
