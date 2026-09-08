Usage
===

## Overview

MISFIT is a **command-line tool** for pretraining and deploying 3D medical
imaging foundation models. The core pipeline consists of four stages:

1. **Indexing** — Scans a manifest of NIfTI files, computes per-volume intensity
   statistics and spacing, assigns train/val/test splits, and produces a Parquet
   index that all downstream commands consume.

2. **Pretraining** — Trains a SwinUNETR masked autoencoder (MAE) on the indexed
   dataset. The encoder learns to reconstruct randomly masked patches of each
   volume, producing generalizable semantic representations without any labels.

3. **Evaluation / Inspection** — Commands for measuring reconstruction quality
   on the validation set and generating full-volume reconstruction NIfTIs for
   visual inspection.

4. **Embedding** — Extracts per-volume feature vectors from the pretrained
   encoder for downstream tasks such as classification, retrieval, and anomaly
   detection. A lightweight aggregator can be fine-tuned on labeled data with
   `misfit_embed_train`.

---

## Indexing

The **indexing step** scans your NIfTI files and records the path, intensity
statistics (p1, p99, foreground mean, foreground std), voxel spacing, image
shape, and affine transform for each volume. It also assigns each volume a
`split` label (`train`, `val`, or `test`) using configurable ratios. The output
is a single Parquet file consumed by all downstream MISFIT commands.

Run indexing with `misfit_index`:

- `--input FILE` (**required**): CSV or Parquet file with a `path` column
  listing absolute paths to NIfTI files.
- `--output PARQUET` (**required**): Destination path for the output Parquet
  index.
- `--num-workers-index N`: Number of parallel worker processes. _(default: 32)_

### Split configuration

On first run, `misfit_index` writes a companion `<output_stem>_config.json` file
alongside the Parquet index. This file records the split ratios and random seed
used to assign the `split` column. Edit it and re-run `misfit_index` to change
the split proportions.

### Example

Index from a CSV manifest.

```console
misfit_index --input  /data/paths.csv \
             --output /data/index.parquet
```

Index with 64 parallel workers.

```console
misfit_index --input             /data/paths.csv \
             --output            /data/index.parquet \
             --num-workers-index 64
```

### Output

A single Parquet file at the path given by `--output`. Each row corresponds to
one NIfTI volume and contains the following columns:

| Column                                  | Description                                                               |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `volume_id`                             | Unique identifier derived from the filename (no extension).               |
| `path`                                  | Absolute path to the NIfTI file.                                          |
| `split`                                 | Dataset split: `train`, `val`, or `test`.                                 |
| `shape_d` / `shape_h` / `shape_w`       | Voxel dimensions (depth, height, width).                                  |
| `spacing_d` / `spacing_h` / `spacing_w` | Voxel spacing in mm.                                                      |
| `affine`                                | JSON-encoded 4×4 affine transform matrix.                                 |
| `fg_x_start` / `fg_x_end`               | Foreground bounding box extent along x.                                   |
| `fg_y_start` / `fg_y_end`               | Foreground bounding box extent along y.                                   |
| `fg_z_start` / `fg_z_end`               | Foreground bounding box extent along z.                                   |
| `p1`                                    | 1st-percentile foreground intensity (lower clip bound for normalization). |
| `p99`                                   | 99th-percentile foreground intensity (upper clip bound).                  |
| `fg_mean`                               | Foreground mean intensity after clipping.                                 |
| `fg_std`                                | Foreground standard deviation after clipping.                             |

---

## Training

The **training step** pretrains a SwinUNETR masked autoencoder on the indexed
dataset. At each iteration a random 75% of patch tokens are masked, and the
model learns to reconstruct the missing voxels from the visible context.
Train/val split is determined by the `split` column in the index.

Run training with `misfit_train`:

### Data

- `--index PARQUET` (**required**): Parquet index produced by `misfit_index`.
  Rows with `split='train'` are used for training; rows with `split='val'` for
  validation.
- `--num-cpu-workers N`: CPU worker processes for data loading. _(default: 8)_

### Output

- `--results DIR` (**required**): Directory for checkpoints, `config.json`, and
  TensorBoard logs.

### Model

- `--model NAME`: SwinMAE variant to train. _(default: `swinunetr-base`)_ See
  [Model Variants](advanced_topics.md#model-variants) for details.
- `--patch-size D H W`: Spatial crop size fed to the model in voxels. Must be
  divisible by 32. _(default: `96 96 96`)_
- `--mask-patch-size P`: Edge length of each masked 3D cube in voxels.
  _(default: 16)_
- `--mask-ratio R`: Fraction of patch tokens to mask. _(default: 0.75)_

### Loss

- `--loss NAME`: Reconstruction loss function. _(default:
  `normalized_masked_mse`)_ See
  [Loss Functions](advanced_topics.md#loss-functions) for available options.

### Optimisation

- `--epochs N`: Total training epochs. _(default: 200)_
- `--batch-size N`: Batch size per GPU. _(default: 2)_
- `--optimizer NAME`: Optimizer. _(default: `adamw`)_
- `--learning-rate LR`: Initial learning rate. _(default: 1e-4)_
- `--weight-decay WD`: L2 weight decay. _(default: 0.05)_
- `--lr-scheduler NAME`: Learning rate schedule. _(default: `cosine`)_
- `--warmup-epochs N`: Linear warmup epochs before cosine decay. _(default: 20)_

### Miscellaneous

- `--seed N`: Random seed for reproducibility. _(default: 42)_
- `--resume`: Resume from the latest checkpoint in `--results`. The model
  architecture and patch size must match the saved `config.json`; changes to
  other hyperparameters emit warnings but are allowed.
- `--overwrite`: Discard any existing checkpoint and `config.json` in
  `--results` and start fresh.

<!-- prettier-ignore -->
!!!note
    `--resume` and `--overwrite` are mutually exclusive. If neither is passed and
    a `config.json` already exists in `--results`, `misfit_train` will exit with
    an error rather than silently overwriting your run.

### Example

Train a base SwinMAE for 200 epochs.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1
```

Train a small model on 4 GPUs using `torchrun`.

```console
torchrun --nproc_per_node=4 \
    -m misfit.cli.train_entrypoint \
        --index   /data/index.parquet \
        --results /runs/exp1 \
        --model   swinunetr-small
```

Resume a run that was interrupted.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume
```

### Output

```text
results/
    checkpoints/
        checkpoint.pt       Latest checkpoint (overwritten each epoch).
    models/
        best_model.pt       Checkpoint with the lowest validation loss.
    logs/                   TensorBoard event files.
    config.json             Reproducibility config (architecture, patch size,
                            hyperparameters, and MISFIT version).
```

`config.json` is the **single source of truth** for model architecture. It is
required by `misfit_evaluate`, `misfit_inspect`, and `misfit_embed` to
reconstruct the correct model without re-specifying any flags.

---

## Evaluation

The **evaluation step** measures reconstruction quality on the validation split.
For each volume, the evaluator tiles the full volume into non-overlapping
patches, runs a complete MAE forward pass on each patch, computes masked
reconstruction metrics per patch, and averages the results across all patches to
produce one score per volume. The per-tile mask is drawn from `--seed` so the
whole evaluation is reproducible.

Metrics are computed in the **training loss's space**: when the run used
`normalized_masked_mse` the target is normalised per `mask_patch_size` cube —
exactly as the loss does — so `masked_mse` is directly comparable to the run's
`best_val_loss`. Because a per-region-normalised error is ≈ 1.0 for _any_
constant predictor, every metric is also reported for a **naive baseline**
(impute the masked voxels with the visible-region mean) plus a `_skill` column
(`1 - model/naive` for lower-is-better metrics, `model - naive` for
higher-is-better) — positive means the encoder beats the trivial baseline.

<!-- prettier-ignore -->
!!!note
    Metrics are computed on the **masked patches only** — the voxels the encoder
    never saw. This directly measures the MAE pretraining objective.

Run evaluation with `misfit_evaluate`:

- `--checkpoint PT` (**required**): Path to a checkpoint produced by
  `misfit_train`.
- `--index PARQUET` (**required**): Parquet index produced by `misfit_index`.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and the metric space are read from this
  file; the `evaluation` section lists the metrics when `--metrics` is omitted.
- `--output-csv CSV` (**required**): Path where the evaluation results CSV will
  be written.
- `--metrics NAME [NAME ...]`: Metrics to compute, overriding the config.
  _(default: `masked_mae masked_mse masked_psnr`)_. `ssim` is opt-in — see the
  note below.
- `--seed N`: Base RNG seed; the mask for tile _i_ is drawn from `seed + i`.
  _(default: 42)_
- `--split SPLIT`: If the index contains a `split` column, only rows whose split
  matches this value are evaluated. Pass `--split ""` to evaluate all rows.
  _(default: `val`)_
- `--device DEVICE`: Torch device (e.g. `cuda:0`, `cpu`). _(default: auto)_

### Example

Evaluate on the validation split (default).

```console
misfit_evaluate --checkpoint  /runs/exp1/models/best_model.pt \
                --index       /data/index.parquet \
                --config      /runs/exp1/config.json \
                --output-csv  /runs/exp1/eval_results.csv
```

Evaluate on the test split.

```console
misfit_evaluate --checkpoint  /runs/exp1/models/best_model.pt \
                --index       /data/index.parquet \
                --config      /runs/exp1/config.json \
                --output-csv  /runs/exp1/test_results.csv \
                --split       test
```

### Output

A single CSV file at the path given by `--output-csv`. Each row is one volume,
with three columns per metric — `<metric>`, `<metric>_naive`, `<metric>_skill`.
Five summary rows are appended at the bottom of the file.

| `volume_id`     | `masked_mse` | `masked_mse_naive` | `masked_mse_skill` | ... |
| --------------- | ------------ | ------------------ | ------------------ | --- |
| BRAIN_001       | 0.44         | 1.00               | +0.56              | ... |
| BRAIN_002       | 0.47         | 1.00               | +0.53              | ... |
| ...             |              |                    |                    |     |
| Mean            | 0.46         | 1.00               | +0.54              | ... |
| Std             | 0.02         | 0.01               | 0.02               | ... |
| 25th Percentile | 0.44         | 0.99               | +0.52              | ... |
| Median          | 0.46         | 1.00               | +0.54              | ... |
| 75th Percentile | 0.48         | 1.01               | +0.56              | ... |

`misfit_evaluate` also prints a short `model | naive | verdict` summary to the
console.

| Metric        | Description                                       | Direction        |
| ------------- | ------------------------------------------------- | ---------------- |
| `masked_mae`  | Mean absolute error on masked voxels.             | Lower is better  |
| `masked_mse`  | Mean squared error on masked voxels — the value   | Lower is better  |
|               | `normalized_masked_mse` optimises.                |                  |
| `masked_psnr` | Peak signal-to-noise ratio on masked voxels (dB). | Higher is better |
| `ssim`        | Structural similarity (opt-in, see note).         | Higher is better |

<!-- prettier-ignore -->
!!!note
    `ssim` is **not** in the default set. It needs a consistent absolute-intensity
    space and a spatial window, and per-cube normalisation depresses its absolute
    value with boundary seams. Request it with `--metrics ssim` if you want it in
    the CSV, but for a viewer-space structural read use `misfit_inspect` instead.

---

## Inspection

The **inspection step** produces a full-volume reconstruction NIfTI for every
volume in an index. This is the primary tool for visually assessing pretraining
quality — open the input and output side-by-side in a viewer such as ITK-SNAP or
3D Slicer to see where reconstruction succeeds and fails.

The reconstruction pipeline:

1. Loads and z-score normalises the volume.
2. Zero-pads to the nearest multiple of the patch size in every dimension.
3. Tiles the padded volume into non-overlapping patches and runs MAE
   reconstruction on each.
4. Stitches the reconstructed patches and their masks back into the full padded
   volume.
5. Trims padding to restore the original voxel dimensions.
6. Denormalises reconstruction intensities back to the original intensity space
   (`reconstruction × fg_std + fg_mean`).
7. Saves outputs under two subdirectories of `--output-dir`:
   - `reconstructions/<volume_id>.nii.gz` — full-volume reconstruction.
   - `masks/<volume_id>.nii.gz` — binary visibility mask in the same space.

Run inspection with `misfit_inspect`:

- `--checkpoint PT` (**required**): Path to a pretrained MISFIT checkpoint.
- `--index PARQUET` (**required**): Parquet index of volumes to reconstruct.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and patch size are read from this file.
- `--output-dir DIR` (**required**): Directory where `<volume_id>.nii.gz` files
  are written.
- `--split SPLIT`: If the index contains a `split` column, only rows whose split
  matches this value are reconstructed. Omit to reconstruct all rows. _(default:
  None — all rows)_
- `--device DEVICE`: Torch device. _(default: auto)_

### Example

Reconstruct all volumes in the index.

```console
misfit_inspect --checkpoint  /runs/exp1/models/best_model.pt \
               --index       /data/index.parquet \
               --config      /runs/exp1/config.json \
               --output-dir  /runs/exp1/reconstructions
```

Reconstruct only the validation split.

```console
misfit_inspect --checkpoint  /runs/exp1/models/best_model.pt \
               --index       /data/index.parquet \
               --config      /runs/exp1/config.json \
               --output-dir  /runs/exp1/reconstructions \
               --split       val
```

<!-- prettier-ignore -->
!!!note
    Both outputs are in the **original coordinate space** (same affine as the
    input volume). The reconstruction has **denormalized intensities** and the
    mask is binary (0/1). Load both in ITK-SNAP or 3D Slicer and overlay the
    mask to see exactly which regions the model reconstructed from scratch.

### Output

```text
output-dir/
    reconstructions/
        <volume_id>.nii.gz   Full-volume reconstruction with denormalized intensities.
    masks/
        <volume_id>.nii.gz   Binary mask: 1 = masked (reconstructed by model),
                             0 = visible (seen by encoder). Overlay in a
                             viewer to highlight reconstructed regions.
```

---

## Embedding

The **embedding step** tiles each volume into non-overlapping crops, encodes
each crop with the pretrained encoder, and aggregates all crop features into a
single global `(C,)` embedding vector per volume. With `--aggregator mean_pool`
(default) no training is needed — embeddings are ready immediately for zero-shot
retrieval or UMAP visualization. Pass a trained aggregator checkpoint via
`--aggregator-checkpoint` for task-specific pooling.

Run embedding extraction with `misfit_embed`:

- `--encoder-checkpoint PT` (**required**): Path to a pretrained MISFIT encoder
  checkpoint.
- `--index PARQUET` (**required**): Parquet index of volumes to embed.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and patch size are read from this file.
- `--output-dir DIR` (**required**): Directory where `.npz` files are saved.
- `--aggregator NAME`: Aggregation strategy for combining patch-level features
  into a single volume-level embedding. _(default: `mean_pool`)_ Options:
  `mean_pool`, `attention_pool`.
- `--aggregator-checkpoint PT`: Path to a trained aggregator checkpoint produced
  by `misfit_embed_train`. Required when `--aggregator attention_pool`.
- `--split SPLIT`: If the index contains a `split` column, only rows whose split
  matches this value are embedded. Omit to embed all rows. _(default: None — all
  rows)_
- `--device DEVICE`: Torch device. _(default: auto)_

### Example

Extract mean-pooled embeddings for all volumes (zero-shot, no aggregator
training required).

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index              /data/index.parquet \
             --config             /runs/exp1/config.json \
             --output-dir         /data/embeddings
```

Embed only the test split.

```console
misfit_embed --encoder-checkpoint /runs/exp1/models/best_model.pt \
             --index              /data/index.parquet \
             --config             /runs/exp1/config.json \
             --output-dir         /data/embeddings \
             --split              test
```

Extract embeddings with a trained attention-pooling aggregator.

```console
misfit_embed --encoder-checkpoint  /runs/exp1/models/best_model.pt \
             --index               /data/index.parquet \
             --config              /runs/exp1/config.json \
             --output-dir          /data/embeddings \
             --aggregator          attention_pool \
             --aggregator-checkpoint /runs/agg/aggregator.pt
```

### Output

One `.npz` file per volume at `<output-dir>/<volume_id>.npz`, containing:

| Key         | Shape  | Description                                    |
| ----------- | ------ | ---------------------------------------------- |
| `embedding` | `(C,)` | Single global embedding vector for the volume. |

---

## Encoding

The **encoding step** extracts the full spatial bottleneck feature map from the
pretrained encoder for every crop of every volume. Unlike `misfit_embed`, which
global-average-pools each crop down to a single `(C,)` vector, `misfit_encode`
preserves the spatial structure within each crop, producing
`(N_crops, C, D', H', W')` tensors where `D' = H' = W' = patch_size / 32` (e.g.
3 for a 96-voxel crop).

Use `misfit_encode` when you need spatially-rich features for:

- Training aggregators that attend over spatial tokens within each crop
- Dense prediction fine-tuning (segmentation, detection)
- Any downstream model that benefits from sub-crop spatial context

Run encoding with `misfit_encode`:

- `--encoder-checkpoint PT` (**required**): Path to a pretrained MISFIT encoder
  checkpoint.
- `--index PARQUET` (**required**): Parquet index of volumes to encode.
- `--config JSON` (**required**): Path to the `config.json` produced by
  `misfit_train`. Model architecture and patch size are read from this file.
- `--output-dir DIR` (**required**): Directory where `.npz` files are saved.
- `--split SPLIT`: If the index contains a `split` column, only rows whose split
  matches this value are encoded. Defaults to None (all rows).
- `--device DEVICE`: Torch device. _(default: auto)_

### Example

```console
misfit_encode --encoder-checkpoint /runs/exp1/models/best_model.pt \
              --index               /data/index.parquet \
              --config              /runs/exp1/config.json \
              --output-dir          /data/encodings
```

### Output

One `.npz` file per volume at `<output-dir>/<volume_id>.npz`, containing:

| Key           | Shape                      | Description                                    |
| ------------- | -------------------------- | ---------------------------------------------- |
| `feature_map` | `(N_crops, C, D', H', W')` | Full spatial bottleneck feature maps.          |
| `positions`   | `(N_crops, 3)`             | Normalised 3D centre coordinates of each crop. |

---

## Embedding Training

The **embedding training step** fine-tunes a lightweight aggregator head on top
of frozen per-crop features extracted by `misfit_encode`. The aggregator learns
to pool the per-crop feature vectors into a single discriminative volume-level
embedding. Because the encoder features are pre-computed and cached on disk,
training is fast even for large datasets.

Run aggregator training with `misfit_embed_train`:

### Input

- `--input CSV` (**required**): Unified CSV with columns:

  | Column          | Description                                                   |
  | --------------- | ------------------------------------------------------------- |
  | `volume_id`     | Volume identifier.                                            |
  | `split`         | Dataset split. Only rows where `split='train'` are used.      |
  | `features_path` | Absolute path to the `.npz` file produced by `misfit_encode`. |
  | `label`         | String label for the training objective.                      |

  Labels are always treated as strings. Integer or boolean labels should be
  converted to strings before passing. The mapping from label strings to integer
  indices is saved in the output `aggregator.pt` checkpoint for inference-time
  decoding.

### Output

- `--output-dir DIR` (**required**): Directory for the trained `aggregator.pt`
  and training logs.

### Model

- `--aggregator NAME`: Aggregator architecture. _(default: `attention_pool`)_
- `--objective NAME`: Training objective. _(default: `classification`)_ Options:
  `classification` (cross-entropy), `contrastive` (Supervised Contrastive with
  K=2 pairs per group).
- `--embed-dim C` (**required**): Dimensionality of the encoder bottleneck
  features. Must match the feature files from `misfit_encode`.
- `--no-position-encoding`: Disable learned 3D position encoding in
  `AttentionPoolAggregator`. _(default: off)_

### Training

- `--epochs N`: Training epochs. _(default: 50)_
- `--batch-size N`: Batch size. For contrastive training this must be even.
  _(default: 32)_
- `--learning-rate LR`: Initial learning rate. _(default: 1e-3)_
- `--num-workers-embed N`: DataLoader worker processes. _(default: 4)_
- `--device DEVICE`: Torch device. _(default: `cuda`)_

### Example

Train an attention-pooling aggregator for classification.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768
```

Train with a contrastive objective.

```console
misfit_embed_train --input      /data/train_manifest.csv \
                   --output-dir /runs/agg \
                   --embed-dim  768 \
                   --objective  contrastive
```

### Output

```text
output-dir/
    aggregator.pt   Trained aggregator weights and label_to_idx mapping.
                    Pass to misfit_embed via --aggregator-checkpoint to
                    apply the trained aggregator at inference time.
```
