Advanced Topics
===

## Reproducibility and Resumption

MISFIT writes a `config.json` file to the `--results` directory at the start of
training. This file records the MISFIT version, model architecture, patch size,
and all training hyperparameters. It is the **single source of truth** for model
architecture — `misfit_evaluate`, `misfit_inspect`, and `misfit_embed` all
require it via `--config` rather than re-accepting architecture flags.

### config.json structure

Below is an example `config.json` produced by `misfit_train`.

```json
{
  "misfit_version": "0.1.0-alpha",

  "data": {
    "index": "/data/index.parquet"
  },

  "model": {
    "architecture": "swinunetr-base",
    "patch_size": [96, 96, 96],
    "mask_patch_size": 16,
    "mask_ratio": 0.75
  },

  "training": {
    "epochs": 200,
    "batch_size": 2,
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "weight_decay": 0.05,
    "lr_scheduler": "cosine",
    "warmup_epochs": 20,
    "loss": "normalized_masked_mse",
    "amp": true,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "bucket_cap_mb": 200
  },

  "evaluation": {
    "masked_mae": {},
    "masked_mse": {},
    "masked_psnr": {}
  }
}
```

The `evaluation` section lists the metrics `misfit_evaluate` computes when
`--metrics` is not given. `ssim` is registered but not a default — request it
with `--metrics ssim` (its absolute value is depressed by cube-boundary seams;
use `misfit_inspect` for a viewer-space structural read).

### Resuming a run

Pass `--resume` to continue from the latest checkpoint in `--results`. MISFIT
will read `config.json` and validate that the architecture and patch size are
unchanged. If either has changed, training exits with an error rather than
silently producing an incompatible checkpoint.

Changes to other hyperparameters (learning rate, epochs, optimizer, etc.) are
allowed and emit a warning so you are aware of the discrepancy.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --resume
```

### Starting fresh

If you want to discard a previous run and start from scratch, pass
`--overwrite`. Training proceeds without error even if `config.json` already
exists — a new `config.json` is written at the start and checkpoints are
overwritten epoch by epoch. Prior checkpoint files are not deleted upfront; they
are replaced as training progresses.

```console
misfit_train --index   /data/index.parquet \
             --results /runs/exp1 \
             --overwrite
```

<!-- prettier-ignore -->
!!!warning
    `--resume` and `--overwrite` are mutually exclusive. If neither is passed and
    `config.json` already exists in `--results`, `misfit_train` refuses to run.
    This is intentional — it prevents accidentally overwriting a completed run.

### Immutable vs. mutable parameters

The following parameters are **immutable** — changing them while resuming raises
a hard error:

| Parameter               | Reason                                                    |
| ----------------------- | --------------------------------------------------------- |
| `model.architecture`    | Checkpoint weights are architecture-specific.             |
| `model.patch_size`      | Determines the spatial dimension of all model tensors.    |
| `model.mask_patch_size` | Determines the masking grid structure inside the encoder. |

All other parameters (learning rate, epochs, optimizer, loss, etc.) are
**mutable** — changes produce a warning and training continues.

---

## Model Variants

MISFIT provides three SwinMAE variants corresponding to different encoder
capacities. All variants share the same SwinUNETR backbone architecture but
differ in the width of the feature maps (`feature_size`).

| Variant | `--model`         | `feature_size` | Parameters (approx.)  | Recommended for                        |
| ------- | ----------------- | -------------- | --------------------- | -------------------------------------- |
| Small   | `swinunetr-small` | 24             | ~7M (4.8M encoder)    | Rapid prototyping, small datasets      |
| Base    | `swinunetr-base`  | 48             | ~28M (18.6M encoder)  | Standard pretraining (default)         |
| Large   | `swinunetr-large` | 96             | ~110M (73.9M encoder) | Large-scale datasets, maximum capacity |

<!-- prettier-ignore -->
!!!note
    The model variant is locked into `config.json` at the start of training and
    cannot be changed when resuming. To use a different variant, start a new run
    with `--overwrite` or in a new `--results` directory.

---

## Patch Size Selection

The `--patch-size` argument controls the spatial crop fed to the model during
training and inference. Every dimension must be divisible by 32 (due to the
SwinUNETR downsampling stages).

A few practical guidelines:

- **GPU memory** is the primary constraint. A 32 GB GPU with batch size 2 can
  comfortably fit `96 96 96`. Reduce to `64 64 64` if you run out of memory.

- **AMP** — MISFIT uses `bfloat16` automatic mixed precision during training.
  BF16 has the same dynamic range as float32, so no GradScaler is needed and
  training is numerically stable for long runs. AMP is _requested_ on by
  default, then resolved against the actual hardware: BF16 acceleration needs a
  GPU with matrix hardware for it — an NVIDIA Ampere+ GPU (A100, H100, RTX
  30xx+) or an AMD CDNA / RDNA3+ GPU (MI200/MI300 series, RX 7000+). Pre-Ampere
  NVIDIA GPUs (V100, T4), older AMD GPUs (RDNA1/2 — RX 5000/6000 series; these
  report BF16 as available but run it on shader ALUs with no speed benefit), and
  CPU automatically fall back to FP32 with a warning. `misfit_train` resolves
  this once and writes the effective value into `config.json`; `misfit_evaluate`
  and `misfit_inspect` re-resolve it against their own hardware, so evaluating
  on a login node without a suitable GPU still works. To disable AMP entirely,
  let training run for at least one epoch (so `config.json` is written), then
  set `"amp": false` in the `training` section of `config.json` and restart with
  `--resume`.

- **Anisotropic data is handled natively.** MISFIT records each volume's voxel
  spacing (mm) in the index and injects it into the model via sinusoidal
  `SpacingEmbedding` conditioning on the decoder bottleneck. The model learns
  the physical size of each crop regardless of acquisition protocol — no
  resampling to isotropic spacing is required. Volumes from thick-slice CT (e.g.
  5 mm slices) and thin-slice MRI can coexist in the same training run.

- **The patch size is fixed at inference time.** `misfit_inspect`,
  `misfit_evaluate`, and `misfit_embed` all read `patch_size` from `config.json`
  via `--config`. You do not need to specify it again on the command line.

---

## Mask Ratio

The `--mask-ratio` controls what fraction of patch tokens are hidden from the
encoder during pretraining. The default of 0.75 (75%) follows the original MAE
paper and works well for 3D medical images.

Higher mask ratios force the model to learn longer-range spatial dependencies;
lower ratios make the reconstruction task easier and may be preferable for
datasets with complex, fine-grained anatomy where local context is important.

---

## Loss Functions

| Loss                  | `--loss`                | Description                                                                                                       |
| --------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Normalized Masked MSE | `normalized_masked_mse` | MSE computed on masked patches, normalized by patch variance. Recommended for mixed-modality datasets (CT + MRI). |
| Masked MSE            | `masked_mse`            | Standard MSE on masked patches without variance normalization.                                                    |
| Masked L1             | `masked_l1`             | Mean absolute error on masked patches. More robust to intensity outliers than MSE.                                |

The `normalized_masked_mse` loss is recommended as the default because it
normalizes each patch's contribution by its local variance, preventing
high-contrast regions (e.g., bone in CT) from dominating the gradient signal.
This is especially useful when pretraining on datasets that mix modalities with
very different intensity distributions.

---

## Optimizers

| Optimizer | `--optimizer` | Notes                                                                       |
| --------- | ------------- | --------------------------------------------------------------------------- |
| AdamW     | `adamw`       | Default. Best general-purpose choice for ViT-based architectures.           |
| Adam      | `adam`        | No weight decay. Use `adamw` for better regularization.                     |
| SGD       | `sgd`         | Requires careful learning rate tuning. Not recommended for MAE pretraining. |

---

## Learning Rate Schedulers

| Scheduler  | `--lr-scheduler` | Description                                            |
| ---------- | ---------------- | ------------------------------------------------------ |
| Cosine     | `cosine`         | Cosine annealing from `--learning-rate` to 0. Default. |
| Polynomial | `polynomial`     | Polynomial decay.                                      |
| Constant   | `constant`       | No decay; learning rate stays at `--learning-rate`.    |

All schedulers support a **linear warmup** phase controlled by
`--warmup-epochs`. During warmup the learning rate increases linearly from 0 to
`--learning-rate`. A warmup of 20 epochs is recommended for SwinMAE — skipping
warmup can cause instability in the early stages of training.

---

## Multi-Node Training

MISFIT uses `torch.distributed` and can be launched with `torchrun` on any
cluster that supports NCCL. Distributed setup is handled automatically when
`torchrun` sets the `LOCAL_RANK` environment variable.

`-m misfit.cli.train_entrypoint` is the training target: it needs nothing on
`PATH` and no shell, so it works identically from an interactive shell, inside a
container image, and in a Kubernetes `command:` array. `$(which misfit_train)`
also works, but only when a shell evaluates it — a bare `command` / `args` list
in a pod spec passes the literal string `$(which` to Python.

### Single node, multiple GPUs

```console
torchrun --nproc_per_node=4 \
    -m misfit.cli.train_entrypoint \
        --index      /data/index.parquet \
        --results    /runs/exp1 \
        --batch-size 2
```

<!-- prettier-ignore -->
!!!note
    `--batch-size` is the **per-GPU** batch size. The effective global batch
    size is `--batch-size × number of GPUs`. Adjust `--learning-rate` accordingly
    (linear scaling rule: multiply LR by the number of GPUs when scaling up).

### Multiple nodes

On a SLURM cluster or similar, set `MASTER_ADDR` and `MASTER_PORT` and use
`torchrun --nnodes` and `--node_rank`:

```console
torchrun --nnodes=2 \
         --nproc_per_node=4 \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
    -m misfit.cli.train_entrypoint \
        --index   /data/index.parquet \
        --results /runs/exp1
```

### Kubernetes

A pod's `command` / `args` go straight to `execve` — there is no shell — so
`$(which misfit_train)` reaches Python as the literal string `$(which`. Use the
`-m` form (nothing to resolve), or run the command through an explicit shell:

```yaml
# Direct — no shell.
command: ["torchrun", "--nproc_per_node=4", "-m", "misfit.cli.train_entrypoint"]
args: ["--index", "/data/index.parquet", "--results", "/runs/exp1"]

# Or, to keep $(...) / env-var expansion, wrap it:
command: ["bash", "-lc"]
args: ["torchrun --nproc_per_node=4 -m misfit.cli.train_entrypoint --index ..."]
```

Multi-pod `Job`s set `--nnodes` / `--node_rank` / `--master_addr` /
`--master_port` exactly as in the SLURM example above.

### If a multi-GPU run hangs at startup

MISFIT pins each rank to its own GPU before initialising the process group, so
the common "every rank piled onto `cuda:0`" hang does not apply. If a run still
stalls at startup and (after ~10 minutes) dies with a NCCL collective-timeout
(`Watchdog caught collective operation timeout`), the cause is GPU-to-GPU
transport, not MISFIT. The usual trigger is a GPU set that spans two CPU sockets
/ PCIe root complexes with no NVLink path between them — common when
`CUDA_VISIBLE_DEVICES` hand-picks GPUs on a shared node. Run with
`NCCL_DEBUG=INFO` and look at the `Setting affinity for GPU N to <cpu list>`
lines: if the selected GPUs fall into two different CPU ranges, they're on
separate sockets. Fixes, in order of preference:

```console
# 1. Pick GPUs on one socket (same CPU-affinity range) — best interconnect.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m misfit.cli.train_entrypoint ...

# 2. Keep all the GPUs, let NCCL use NVLink where it exists and shared memory
#    across the gap.
NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=4 -m misfit.cli.train_entrypoint ...

# 3. Disable direct P2P entirely — always works, uses shared-memory staging
#    (some throughput cost).
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 -m misfit.cli.train_entrypoint ...
```

---

## Embedding Aggregators

When running `misfit_embed`, the encoder produces a feature map for each cubic
crop extracted from the volume. An **aggregator** pools these per-crop feature
vectors into a single volume-level embedding.

### Mean Pool (`mean_pool`)

The simplest aggregator: computes the unweighted mean of all crop feature
vectors. No training is required — it works zero-shot directly after
pretraining.

Use `mean_pool` when you want a quick, training-free embedding for retrieval or
visualization (e.g., UMAP of a cohort).

### Attention Pool (`attention_pool`)

A learnable aggregator that uses a multi-head cross-attention mechanism to
weight crop contributions, conditioned on their 3D spatial positions. It can
learn to focus on diagnostically relevant regions for a given task.

Requires training with `misfit_embed_train` before use. Pass the resulting
`aggregator.pt` to `misfit_embed` via `--aggregator-checkpoint`.

---

## Embedding Training Objectives

Both objectives train only the aggregator — the pretrained encoder weights are
frozen. This makes embedding training fast and memory-efficient even on large
feature sets.

### Classification (`classification`)

Optimizes a cross-entropy loss for multi-class label prediction. The aggregator
learns to produce discriminative embeddings for the `label` column in your
`--input` CSV. Labels are treated as strings and mapped to integer indices
lexicographically; the mapping is saved in `aggregator.pt` for inference-time
decoding.

### Contrastive (`contrastive`)

Optimizes a Supervised Contrastive loss (SupCon). Volumes sharing the same label
are pulled together in embedding space; volumes with different labels are pushed
apart. Uses K=2 pairs per group.

Contrastive training generally produces more generalizable embeddings than
classification training, at the cost of requiring balanced sampling.
`--batch-size` must be even.

---

## Preparing the Embedding Training Input CSV

`misfit_embed_train` accepts a single unified CSV with four required columns:

| Column          | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| `volume_id`     | Volume identifier — used for logging only.                      |
| `split`         | Dataset split. Only `split='train'` rows are used for training. |
| `features_path` | Absolute path to the `.npz` file produced by `misfit_encode`.   |
| `label`         | String label for the training objective.                        |

A minimal example:

```csv
volume_id,split,features_path,label
CT_001,train,/data/embeddings/CT_001.npz,adenocarcinoma
CT_002,train,/data/embeddings/CT_002.npz,squamous_cell
CT_003,val,/data/embeddings/CT_003.npz,adenocarcinoma
CT_004,val,/data/embeddings/CT_004.npz,squamous_cell
```

You can include val and test rows in the same file — only `split='train'` rows
are loaded for aggregator training.

---

## MIST Integration

MISFIT pretrained encoders can be transferred directly into
[MIST](https://github.com/mist-medical/MIST) for supervised 3D medical image
segmentation. The pretrained SwinViT encoder provides a better initialization
than random weights, especially when labeled data is scarce.

### Overview

```
Unlabeled NIfTI corpus
        │
        ▼
  misfit_index          ← scan paths, compute intensity stats, assign splits
        │
        ▼
  misfit_train          ← self-supervised MAE pretraining
        │                 encoder_weights.pt saved automatically
        ▼
  misfit_evaluate /     ← optional: verify reconstruction quality
  misfit_inspect
        │
        ▼
  mist_train            ← supervised segmentation fine-tuning
  --pretrained-weights  ← point at encoder_weights.pt
```

### Encoder export

`misfit_train` automatically saves `encoder_weights.pt` to `results/models/`
whenever the validation loss improves. This file contains encoder weights with
keys remapped from `encoder.*` to `model.swinViT.*` — the format MIST's
SwinUNETR expects — so no manual export step is required.

```text
results/
    models/
        best_model.pt       Full MAE checkpoint.
        encoder_weights.pt  Encoder-only weights, remapped for MIST.
```

### Fine-tuning in MIST

Pass `encoder_weights.pt` to `mist_train` via `--pretrained-weights`. The
architecture variant must match the one used during MISFIT pretraining — both
tools use the same variant names (`swinunetr-small`, `swinunetr-base`,
`swinunetr-large`) with identical `feature_size` values:

```console
mist_train \
    --numpy              /path/to/preprocessed/data \
    --results            /path/to/mist/results \
    --model              swinunetr-small \
    --pretrained-weights /runs/pretrain/models/encoder_weights.pt \
    --warmup-epochs      10
```

### Handling channel mismatches

MISFIT trains on single-channel images. MIST tasks are often multi-channel
(e.g., four MRI contrasts for brain tumor segmentation). MIST's
`--input-channel-strategy` flag controls how the single-channel patch embedding
is adapted to the multi-channel model:

| Strategy  | Behaviour                                                                                |
| --------- | ---------------------------------------------------------------------------------------- |
| `average` | Average source channels to one, then tile to match the target channel count. _(default)_ |
| `first`   | Use only the first source channel, then tile to match the target channel count.          |
| `skip`    | Keep the patch embedding at random initialization; do not transfer it.                   |

`average` is the recommended default:

```console
mist_train \
    --numpy                  /path/to/preprocessed/data \
    --results                /path/to/mist/results \
    --model                  swinunetr-small \
    --pretrained-weights     /runs/pretrain/models/encoder_weights.pt \
    --input-channel-strategy average \
    --warmup-epochs          10
```

### When pretraining helps most

Transfer is most beneficial in **low-label regimes** — tasks where the number of
annotated cases is small relative to the model capacity.

- **Few labeled cases (< ~50)** — expect the largest gains. The pretrained
  encoder reduces the number of labeled cases needed to reach a given Dice
  score.
- **Domain match matters** — pretraining on volumes from the same scanner, field
  strength, and modality as the target task transfers better than out-of-domain
  pretraining.
- **Warmup is important** — always use `--warmup-epochs` (5–10 epochs) in MIST
  when fine-tuning from MISFIT weights. A full-LR update at epoch 0 can damage
  pretrained encoder features before the decoder has adapted.

<!-- prettier-ignore -->
!!! note
    MISFIT encoder weights are only compatible with MIST's SwinUNETR
    architectures (`swinunetr-small`, `swinunetr-base`, `swinunetr-large`).
    Other MIST architectures (nnUNet, MedNeXt, FMG-Net, W-Net) have different
    encoder structures and are not compatible with MISFIT checkpoints.
