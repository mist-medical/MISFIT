# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

MISFIT (Medical Imaging Semantic Foundation Toolkit) trains 3D medical imaging
foundation models using masked autoencoders (MAE). It reads unlabeled NIfTI
files and produces a pretrained SwinUNETR-V2 encoder that transfers directly to
MIST for segmentation fine-tuning. No labels required.

## Running tests

Always use the `mist` mamba environment — MISFIT and MIST are both installed
editably there:

```bash
mamba run -n mist pytest
```

Never use plain `pytest` or `python -m pytest`.

## Continuous integration

Two workflows run on every push and PR to `main` (and via `workflow_dispatch`):

- **`test.yml`** — the pytest suite on Python 3.10 / 3.11 / 3.12, gated at
  `--cov-fail-under=100` (the suite is 100%-covered; a regression fails the
  build). A separate `distributed` leg runs the gloo DDP tests
  (`pytest -o addopts= -m distributed`), which the default run skips. On push to
  `main` the 3.12 leg regenerates `coverage.svg` (the README badge) and commits
  it back with `[skip ci]`.
- **`lint.yml`** — `ruff check .` (pinned to the `mist` env's ruff) and
  `codespell` (config in `[tool.codespell]`). `ruff format` is intentionally not
  enforced.

`coverage.svg` is committed to the repo (not git-ignored, unlike `coverage.xml`
/ `.coverage*`). Add domain abbreviations that trip codespell to
`ignore-words-list` in `pyproject.toml` rather than scattering inline ignores.

## Formatting

Python: `ruff check` (lint + import order). All Markdown: Prettier
(`proseWrap: always`, 80 cols) — run `npm run format` / `npm run format:check`
after `npm install`. The Prettier tooling (`package.json`, `.prettierrc.json`,
`node_modules/`) is git-ignored; it is not part of the Python package. Each
mkdocs admonition in `docs/` is preceded by a `<!-- prettier-ignore -->` comment
so Prettier leaves the `!!! note` block and its indented body intact — keep that
comment when adding a new admonition.

## Releasing

`misfit-medical` (PyPI) and `mistmedical/misfit` (Docker Hub) publish on a
GitHub Release, mirroring MIST:

1. Bump the version in **both** `pyproject.toml` and `misfit/__init__.py` (they
   must agree — `tests/unit/test_packaging.py` enforces it; `_build_config`
   writes `misfit.__version__` into `config.json`, the Docker workflow reads
   `pyproject.toml`).
2. Merge to `main` with `test.yml` + `lint.yml` green (they run on the PR, but
   no workflow _gates_ the Release itself — the tag is what ships). Run `pytest`
   locally first regardless.
3. Create a GitHub Release tagged `v<version>` targeting `main`.
   - `.github/workflows/python-publish.yml` builds the sdist+wheel and uploads
     to PyPI using the `PYPI_API_TOKEN` repo secret.
   - `.github/workflows/docker-publish.yml` then fires (`workflow_run`),
     rebuilds from `./Dockerfile` (which `pip install`s the just-published PyPI
     package), and pushes `mistmedical/misfit:<version>` + `:latest`. Needs repo
     secrets `DOCKER_USERNAME` / `DOCKER_PASSWORD`.

The `-alpha`/`-beta`/`-rc` suffix in `pyproject.toml` is transformed to the PEP
440 form (`a0`/`b0`/`rc0`) for the Docker tag.

## Pipeline overview

```
misfit_index  →  misfit_train  →  misfit_evaluate / misfit_inspect
                                        ↓
                               misfit_encode (raw spatial features)
                                        ↓
                            misfit_embed_train (optional aggregator)
                                        ↓
                               misfit_embed (global vector)
```

### CLI entry points (pyproject.toml)

| Command              | Module                          | Purpose                                              |
| -------------------- | ------------------------------- | ---------------------------------------------------- |
| `misfit_index`       | `cli/index_entrypoint.py`       | Build Parquet index from CSV of NIfTI paths          |
| `misfit_train`       | `cli/train_entrypoint.py`       | MAE pretraining (single-GPU to multi-node)           |
| `misfit_evaluate`    | `cli/evaluate_entrypoint.py`    | Reconstruction metrics → CSV                         |
| `misfit_inspect`     | `cli/inspect_entrypoint.py`     | Full-volume reconstruction → NIfTI                   |
| `misfit_encode`      | `cli/encode_entrypoint.py`      | Raw spatial features (N_crops, C, D', H', W')        |
| `misfit_embed`       | `cli/embed_entrypoint.py`       | Global embedding vector (C,) per volume              |
| `misfit_embed_train` | `cli/embed_train_entrypoint.py` | Train crop aggregator (classification / contrastive) |

All argument parsing lives in `cli/args.py`. The `ArgParser` subclass adds
`.arg()` and `.flag()` shorthands. `add_*_args` functions are shared across the
individual entrypoints.

Every `*_entrypoint.py` ends with `if __name__ == "__main__": <x>_entry()` so it
is runnable as `python -m misfit.cli.<x>_entrypoint` — this is the form
`torchrun -m misfit.cli.train_entrypoint` and shell-less container/Kubernetes
manifests use instead of `$(which misfit_train)`. `tests/unit/test_packaging.py`
guards both the guard block and the `-m` path.

## Module map

```
misfit/
  cli/                  Entry points + shared ArgParser / add_*_args
  preprocessing/        NIfTI indexer → Parquet (parallel, ProcessPoolExecutor)
  data_loading/         MISFITDataset + DataLoader; on-the-fly clip+z-score normalization
  models/               MISFITModel base class; SwinMAE (SwinUNETR-V2 + MAE head)
  training/             MAETrainer; optimizer/LR-scheduler registries; training_utils
  loss_functions/       ReconstructionLoss base; masked_mse, masked_l1, normalized_mse
  metrics/              ReconstructionMetric base; masked_mae/mse/psnr (defaults) + ssim (opt-in)
  evaluation/           ReconstructionEvaluator; tiled full-volume inference + CSV output
  inference/            InferenceRunners; tiled reconstruct pipeline (pad→tile→stitch)
  embedding/            Embedder; EmbedTrainer; aggregators (mean_pool, attention_pool);
                        objectives (classification, contrastive)
  utils/                console (Rich), io (read/write JSON), progress_bar,
                        hardware (get_accelerator_type / bf16_supported /
                        resolve_amp / autocast_context),
                        normalization (normalize_patchwise / denormalize_patchwise)
```

## Key design decisions

### Registry pattern

Models, losses, metrics, aggregators, and objectives all use a registry
(`@register_*` decorator, `get_*` lookup, `list_*` for CLI choices).
Registrations are triggered by importing the module (e.g.,
`import misfit.loss_functions`). New implementations only need to inherit the
base class and apply the decorator.

### Normalization

Per-volume clip to [p1, p99] then z-score with foreground mean/std — computed
once at index time, stored in the Parquet index, applied on the fly at load
time. This handles CT and MRI in the same batch without dataset-level
statistics.

### `normalized_masked_mse` loss (default)

Normalizes the target within each `mask_patch_size` cube (zero mean, unit
variance) before computing MSE on masked voxels. This equalizes loss scale
across CT (HU values, large range) and MRI (arbitrary units), enabling
mixed-modality pretraining in one run. The per-cube transform lives in
`misfit.utils.normalization.normalize_patchwise` (with its inverse
`denormalize_patchwise`) — the loss, `misfit_evaluate`, and `misfit_inspect` all
call it, so metrics/reconstructions are in the space the model was optimized in.

### Evaluation metric space (`misfit_evaluate`)

Metrics are computed on masked voxels in the **training loss's space**
(per-`mask_patch_size`-cube normalized target for `normalized_masked_mse`), so
`masked_mse` is directly comparable to `best_val_loss`. The evaluator crops each
volume to its foreground bbox (matching `MISFITDataset`) and draws deterministic
masks from `--seed`. Every metric is also reported for a naive baseline (impute
masked voxels with the visible-region mean) with a `_skill` column (`<metric>` /
`<metric>_naive` / `<metric>_skill` in the CSV); positive skill = beats trivial.
Defaults: `masked_mae`, `masked_mse`, `masked_psnr` (`DEFAULT_METRICS`). `ssim`
is registered but opt-in via `--metrics ssim` — it has no consistent space here;
use `misfit_inspect` for a viewer-space read.

### Model architecture (`models/swinunetr/misfit_swinunetr_mae.py`)

`SwinMAE(MISFITModel)` wraps `SwinUNETR.swinViT` as `self.encoder` (UNet decoder
discarded). The MAE decoder is a lightweight `ConvTranspose3d` stack
(`MAEDecoder`). Masking is applied at the image level before encoding
(SimMIM-style) — required because Swin windowed attention breaks with irregular
token counts.

Constraints:

- All `img_size` dimensions must be divisible by 32 (SwinUNETR-V2 downsamples
  32×).
- All `img_size` dimensions must be divisible by `mask_patch_size` (default 16).
- Default patch size for training: `96×96×96`.

Model sizes (`feature_size`): small=24, base=48 (default), large=96.

### `config.json` as source of truth

`misfit_train` writes `config.json` to the results directory. All downstream
commands (`misfit_evaluate`, `misfit_inspect`, `misfit_encode`, `misfit_embed`)
require `--config` and load architecture from it — no re-specifying model flags.
Resume validation checks model name and patch size (hard error on mismatch).

### Distributed training

`MAETrainer` reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from torchrun environment
variables. The same class runs on 1 GPU or N×M GPUs (NCCL — RCCL on AMD ROCm,
same backend name), and also on CPU / multi-process CPU (gloo) for testing —
`self.use_cuda` gates device placement, backend, and cuDNN tuning; CPU
pretraining is very slow and warns once. ROCm needs no special-casing: PyTorch's
ROCm build reuses the `torch.cuda` namespace and the `cuda:<rank>` device string
as a shim, so `use_cuda` is `True` and every CUDA code path already works.

`_setup_distributed` calls `torch.cuda.set_device(local_rank)` **before**
`dist.init_process_group`, and passes
`device_id=torch.device("cuda", local_rank)`. Both are load-bearing on
multi-GPU: a NCCL group created while every rank is still on the default
`cuda:0` binds DDP's construction-time param-shape allgather to device 0 on all
ranks, so the first collective hangs (`rank 0 has inconsistent 0 params`, 10-min
watchdog timeout). Guarded by
`test_setup_distributed_sets_cuda_device_before_init_process_group`.

AMP is BF16-only — there is no fp16 path and no GradScaler, since BF16 has
float32's dynamic range. It is _requested_ on by default, then resolved against
the actual hardware by `misfit.utils.hardware.resolve_amp` → `bf16_supported()`,
which branches on `get_accelerator_type()` (cuda / rocm / cpu, via
`torch.version.hip`):

- **CUDA:** compute capability ≥ 8.0 (Ampere+: A100, H100, RTX 30xx+). Uses the
  capability, not `torch.cuda.is_bf16_supported()`, which reports True on
  V100/T4 via emulation.
- **ROCm:** `gcnArchName` against `_ROCM_BF16_ACCELERATED_ARCHES` — an
  allow-list of CDNA (MFMA) and RDNA3+ (WMMA) arches. The capability check is
  useless here (ROCm reports ≥ (9,0) for every AMD GPU) and
  `is_bf16_supported()` lies the same way it does on pre-Ampere NVIDIA — it
  returns True on RDNA1/2 (gfx103x), which runs BF16 on shader ALUs with no
  speedup. Unrecognized arch → unsupported (conservative default).
- **CPU:** always False.

`MAETrainer.train()` resolves once (from the CLI default, or from the saved
config on `--resume`) and persists the effective value to `config.json`;
`misfit_evaluate` and `misfit_inspect` re-resolve the config value against their
own hardware. Disable AMP entirely by setting `"amp": false` in `config.json`.
Shared helper: `hardware.autocast_context(enabled)` (device type `"cuda"` for
both CUDA and ROCm, `"cpu"` otherwise).

### Transfer learning to MIST

`get_encoder_state_dict()` remaps keys `encoder.<name>` → `model.swinViT.<name>`
to match MIST's `MistSwinUNETR` checkpoint format. Its output is saved to
`models/encoder_weights.pt` whenever validation loss improves (alongside
`best_model.pt`), so the MIST-ready encoder is always available after training.
Pass it to `mist_train --pretrained-weights models/encoder_weights.pt` along
with `--pretrained-config <misfit results>/config.json` (MIST uses the source
config to validate encoder compatibility; omitting it only warns and skips that
check). Channel mismatch (MISFIT single-channel → MIST multi-channel) is handled
by MIST's `--input-channel-strategy` (default: average).

## Adding new components

**New loss:** subclass `ReconstructionLoss`, apply `@register_loss(name="...")`,
place under `loss_functions/reconstruction/`. Import in
`loss_functions/__init__.py`.

**New metric:** subclass `ReconstructionMetric`, apply
`@register_metric(name="...")`, place under `metrics/`. Import in
`metrics/__init__.py` (or `metrics_registry.py`).

**New model:** subclass `MISFITModel`, implement `get_encoder_state_dict()`,
apply `@register_model(name="...")`, place under `models/<name>/`. Import in
`models/__init__.py`.

**New aggregator:** subclass `AbstractAggregator`, apply
`@register_aggregator(name="...")`, place under `embedding/aggregators/`. Import
in `embedding/aggregators/__init__.py`.

## Output structure (misfit_train)

```
results/
    checkpoints/checkpoint.pt    Latest checkpoint (overwritten each epoch)
    models/best_model.pt         Lowest validation loss
    models/encoder_weights.pt    Encoder-only weights remapped for MIST (model.swinViT.*); refreshed with best_model.pt
    logs/                        TensorBoard event files
    config.json                  Architecture + hyperparameters (required by downstream commands)
```

## Planned next features (not yet implemented)

- `misfit_anomaly` — reconstruction error as anomaly score; zero-label anomaly
  detection
- `misfit_search` — FAISS-backed nearest-neighbor retrieval over embedding
  corpus
- `misfit_visualize` — UMAP of embedding space, attention weight heatmaps
