# speckle-py

### DISCLAIMER: the content of this directory is a python rewrite of the original (human-written) Rust code, but this one is mostly end-to-end AI-generated, based on it (used to construct a Kaggle experiment)

PyTorch reimplementation of the Rust/tch-rs `speckle` project (a PixelTransformer-style
model, arXiv:2103.15813): a random subset of observed pixels (positions via Fourier
features + values via a linear embedding) forms transformer tokens, and each token
predicts a heteroscedastic Gaussian over a `p x p` patch of pixels; overlapping patch
predictions are stitched with per-pixel softmax weighting by `-logvar`.

Design goal: **faithful 1:1 port** - same quirks, same config format, same parameter
names, so Rust and Python artifacts are interoperable.

## Environment

Developed against the venv at `$PYENV` (torch 2.7.1+rocm6.3, safetensors, pyarrow, pillow, numpy). CPU is the default device;
pass `--device cuda` to override. The `data/` directory is a symlink to the Rust
project's cache (shared parquet downloads); replace it freely.

## Usage

```bash
PY= #PATH_TO_PYTHON

# train from a TOML config (Rust-compatible)
$PY -m speckle train --config my.toml --tag mytag

# train with CLI overrides (any config field; kebab-case flags like the Rust CLI)
$PY -m speckle train --dataset SplitLine --emb-dim 32 --n-blocks 1 --qk-norm \
    --norm RMSNorm --max-iters 101 --eval-interval 50 --tag smoke

# train on the STL-10 unlabeled pool (100k 96x96 RGB; cap it for quick runs)
$PY -m speckle train --dataset Stl10Unlabeled --max-images 2000 --batch-size 4 \
    --starting-density 0.25 --ending-density 0.05 --device cuda --tag stl

# inspect a checkpoint (Rust- or Python-produced)
$PY -m speckle eval checkpoints/run_20260914_2148_py_smoke

# visualize random source-point selection (writes vis_sample{seed}.png;
# replaces the Rust winit window with a PNG)
$PY -m speckle --seed 42 vis --dataset SplitLine --source-points 192
```

Outputs land in `checkpoints/run_%Y%m%d_%H%M_py[_tag]/`: `config.toml`,
`scheme.txt`, `losses.csv`, `train_step*_pred*.png` / `val_step*_pred*.png`
(4-row grids: target | context | prediction | variance), `model.safetensors`.

## Interoperability with the Rust version

- **Config**: `config.toml` written here parses in Rust and vice versa. The loader
  ignores unknown fields (older Rust checkpoints carry legacy keys such as `pe`,
  `triangular_distance_field`, `pope_frequency_base`).
- **Weights**: state-dict keys are identical to tch VarStore names
  (`b0.self_attention.w_q.weight`, `pos_enc.proj`, `unembedding.weight`, ...).
  Rust `model.safetensors` loads here and Python checkpoints load in Rust.
- **Numerics**: loading the Rust CIFAR checkpoint
  (`run_20260913_1929_-dirty_ff-cifar-b3-h4`) and re-evaluating reproduces the
  Rust losses (train -3.5765 vs -3.5736, val -3.5860 vs -3.5281; residual spread
  is due to different eval-time pixel permutations).
- **scheme.txt**: mirrors Rust's quirk of reporting `trainable == total` (tch marks
  every VarStore variable trainable at creation; `requires_grad_(false)` only freezes
  autograd), while the per-variable `(trainable)` marker follows the autograd flag.

## Tests

```bash
$PY tests/smoke_test.py     # unit checks: NS iteration, config TOML, forward/backward,
                            # attentive residual, LayerNorm variant, safetensors roundtrip
$PY tests/interop_test.py   # load Rust checkpoint, numeric cross-check vs its losses.csv
```

## Deliberate divergences from the Rust code

- `make_pictures` uses a true pixelwise softmax: the max subtraction is a
  `scatter_reduce_(amax)` over the patches covering each pixel, instead of Rust's
  per-patch max (a workaround for the missing scatter-amax in the tch-rs/ATen of that
  version). This makes the stitching exact inverse-variance blending, and the
  variance channel reports the true fused (product-of-Gaussians) variance.
  Visualization-only; training loss untouched. See `NOTES.md`.
- `data/` shared via symlink; `py` instead of a git hash in run-dir names.
- `vis` mode writes a PNG instead of opening a winit/softbuffer window.
- Timing strings in `losses.csv`/stdout are Python-formatted seconds (`12.34s`).
- RNG streams differ (same seeds, same algorithms, not bit-identical runs).
- `Batcher` no longer materializes per-image pixel permutations at all (Rust builds one
  int64 N x h*w table, earlier this port stored int32 chunks); permutations are now
  generated lazily per batch from a dedicated generator — see below.
- Only `SplitLine` synthetic generation is implemented; `SmoothedSplitLine` raises
  `NotImplementedError`, matching the `todo!()` in Rust.
- **Datasets are decoded to uint8** (`[0, 255]`); `Batcher` converts each batch to
  float32 `[0, 1]` (bit-identical to the previous decode-time `/255`). Cuts the 100k
  STL-10 pool from ~11 GB to ~2.8 GB. `SplitLine.generate` now emits uint8 too.
- **`Batcher` generates pixel permutations lazily per batch** from a dedicated
  `torch.Generator` (seeded by `--seed` in `train()`, offset per batcher) instead of
  pre-generating the whole N x h*w permutation table. Same distribution, same
  deterministic batches for a given seed, ~0 steady-state cost (the old table needed
  ~3.3 GB per 100k-image 96x96 batcher).
- **`DatasetConfig.max_images`** (default -1 = all): deterministic first-N subsample
  cap, applied in `employ()`; the multi-shard loader also stops downloading/decoding
  early. Emitted into `config.toml`; the Rust loader ignores the unknown key, but Rust
  itself has no `Stl10Unlabeled` enum value, so STL-10 configs are Python-only.
- **`Stl10Unlabeled`**: the `unlabeled` split (100k 96x96 RGB) of
  `huggingface.co/datasets/jxie/stl10`, downloaded as 4 parquet shards with HTTP-Range
  resume + retries, decoded to `data/train/stl10_unlabeled[_<cap>|_full].npy` so the
  one-time PIL decode never repeats. There is no test split; the val pool comes from
  `train_val_split` over the same pool.
- **Batches are moved to the compute device** in `train()`/eval/vis paths (previously
  they stayed on CPU, so `--device cuda` crashed).
- The eval-loss loop **recycles an exhausted Batcher** instead of returning a partial
  average (matters only when the pool is smaller than `eval_iters * batch_size`, e.g.
  capped STL-10 runs); identical behavior for large pools.
- `vis`'s source-point mask is per-pixel (the old per-channel mask crashed
  `tensor_to_rgba` for RGB datasets).
- Test files locate the Rust checkpoint configs relative to the test file instead of
  the historical hardcoded `../../Programming/...` cwd.

See `NOTES.md` for open questions to discuss (patch-centering offset semantics,
pixelwise-softmax max-subtraction subtlety in `make_pictures`).
