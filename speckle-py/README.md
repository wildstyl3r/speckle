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

Developed against the venv at `../.env` (torch 2.7.1+rocm6.3, safetensors, pyarrow, pillow, numpy). CPU is the default device;
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
- `Batcher` stores the fixed per-image pixel permutations as int32 in chunks
  (Rust materializes one int64 tensor); same distribution, half the RAM.
- Only `SplitLine` synthetic generation is implemented; `SmoothedSplitLine` raises
  `NotImplementedError`, matching the `todo!()` in Rust.

See `NOTES.md` for open questions to discuss (patch-centering offset semantics,
pixelwise-softmax max-subtraction subtlety in `make_pictures`).
