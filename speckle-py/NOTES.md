# Notes

## Patch centering / coordinate offset (OPEN - to discuss)

Current behavior (`patch_coords` + `forward_with_loss` + `make_pictures`, identical in
Rust and this port):

- `padding = patch_side // 2` (integer division)
- patch anchors: `dy, dx in [0, patch_side)` relative to the source pixel, i.e.
  top-left anchoring in *padded* space
- a real pixel `(i, j)` lives at padded index `(i + padding, j + padding)`
- patch pixels land at padded indices `y .. y+patch_side-1` (same for x)

For ODD `patch_side` this already centers every patch on its source pixel: the patch
covers padded rows `[y, y + patch_side)`, whose midpoint `y + patch_side // 2` maps
back to real coordinate `y` after the padding crop is removed. For EVEN `patch_side`
the patch is asymmetric (one extra row/column below-right) and `patch_side // 2`
floors.

Open questions to settle later:

1. If intent = explicit centering, should we switch to offsets
   `dy, dx in [-(p//2), ...]` and re-verify the mask/padding interplay (mask placement
   at `[1..h+1]` is tied to the current convention)?
2. Should even `patch_side` be rejected explicitly?

## make_pictures stitching: pixelwise softmax semantics (RESOLVED 2026-09-14)

Original author's remark: the function constructing images from patches should act
like softmax but pixelwise - for each pixel, the weights are gathered from the patches
having presence in that pixel, and a pixel may be covered by no patch at all.

Rust behavior (per-patch max): scores = -logvar [b, n, p*p]; the max subtraction was
done PER PATCH (`amax` over each patch's own p*p scores) because the tch-rs/ATen of
that version had no scatter max reduction for the correct stabilization. The
per-pixel normalization and the no-coverage case were already correct (scatter_add of
exp scores into numerator/denominator at destination pixels, division per pixel,
uncovered pixels guarded by the coverage mask), but the weights were
`softmax(score - patch_internal_max)` rather than an exact pixelwise softmax of the
raw scores, since per-element shifts do not cancel inside the per-pixel ratio.

Resolution: this implementation uses a TRUE pixelwise softmax. For each destination
pixel, the max over the scores of all patches covering it is computed via
`scatter_reduce_(reduce="amax")` (available in torch >= 1.12, missing in the old
ATen), and the shifted scores are exponentiated, scatter_add'ed and divided exactly
as before. Every exp argument is <= 0 (overflow-safe) and each covered pixel has at
least one term equal to exp(0) = 1, so the denominator is >= 1. The change affects
evaluation visualization only; the training loss is patchwise and untouched.

Because scores = -logvar and the per-pixel max now cancels inside the per-pixel
ratio, the stitching weights are exactly normalized inverse variances
(w_j proportional to 1/sigma_j^2), i.e. the mean of the product of Gaussians -
inverse-variance blending, the precision-weighted MAP fusion of the overlapping
patch predictions.

The variance channel reports the true FUSED variance as well:
1 / sum_j exp(s_j) = exp(-M_q) / denominator, where -M_q = min_j logvar_j <= 0
(numerically safe; underflows to 0 only for extremely confident pixels). Pixels
covered by a single patch show that patch's actual predicted sigma^2 (previously a
constant 1 there); uncovered pixels are inf, masked out of display as before.
