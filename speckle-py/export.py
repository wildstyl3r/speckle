"""Single-file export of the speckle-py codebase (for copy-paste into a Jupyter notebook).

Generated from the `speckle/` package: all modules concatenated verbatim in dependency
order; only relative imports were removed and `dataset.employ/hwc(...)` call sites in
`train()` adapted (there is no `dataset` module object in a flat file). `main()` and the
CLI entry point were dropped — the notebook config cell at the bottom replaces them.

Also carries the notebook-only divergences from the Rust code (see README.md): uint8
dataset pipeline with per-batch float conversion, lazy per-batch pixel permutations with
dedicated Batcher generators, `DatasetConfig.max_images` subsampling cap, the
`Stl10Unlabeled` dataset (sharded HF parquet + resumable download + decoded-npy cache),
batches moved to the compute device, and eval-loop batcher recycling.

Sections (top to bottom): config, batcher, muon, model primitives (fourier_pe, norm,
mhsa, swiglu, block), utils (output, split/eval, image, info), Model, dataset loaders,
cli, training/eval/vis entry points, notebook config cell.
"""

import argparse
import csv
import dataclasses
import io
import json
import math
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


# ===== speckle/config.py =====

DATASETS = ("SplitLine", "SmoothedSplitLine", "Mnist", "Cifar10", "Stl10Unlabeled")
DECAY_MODES = ("Cosine", "Linear")
NORM_OPTIONS = ("LayerNorm", "RMSNorm")
RESIDUAL_OPTIONS = ("Plain", "Attentive")


@dataclass
class LrScheduleConfig:
    max_iters: int = 10001
    max_lr: float = 3e-3
    min_lr_share: float = 0.10
    lr_decay: str = "Cosine"
    warmup_iters: int = 1500
    max_coast_iters: int = 1000

    def get_lr(self, step: int) -> float:
        if step < self.warmup_iters:
            return self.max_lr * (step / self.warmup_iters)
        if step < self.warmup_iters + self.max_coast_iters:
            return self.max_lr
        decay_steps = float(self.max_iters - self.warmup_iters - self.max_coast_iters)
        s = float(step) - decay_steps
        if self.lr_decay == "Cosine":
            return self.max_lr * (
                self.min_lr_share
                + (1.0 - self.min_lr_share) * 0.5 * (1.0 + math.cos(math.pi * s / decay_steps))
            )
        if self.lr_decay == "Linear":
            return self.max_lr * (
                self.min_lr_share + (1.0 - self.min_lr_share) * (1.0 - s / decay_steps)
            )
        raise ValueError(f"unknown lr_decay: {self.lr_decay!r}")


@dataclass
class AttentionConfig:
    num_heads: int = 4
    qk_norm: bool = False


def _check_enum(value, allowed, what):
    if value not in allowed:
        raise ValueError(f"invalid {what}: {value!r} (expected one of {allowed})")
    return value


@dataclass
class BlockConfig:
    norm: str = "LayerNorm"
    swiglu_dim: int = 128
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    def __post_init__(self):
        if isinstance(self.attention, dict):
            self.attention = _build(AttentionConfig, self.attention)
        _check_enum(self.norm, NORM_OPTIONS, "norm")


@dataclass
class ModelConfig:
    dropout: float = 0.2
    id_loss_scale: float = 0.0
    image_channels: int = 3
    max_image_side: int = -1
    n_blocks: int = 4
    block: BlockConfig = field(default_factory=BlockConfig)
    patch_side: int = 3
    emb_dim: int = 64
    batch_size: int = 1
    eval_batch_size: int = 10
    residual: str = "Plain"

    def __post_init__(self):
        if isinstance(self.block, dict):
            self.block = _build(BlockConfig, self.block)
        _check_enum(self.residual, RESIDUAL_OPTIONS, "residual")


@dataclass
class DatasetConfig:
    train_share: float = 0.9
    dataset: str = "Mnist"
    max_images: int = -1

    def __post_init__(self):
        _check_enum(self.dataset, DATASETS, "dataset")


@dataclass
class TrainConfig:
    lr_schedule: LrScheduleConfig = field(default_factory=LrScheduleConfig)
    eval_interval: int = 500
    eval_iters: int = 200
    accumulation_steps: int = 4
    starting_density: float = 0.5
    ending_density: float = 0.1
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self):
        if isinstance(self.lr_schedule, dict):
            self.lr_schedule = _build(LrScheduleConfig, self.lr_schedule)
        if isinstance(self.model, dict):
            self.model = _build(ModelConfig, self.model)
        if isinstance(self.dataset, dict):
            self.dataset = _build(DatasetConfig, self.dataset)

    @classmethod
    def load(cls, path) -> "TrainConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return _build(cls, data)

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_toml())

    def to_toml(self) -> str:
        m = self.model
        a = m.block.attention
        lines = []
        lines.append(f"eval_interval = {_fmt(self.eval_interval)}")
        lines.append(f"eval_iters = {_fmt(self.eval_iters)}")
        lines.append(f"accumulation_steps = {_fmt(self.accumulation_steps)}")
        lines.append(f"starting_density = {_fmt(self.starting_density)}")
        lines.append(f"ending_density = {_fmt(self.ending_density)}")
        lines.append("")
        lines.append("[lr_schedule]")
        ls = self.lr_schedule
        lines.append(f"max_iters = {_fmt(ls.max_iters)}")
        lines.append(f"max_lr = {_fmt(ls.max_lr)}")
        lines.append(f"min_lr_share = {_fmt(ls.min_lr_share)}")
        lines.append(f'lr_decay = "{ls.lr_decay}"')
        lines.append(f"warmup_iters = {_fmt(ls.warmup_iters)}")
        lines.append(f"max_coast_iters = {_fmt(ls.max_coast_iters)}")
        lines.append("")
        lines.append("[model]")
        lines.append(f"dropout = {_fmt(m.dropout)}")
        lines.append(f"id_loss_scale = {_fmt(m.id_loss_scale)}")
        lines.append(f"image_channels = {_fmt(m.image_channels)}")
        lines.append(f"n_blocks = {_fmt(m.n_blocks)}")
        lines.append(f"patch_side = {_fmt(m.patch_side)}")
        lines.append(f"emb_dim = {_fmt(m.emb_dim)}")
        lines.append(f"batch_size = {_fmt(m.batch_size)}")
        lines.append(f"eval_batch_size = {_fmt(m.eval_batch_size)}")
        lines.append(f'residual = "{m.residual}"')
        lines.append("")
        lines.append("[model.block]")
        lines.append(f'norm = "{m.block.norm}"')
        lines.append(f"swiglu_dim = {_fmt(m.block.swiglu_dim)}")
        lines.append("")
        lines.append("[model.block.attention]")
        lines.append(f"num_heads = {_fmt(a.num_heads)}")
        lines.append(f"qk_norm = {'true' if a.qk_norm else 'false'}")
        lines.append("")
        lines.append("[dataset]")
        lines.append(f"train_share = {_fmt(self.dataset.train_share)}")
        lines.append(f"max_images = {_fmt(self.dataset.max_images)}")
        lines.append(f'dataset = "{self.dataset.dataset}"')
        return "\n".join(lines) + "\n"


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _build(cls, data: dict):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


# ===== speckle/batcher.py =====

class Batcher:
    def __init__(self, data: torch.Tensor, batch_size: int, seed: int | None = None):
        total_size, h, w, channels = data.shape
        self.data = data
        self.batch_size = batch_size
        self.total_size = total_size
        self.channels = channels
        self.h = h
        self.w = w
        self.current_index = 0
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.generator = torch.Generator().manual_seed(seed)

    def next(self):
        if self.current_index >= self.total_size:
            return None
        end_index = min(self.current_index + self.batch_size, self.total_size)
        effective_batch_size = end_index - self.current_index
        scores = torch.randint(
            self.h * self.w * self.h * self.w,
            (effective_batch_size, self.h * self.w),
            generator=self.generator,
        )
        source_ix = scores.argsort(-1).view(effective_batch_size, self.h * self.w)
        targets = self.data[self.current_index:end_index].view(
            effective_batch_size, self.h, self.w, self.channels
        )
        if targets.dtype == torch.uint8:
            targets = targets.float().div_(255)
        self.current_index = end_index
        return source_ix, targets


# ===== speckle/muon.py =====

A = 3.4445
B = -4.775
C = 2.0315


def newton_schulz(x: torch.Tensor, turbo: bool, eps: float) -> torch.Tensor:
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.transpose(-1, -2)

    if turbo:
        a = x @ x.transpose(-1, -2)
        s = a.abs().sum(-1).clamp_min(eps).rsqrt()
        a = a * s.unsqueeze(-2) * s.unsqueeze(-1)
        x = x * s.unsqueeze(-1)
        iters = 4
    else:
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
        a = x @ x.transpose(-1, -2)
        iters = 5

    for i in range(iters):
        if i != 0:
            a = x @ x.transpose(-1, -2)
        b = B * a + C * (a @ a)
        x = A * x + b @ x

    if tall:
        x = x.transpose(-1, -2)
    return x


class Muon:
    def __init__(self, parameters, lr: float, mu: float, weight_decay: float, nesterov: bool, turbo: bool, eps: float):
        self.parameters = list(parameters)
        self.momenta = [torch.zeros_like(w) for w in self.parameters]
        self.lr = lr
        self.mu = mu
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.turbo = turbo
        self.eps = eps

    def step(self):
        with torch.no_grad():
            for w, m in zip(self.parameters, self.momenta):
                g = w.grad
                if g is None:
                    continue
                m.lerp_(g, self.mu)
                if self.nesterov:
                    update_prepare = g.lerp(m, self.mu)
                else:
                    update_prepare = m
                rms_scale = 0.2 * math.sqrt(max(update_prepare.shape[-2], update_prepare.shape[-1]))
                update = newton_schulz(update_prepare.clone(), self.turbo, self.eps)
                w.sub_(self.lr * (rms_scale * update + w * self.weight_decay))

    def zero_grad(self):
        for w in self.parameters:
            w.grad = None

    def set_lr(self, lr: float):
        self.lr = lr


def muon(parameters, lr: float, mu: float, weight_decay: float, nesterov: bool, turbo: bool, eps: float) -> Muon:
    return Muon(parameters, lr, mu, weight_decay, nesterov, turbo, eps)


# ===== speckle/model/fourier_pe.py =====

TAU = 2.0 * math.pi


class PositionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        proj = torch.empty(in_dim, out_dim // 2).normal_(0.0, 2.0)
        self.register_buffer("proj", proj)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        stub = (pos * TAU) @ self.proj
        return torch.cat([stub.cos(), stub.sin()], -1)


# ===== speckle/model/norm.py =====

class Norm(nn.Module):
    def __init__(self, options: str, emb_dim: int):
        super().__init__()
        if options not in NORM_OPTIONS:
            raise ValueError(f"invalid norm: {options!r}")
        self.kind = options
        self.normalized_shape = (emb_dim,)
        self.weight = nn.Parameter(torch.ones(emb_dim))
        if options == "LayerNorm":
            self.bias = nn.Parameter(torch.zeros(emb_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        if self.kind == "LayerNorm":
            return F.layer_norm(xs, self.normalized_shape, self.weight, self.bias, 1e-5)
        return F.rms_norm(xs, self.normalized_shape, self.weight, 1e-5)


def norm(options: str, emb_dim: int) -> Norm:
    return Norm(options, emb_dim)


# ===== speckle/model/mhsa.py =====

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout: float, train: bool) -> torch.Tensor:
    att = (q @ k.transpose(-1, -2)).softmax(-1)
    att = F.dropout(att, dropout, train)
    return att @ v


class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, qk_norm: bool, dropout: float):
        super().__init__()
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_o = nn.Linear(emb_dim, emb_dim, bias=False)
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.vec_scale = (emb_dim / num_heads) ** -0.25
        self.qk_norm = qk_norm
        if qk_norm:
            self.k_norm_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(self, xs: torch.Tensor, train: bool) -> torch.Tensor:
        b, t, c = xs.shape

        q = (self.w_q(xs) * self.vec_scale).view(b, t, c // self.head_dim, self.head_dim).transpose(-2, -3)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,), None, 1e-5)

        k = (self.w_k(xs) * self.vec_scale).view(b, t, c // self.head_dim, self.head_dim).transpose(-2, -3)
        if self.qk_norm:
            k = F.rms_norm(k, (self.head_dim,), None, 1e-5) * self.k_norm_scale

        v = self.w_v(xs).view(b, t, c // self.head_dim, self.head_dim).transpose(-2, -3)

        out = attention(q, k, v, self.dropout, train).transpose(-2, -3).reshape(b, t, c)
        return F.dropout(self.w_o(out), self.dropout, train)


def self_attention(emb_dim: int, num_heads: int, qk_norm: bool, dropout: float) -> SelfAttention:
    return SelfAttention(emb_dim, num_heads, qk_norm, dropout)


# ===== speckle/model/swiglu.py =====

class SwiGLU(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.glu_proj = nn.Linear(emb_dim, 2 * hidden_dim)
        self.glu_out = nn.Linear(hidden_dim, emb_dim)
        self.dropout = dropout

    def forward(self, xs: torch.Tensor, train: bool) -> torch.Tensor:
        proj = self.glu_proj(xs)
        scale, gate = proj.chunk(2, dim=-1)
        gate = gate * gate.sigmoid()
        return F.dropout(self.glu_out(scale * gate), self.dropout, train)


def swiglu(emb_dim: int, hidden_dim: int, dropout: float) -> SwiGLU:
    return SwiGLU(emb_dim, hidden_dim, dropout)


# ===== speckle/model/block.py =====

class Block(nn.Module):
    def __init__(self, config: BlockConfig, emb_dim: int, dropout: float):
        super().__init__()
        self.sa_norm = norm(config.norm, emb_dim)
        self.self_attention = self_attention(emb_dim, config.attention.num_heads, config.attention.qk_norm, dropout)
        self.attention_residual_scale = nn.Parameter(torch.ones(1))
        self.st_norm = norm(config.norm, emb_dim)
        self.storage = swiglu(emb_dim, config.swiglu_dim, dropout)

    def forward(self, xs: torch.Tensor, train: bool) -> torch.Tensor:
        xs = self.self_attention(self.sa_norm(xs), train) * self.attention_residual_scale + xs
        return self.storage(self.st_norm(xs), train) + xs


def block(config: BlockConfig, emb_dim: int, dropout: float) -> Block:
    return Block(config, emb_dim, dropout)


# ===== speckle/utils/output.py =====

def flat_to_2d(flat: torch.Tensor, width: int) -> torch.Tensor:
    heights = torch.div(flat, width, rounding_mode="floor")
    widths = flat.remainder(width)
    return torch.stack([heights, widths], -1)


def patch_coords(centers: torch.Tensor, p_side: int, width: int) -> torch.Tensor:
    b, n, _ = centers.shape
    padding = p_side // 2
    padded_w = width + 2 * padding
    steps = torch.arange(p_side, dtype=centers.dtype, device=centers.device)
    dy = steps.view(-1, 1).expand(p_side, p_side)
    dx = steps.view(1, -1).expand(p_side, p_side)

    absolute_coords = centers.view(b, n, 1, 1, 2) + torch.stack([dy, dx], -1).view(1, 1, p_side, p_side, 2)
    return (absolute_coords[..., 0] * padded_w + absolute_coords[..., 1]).view(b, n * p_side * p_side)


def make_pictures(flat_coords: torch.Tensor, patches: torch.Tensor, scores: torch.Tensor, p_side: int, hwc):
    height, width, channels = hwc
    b, npp = flat_coords.shape
    padding = p_side // 2
    padded_h = height + 2 * padding
    padded_w = width + 2 * padding
    padded_hw = padded_h * padded_w

    flat_scores = scores.reshape(b, npp)
    pixel_max = torch.full(
        (b, padded_hw), float("-inf"), dtype=flat_scores.dtype, device=patches.device
    ).scatter_reduce_(1, flat_coords, flat_scores, reduce="amax", include_self=True)
    exp_scores = (flat_scores - pixel_max.gather(1, flat_coords)).exp()

    weighted_values = patches.reshape(b, npp, channels) * exp_scores.unsqueeze(-1)

    numerator = torch.zeros(b, padded_hw, channels, dtype=flat_scores.dtype, device=patches.device).scatter_add_(
        1, flat_coords.unsqueeze(-1).expand(b, npp, channels), weighted_values
    )
    denominator = torch.zeros(b, padded_hw, dtype=flat_scores.dtype, device=patches.device).scatter_add_(
        1, flat_coords, exp_scores
    )

    mask_has_data = (
        torch.zeros(b, padded_hw, dtype=torch.int64, device=patches.device)
        .scatter_(1, flat_coords, 1)
        .eq(1)
    )

    safe_denominator = torch.where(mask_has_data, denominator, torch.ones_like(denominator))
    prediction = (numerator / safe_denominator.view(b, padded_hw, 1)).view(b, padded_h, padded_w, channels)[
        :, padding : padding + height, padding : padding + width
    ]
    variances = (
        torch.exp(-pixel_max).unsqueeze(-1) / denominator.view(b, padded_hw, 1)
    ).view(b, padded_h, padded_w, 1)[:, padding : padding + height, padding : padding + width]
    mask = mask_has_data.view(b, padded_h, padded_w)[:, padding : padding + height, padding : padding + width]
    return prediction, variances, mask


# ===== speckle/utils/__init__.py =====

def train_val_split(data: torch.Tensor, train_share: float):
    n_total = data.shape[0]
    data = data.index_select(0, torch.randperm(n_total, device=data.device))
    n = int(train_share * n_total)
    return data[:n], data[n : n_total - 1]


def loss_to_double(config, batcher: Batcher, m, class_sizes) -> float:
    with torch.no_grad():
        pred = []
        limit = sum(class_sizes)
        device = next(m.parameters()).device
        for _ in range(config.eval_iters):
            batch = batcher.next()
            if batch is None:
                batcher.current_index = 0
                batch = batcher.next()
                if batch is None:
                    break
            sx, target = batch
            sx = sx[:, :limit].to(device)
            target = target.to(device)
            loss, _ = m.forward_with_loss(sx, target, False)
            pred.append(loss)
        return float(torch.stack(pred).mean())


def estimate_loss(config, train_batcher: Batcher, validation_batcher: Batcher, m, class_sizes):
    return (
        loss_to_double(config, train_batcher, m, class_sizes),
        loss_to_double(config, validation_batcher, m, class_sizes),
    )


# ===== speckle/utils/image.py =====

def merge_images(images, horizontally: bool) -> Image.Image:
    w, h = images[0].size
    if horizontally:
        canvas = Image.new("RGBA", (w * len(images), h))
        for i, img in enumerate(images):
            canvas.paste(img, (i * w, 0))
    else:
        canvas = Image.new("RGBA", (w, h * len(images)))
        for i, img in enumerate(images):
            canvas.paste(img, (0, i * h))
    return canvas


def tensor_to_rgba(image: torch.Tensor, active: torch.Tensor) -> Image.Image:
    h, w, c = image.shape
    image_flat = image.reshape(h * w, c).to(torch.uint8).cpu().numpy()
    active_flat = active.reshape(-1).to(torch.bool).cpu().numpy()

    rgba = np.zeros((h * w, 4), dtype=np.uint8)
    if c == 3:
        rgba[active_flat, 0:3] = image_flat[active_flat]
    else:
        rgba[active_flat, 0:3] = image_flat[active_flat, 0:1]
    rgba[active_flat, 3] = 255
    return Image.fromarray(rgba.reshape(h, w, 4), "RGBA")


def visualize_generation(visualization_classes, batcher, m):
    with torch.no_grad():
        batch = batcher.next()
        if batch is None:
            raise RuntimeError("visualization batcher is exhausted")
        sx, target = batch
        device = next(m.parameters()).device
        sx = sx.to(device)
        target = target.to(device)
        b, h, w, c = target.shape
        sx = sx[:, : sum(visualization_classes)]
        n = sx.shape[1]

        gathered = target.view(b, h * w, c).gather(1, sx.unsqueeze(-1).expand(b, n, c))
        context = (
            torch.zeros(b, h * w, c, device=target.device)
            .scatter(1, sx.unsqueeze(-1).expand(b, n, c), gathered)
            .view(b, h, w, c)
        )
        context_mask = torch.zeros(b, h * w, device=target.device).scatter(1, sx, 1).view(b, h, w)

        loss, raw_pred = m.forward_with_loss(sx, target, False)
        mu, logvar = m.mu_logvar(raw_pred, c)
        prediction, variances, generated_pixels_mask = make_pictures(
            patch_coords(flat_to_2d(sx, w), m.patch_side, w),
            mu,
            -logvar,
            m.patch_side,
            (h, w, c),
        )

        def batch_to_images(batch_t, active_mask):
            return [
                tensor_to_rgba(batch_t[i] * 255.0, active_mask[i]) for i in range(b)
            ]

        target_imgs = batch_to_images(target, generated_pixels_mask)
        context_imgs = batch_to_images(context, context_mask)
        pred_imgs = batch_to_images(prediction, generated_pixels_mask)
        var_imgs = batch_to_images(variances, generated_pixels_mask)
        grid = merge_images(
            [
                merge_images(target_imgs, True),
                merge_images(context_imgs, True),
                merge_images(pred_imgs, True),
                merge_images(var_imgs, True),
            ],
            False,
        )
        return float(loss.item()), grid


def visualize_source_points(images: torch.Tensor, sample_idx: int, source_points: int) -> Image.Image:
    single_img = images[sample_idx]
    if single_img.dtype == torch.uint8:
        single_img = single_img.float()
    else:
        single_img = single_img * 255.0
    h, w, c = single_img.shape
    class_ix = torch.randint(100000, (h * w,)).argsort(-1)
    source_ix = class_ix[:source_points]
    mask = torch.zeros(h * w).scatter(0, source_ix, 1).eq(1).view(h, w)
    return tensor_to_rgba(single_img, mask)


# ===== speckle/utils/info.py =====

def param_count(model: torch.nn.Module):
    # tch VarStore flags every variable trainable at creation; requires_grad_(false)
    # only freezes autograd, so Rust scheme.txt always reports trainable == total.
    total = sum(t.numel() for t in model.state_dict().values())
    trainable = total
    return total, trainable


def write_summary(model: torch.nn.Module, wr) -> None:
    grad_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    sorted_vars = sorted(model.state_dict().items())
    wr.write("-" * 85 + "\n")
    for name, tensor in sorted_vars:
        shape = str(list(tensor.shape))
        flag = "(trainable)" if grad_flags.get(name, False) else ""
        wr.write(f"{name:<60} {shape} {flag}\n".rstrip() + "\n")
    wr.write("-" * 85 + "\n")


# ===== speckle/model/__init__.py =====

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_side = config.patch_side
        if config.residual == "Attentive":
            self.block_ar_queries = nn.Parameter(torch.zeros(config.n_blocks, config.emb_dim))
        elif config.residual != "Plain":
            raise ValueError(f"invalid residual: {config.residual!r}")

        self.pos_enc = PositionEncoder(2, config.emb_dim // 2)
        self.embedding = nn.Linear(config.image_channels, config.emb_dim // 2, bias=False)
        self._blocks = []
        for i in range(config.n_blocks):
            blk = block(config.block, config.emb_dim, config.dropout)
            self.add_module(f"b{i}", blk)
            self._blocks.append(blk)
        self.ln_f = norm(config.block.norm, config.emb_dim)
        self.unembedding = nn.Linear(
            config.emb_dim, config.patch_side * config.patch_side * (1 + config.image_channels), bias=False
        )

    @property
    def blocks(self):
        return self._blocks

    def mu_logvar(self, raw_pred: torch.Tensor, c: int):
        b, n, _ = raw_pred.shape
        p2 = self.patch_side * self.patch_side
        logvariance = raw_pred[..., :p2]
        predictions = raw_pred[..., p2:].view(b, n, p2, c)
        return predictions, logvariance

    def forward_with_loss(self, sx: torch.Tensor, target: torch.Tensor, train: bool):
        b, h, w, c = target.shape
        n = sx.shape[1]
        lin = torch.linspace(-1.0, 1.0, h, dtype=target.dtype, device=target.device)
        x_grid = lin.unsqueeze(0).expand(h, w)
        y_grid = lin.unsqueeze(1).expand(h, w)
        pos = (
            torch.stack([y_grid, x_grid], -1)
            .view(h * w, 2)
            .index_select(0, sx.reshape(-1))
            .view(b, n, 2)
        )
        raw_predictions = self.forward_t(
            target.view(b, h * w, c).gather(-2, sx.unsqueeze(-1).expand(b, n, c)),
            pos,
            train,
        )
        mu, logvar = self.mu_logvar(raw_predictions, c)
        padding = self.patch_side // 2
        padded_h = h + 2 * padding
        padded_w = w + 2 * padding
        mask = torch.zeros(b, padded_h, padded_w, dtype=target.dtype, device=target.device)
        mask[:, 1 : h + 1, 1 : w + 1] = 1.0
        padded_target = torch.zeros(b, padded_h, padded_w, c, dtype=target.dtype, device=target.device)
        padded_target[:, 1 : h + 1, 1 : w + 1] = target

        flat_coords = patch_coords(flat_to_2d(sx, w), self.patch_side, w)
        patch_targets = (
            padded_target.view(b, padded_h * padded_w, c)
            .gather(-2, flat_coords.unsqueeze(-1).expand(b, n * self.patch_side * self.patch_side, c))
            .view(b, n, self.patch_side * self.patch_side, c)
        )
        patch_mask = (
            mask.view(b, padded_h * padded_w).gather(-1, flat_coords).view(b, n, self.patch_side * self.patch_side)
        )

        loss = (
            patch_mask.unsqueeze(-1)
            * ((patch_targets - mu).square() * (-logvar.unsqueeze(-1)).exp() + logvar.unsqueeze(-1))
        ).mean(dtype=torch.float32)
        return loss, raw_predictions

    def forward_t(self, context: torch.Tensor, positions: torch.Tensor, train: bool) -> torch.Tensor:
        context = context * 2 - 1
        context = (context + context.pow(5)) * 2
        xs = self.embedding(context)
        ps = self.pos_enc(positions)
        xs = torch.cat([ps, xs], -1)

        if not hasattr(self, "block_ar_queries"):
            for blk in self._blocks:
                xs = blk(xs, train)
            return self.unembedding(self.ln_f(xs))

        prefix_hs = xs.unsqueeze(-2)
        c_dim = xs.shape[-1]
        for i, blk in enumerate(self._blocks):
            norm_keys = F.rms_norm(prefix_hs, (c_dim,), None, 1e-7)
            hs = (
                (norm_keys @ self.block_ar_queries[i].view(-1))
                .softmax(-1)
                .unsqueeze(-2)
                .matmul(prefix_hs)
                .squeeze(-2)
            )
            xs = blk(hs, train)
            if i + 1 < len(self._blocks):
                prefix_hs = torch.cat([prefix_hs, xs.unsqueeze(-2)], -2)
        return self.unembedding(self.ln_f(xs))


# ===== speckle/dataset.py =====

HF_LINKS = {
    ("Mnist", True): ["https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet"],
    ("Mnist", False): ["https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet"],
    ("Cifar10", True): ["https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"],
    ("Cifar10", False): ["https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"],
    ("Stl10Unlabeled", True): [
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00000-of-00004-b74cd68890b47699.parquet",
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00001-of-00004-316ae046518ebbac.parquet",
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00002-of-00004-4180ec8df3024526.parquet",
        "https://huggingface.co/datasets/jxie/stl10/resolve/main/data/unlabeled-00003-of-00004-15264226b952a419.parquet",
    ],
}
HF_LINKS[("Stl10Unlabeled", False)] = HF_LINKS[("Stl10Unlabeled", True)]

IMAGE_COLUMNS = {"Mnist": "image", "Cifar10": "img", "Stl10Unlabeled": "image"}
FILENAMES = {"Mnist": "mnist", "Cifar10": "cifar10", "Stl10Unlabeled": "stl10_unlabeled"}
HWC = {
    "Mnist": (28, 28, 1),
    "Cifar10": (32, 32, 3),
    "SplitLine": (64, 64, 1),
    "SmoothedSplitLine": (64, 64, 1),
    "Stl10Unlabeled": (96, 96, 3),
}


def hwc(dataset: str):
    return HWC[dataset]


def employ(cfg: DatasetConfig, train: bool) -> torch.Tensor:
    dataset = cfg.dataset
    if dataset in ("SplitLine", "SmoothedSplitLine"):
        images = generate(dataset, train)
        if cfg.max_images > 0:
            images = images[: cfg.max_images]
        return images
    return load_or_download(dataset, train, cfg.max_images)


def generate(dataset: str, train: bool) -> torch.Tensor:
    if dataset == "SplitLine":
        lin = torch.linspace(-1.0, 1.0, 64)
        x_grid = lin.unsqueeze(0).expand(64, 64).unsqueeze(0).unsqueeze(-1)
        y_grid = lin.unsqueeze(1).expand(64, 64).unsqueeze(0).unsqueeze(-1)
        offset = torch.rand(100000, 1, 1, 1) - 0.5
        tilt = torch.rand(100000, 1, 1, 1) * math.pi
        a = tilt.cos()
        b = tilt.sin()
        distances = a * x_grid + b * y_grid + offset
        return (distances >= 0).to(torch.uint8)
    raise NotImplementedError(f"dataset {dataset!r} generation is not implemented (todo in Rust source)")


def _download(url: str, file_path: Path, attempts: int = 5) -> None:
    tmp_path = file_path.with_suffix(file_path.suffix + ".part")
    for attempt in range(attempts):
        resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
        request = urllib.request.Request(url)
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")
        try:
            with urllib.request.urlopen(request) as response:
                resume = getattr(response, "status", 200) == 206 and resume_from > 0
                content_length = int(response.headers.get("Content-Length") or 0)
                with open(tmp_path, "ab" if resume else "wb") as f:
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
            expected = resume_from + content_length if resume else content_length
            if expected and tmp_path.stat().st_size != expected:
                raise IOError(f"incomplete download: {tmp_path.stat().st_size} of {expected} bytes")
            tmp_path.rename(file_path)
            return
        except Exception as error:
            print(f"download interrupted ({error}); retrying ({attempt + 1}/{attempts})")
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to download {url} after {attempts} attempts")


def _decode(table, column: str, h: int, w: int, c: int, max_rows: int = -1) -> torch.Tensor:
    image_struct = table.column(column).combine_chunks()
    bytes_col = image_struct.field("bytes")
    n = table.num_rows if max_rows <= 0 else min(max_rows, table.num_rows)
    convert_mode = "L" if c == 1 else "RGB"
    all_images = np.empty((n, h, w, c), dtype=np.uint8)
    for i in range(n):
        img = Image.open(io.BytesIO(bytes_col[i].as_py())).convert(convert_mode)
        arr = np.asarray(img, dtype=np.uint8)
        if c == 1:
            arr = arr[:, :, None]
        all_images[i] = arr
    return torch.from_numpy(all_images)


def load_or_download(dataset: str, train: bool, max_images: int = -1) -> torch.Tensor:
    h, w, c = HWC[dataset]
    column = IMAGE_COLUMNS[dataset]
    split_dir = Path("data") / ("train" if train else "test")
    stem = FILENAMES[dataset]
    urls = HF_LINKS[(dataset, train)]
    cap = max_images if max_images > 0 else None

    full_npy = split_dir / f"{stem}_full.npy"
    capped_npy = split_dir / f"{stem}_{cap}.npy" if cap is not None else full_npy
    if capped_npy.exists():
        print(f"Found decoded dataset cache at: {capped_npy}")
        return torch.from_numpy(np.load(capped_npy))
    if cap is not None and full_npy.exists():
        print(f"Found decoded dataset cache at: {full_npy}")
        return torch.from_numpy(np.load(full_npy))[:cap]

    chunks = []
    done = 0
    for i, url in enumerate(urls):
        if cap is not None and done >= cap:
            break
        file_path = split_dir / f"{stem}.parquet" if len(urls) == 1 else split_dir / stem / f"{i:05d}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            print(f"Found cached shard at: {file_path}")
        else:
            print(f"Downloading dataset from Hugging Face: {url}")
            _download(url, file_path)
        table = pq.read_table(file_path, columns=[column])
        want = -1 if cap is None else cap - done
        chunk = _decode(table, column, h, w, c, want)
        chunks.append(chunk)
        done += chunk.shape[0]
    images = torch.cat(chunks)
    if len(urls) > 1:
        np.save(full_npy if cap is None else capped_npy, images.numpy())
    return images


# ===== speckle/cli.py =====

OVERRIDE_MAP = {
    "eval_interval": ("", "eval_interval"),
    "eval_iters": ("", "eval_iters"),
    "accumulation_steps": ("", "accumulation_steps"),
    "starting_density": ("", "starting_density"),
    "ending_density": ("", "ending_density"),
    "max_iters": ("lr_schedule", "max_iters"),
    "max_lr": ("lr_schedule", "max_lr"),
    "min_lr_share": ("lr_schedule", "min_lr_share"),
    "lr_decay": ("lr_schedule", "lr_decay"),
    "warmup_iters": ("lr_schedule", "warmup_iters"),
    "max_coast_iters": ("lr_schedule", "max_coast_iters"),
    "dropout": ("model", "dropout"),
    "id_loss_scale": ("model", "id_loss_scale"),
    "image_channels": ("model", "image_channels"),
    "n_blocks": ("model", "n_blocks"),
    "patch_side": ("model", "patch_side"),
    "emb_dim": ("model", "emb_dim"),
    "batch_size": ("model", "batch_size"),
    "eval_batch_size": ("model", "eval_batch_size"),
    "residual": ("model", "residual"),
    "norm": ("model.block", "norm"),
    "swiglu_dim": ("model.block", "swiglu_dim"),
    "num_heads": ("model.block.attention", "num_heads"),
    "qk_norm": ("model.block.attention", "qk_norm"),
    "train_share": ("dataset", "train_share"),
    "max_images": ("dataset", "max_images"),
    "dataset": ("dataset", "dataset"),
}


def add_train_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--eval-iters", type=int, default=None)
    p.add_argument("--accumulation-steps", type=int, default=None)
    p.add_argument("--starting-density", type=float, default=None)
    p.add_argument("--ending-density", type=float, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--max-lr", type=float, default=None)
    p.add_argument("--min-lr-share", type=float, default=None)
    p.add_argument("--lr-decay", default=None, choices=DECAY_MODES)
    p.add_argument("--warmup-iters", type=int, default=None)
    p.add_argument("--max-coast-iters", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--id-loss-scale", type=float, default=None)
    p.add_argument("--image-channels", type=int, default=None)
    p.add_argument("--n-blocks", type=int, default=None)
    p.add_argument("--patch-side", type=int, default=None)
    p.add_argument("--emb-dim", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--residual", default=None, choices=RESIDUAL_OPTIONS)
    p.add_argument("--norm", default=None, choices=NORM_OPTIONS)
    p.add_argument("--swiglu-dim", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--train-share", type=float, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--dataset", default=None, choices=DATASETS)


def apply_overrides(cfg: TrainConfig, ns: argparse.Namespace) -> None:
    for key, val in vars(ns).items():
        if key in OVERRIDE_MAP and val is not None:
            section, attr = OVERRIDE_MAP[key]
            target = cfg
            if section:
                for part in section.split("."):
                    target = getattr(target, part)
            setattr(target, attr, val)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="speckle")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    sub = p.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train", help="train a model")
    tr.add_argument("--config", type=Path, default=None, help="TOML config file")
    tr.add_argument("--tag", default=None)
    add_train_flags(tr)

    ev = sub.add_parser("eval", help="load a checkpoint")
    ev.add_argument("checkpoint", type=Path)

    vi = sub.add_parser("vis", help="visualize source point selection")
    vi.add_argument("--dataset", default="Mnist", choices=DATASETS)
    vi.add_argument("--train-share", type=float, default=0.9)
    vi.add_argument("--max-images", type=int, default=None)
    vi.add_argument("--source-points", type=int, default=192)

    return p.parse_args(argv)


# ===== speckle/__main__.py =====

def save_safetensors(model: torch.nn.Module, path: Path) -> None:
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(path))


def load_safetensors(model: torch.nn.Module, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    model.load_state_dict(state)


def split_parameters(model: torch.nn.Module):
    adam_p, muon_p = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() == 1 or "embedding" in name or "bias" in name:
                adam_p.append(p)
            else:
                muon_p.append(p)
    return adam_p, muon_p


def train(args, device: torch.device) -> None:
    config = TrainConfig.load(args.config) if args.config is not None else TrainConfig()
    apply_overrides(config, args)

    images = employ(config.dataset, True)
    h, w, c = hwc(config.dataset.dataset)

    train_data, val_data = train_val_split(images, config.dataset.train_share)

    base_seed = getattr(args, "seed", 1337)
    train_batcher = Batcher(train_data, config.model.batch_size, seed=base_seed)
    val_batcher = Batcher(val_data, config.model.batch_size, seed=base_seed + 1)
    visualization_classes = [256]
    train_vis_batcher = Batcher(train_data, config.model.eval_batch_size, seed=base_seed + 2)
    val_vis_batcher = Batcher(val_data, config.model.eval_batch_size, seed=base_seed + 3)

    model_creation_start = time.time()
    config.model.max_image_side = max(h, w)
    config.model.image_channels = c
    model = Model(config.model).to(device)
    print(f"model created in {time.time() - model_creation_start:.4f}s")

    adam_p, muon_p = split_parameters(model)
    adamw = torch.optim.AdamW(
        [{"params": adam_p, "lr": 0.0}], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    muon_opt = muon(muon_p, 0.0, 0.95, 0.0, True, True, 1e-7)

    log_dir = Path("checkpoints") / (
        f"run_{datetime.now():%Y%m%d_%H%M}_py" + (f"_{args.tag}" if args.tag else "")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "scheme.txt", "w") as scheme:
        total_params, trainable_params = param_count(model)
        scheme.write(f"total parameters: {total_params}\ntrainable parameters: {trainable_params}\n")
        write_summary(model, scheme)
    print(f"total parameters: {total_params}\ntrainable parameters: {trainable_params}")

    config.save(log_dir / "config.toml")

    with open(log_dir / "losses.csv", "w", newline="") as csv_file:
        wtr = csv.writer(csv_file)
        wtr.writerow(["step", "train_loss_pred", "val_loss_pred", "time_from_start"])
        start = time.time()
        adamw.zero_grad()
        muon_opt.zero_grad()

        val_step_size = [192]
        total_steps = config.lr_schedule.max_iters * config.accumulation_steps
        for step in range(total_steps):
            opt_step = step // config.accumulation_steps
            if opt_step % config.eval_interval == 0 and step % config.accumulation_steps == 0:
                eval_start = time.time()
                train_stat, val_stat = estimate_loss(
                    config, train_batcher, val_batcher, model, val_step_size
                )
                train_pred_loss, train_img = visualize_generation(
                    visualization_classes, train_vis_batcher, model
                )
                train_img.save(log_dir / f"train_step{opt_step}_pred{train_pred_loss:.4f}.png")
                val_pred_loss, val_img = visualize_generation(
                    visualization_classes, val_vis_batcher, model
                )
                val_img.save(log_dir / f"val_step{opt_step}_pred{val_pred_loss:.4f}.png")
                eval_end = time.time()
                dt = eval_end - start
                print(
                    f"step: {opt_step}, train loss: [p_i: {train_stat:.4f}], "
                    f"val loss: [p_i: {val_stat:.4f}], time from start {dt:.4f}s, "
                    f"{config.eval_iters} eval iterations done in {eval_end - eval_start:.4f}s"
                )
                wtr.writerow([opt_step, train_stat, val_stat, f"{dt}s"])
                csv_file.flush()

            batch = train_batcher.next()
            if batch is None:
                break
            sx, target = batch
            density = config.ending_density + (
                (config.starting_density - config.ending_density)
                * 0.5
                * (1.0 + math.cos(math.pi * step / total_steps))
            )
            n = int(density * (h * w))
            sx = sx[:, :n].to(device)
            target = target.to(device)
            loss, _ = model.forward_with_loss(sx, target, True)
            (loss / config.accumulation_steps).backward()
            if step % config.accumulation_steps == 0:
                lr = config.lr_schedule.get_lr(opt_step)
                adamw.param_groups[0]["lr"] = lr
                muon_opt.set_lr(7.0 * lr)
                adamw.step()
                muon_opt.step()
                adamw.zero_grad()
                muon_opt.zero_grad()

        print(f"elapsed time: {time.time() - start:.4f}s")

    save_safetensors(model, log_dir / "model.safetensors")


def eval_checkpoint(args, device: torch.device) -> None:
    checkpoint = args.checkpoint
    config = TrainConfig.load(checkpoint / "config.toml").model
    model = Model(config).to(device)
    load_safetensors(model, checkpoint / "model.safetensors")
    total_params, trainable_params = param_count(model)
    print(f"total parameters: {total_params}\ntrainable parameters: {trainable_params}")
    write_summary(model, sys.stdout)


def vis(args, device: torch.device) -> None:
    max_images = getattr(args, "max_images", None)
    cfg = DatasetConfig(
        train_share=args.train_share,
        dataset=args.dataset,
        max_images=-1 if max_images is None else max_images,
    )
    images = employ(cfg, True)
    img = visualize_source_points(images, args.seed, args.source_points)
    out_path = f"vis_sample{args.seed}.png"
    img.save(out_path)
    print(f"saved {out_path}")


# =====================================================================================================
# JUPYTER NOTEBOOK CONFIG CELL
# Everything above is library code (verbatim concatenation of the speckle/ package).
# Paste the block below into a notebook cell (or run export.py directly) to configure
# and launch a run. It writes a TOML artifact for the run, then hands it to the
# verbatim `train()`, which reloads it and copies it into the run directory
# (`checkpoints/run_*/config.toml`), so the artifact always matches what was trained.
# =====================================================================================================

# %pip install -q torch safetensors pyarrow pillow numpy

seed = 1337
device = torch.device("cpu")  # use torch.device("cuda") on GPU
tag = None  # or a string, appended to the run directory name

# --- per-run parameters: edit freely, every TrainConfig field is available ---
config = TrainConfig()
config.dataset.dataset = "Stl10Unlabeled"  # jxie/stl10 `unlabeled` split: 100k 96x96 RGB images
config.dataset.max_images = -1             # subsample cap (-1 = all 100k; e.g. 2000 for a quick local smoke)
config.dataset.train_share = 0.9
config.model.emb_dim = 32
config.model.n_blocks = 1
config.model.batch_size = 4
config.lr_schedule.max_iters = 101
config.eval_interval = 50
config.eval_iters = 50
config.starting_density = 0.25             # at 96x96 one point = 1 of 9216 pixels; 0.5 -> 4608 tokens/image
config.ending_density = 0.05
# config.model.block.norm = "RMSNorm"      # "LayerNorm" | "RMSNorm"
# config.model.residual = "Attentive"      # "Plain" | "Attentive"
# config.model.block.attention.qk_norm = True
# config.model.block.attention.num_heads = 4
# config.model.dropout = 0.2
# config.model.patch_side = 3
# config.lr_schedule.max_lr = 3e-3
# config.accumulation_steps = 4

# --- write the TOML artifact for this run (one timestamped file, never overwritten) ---
config_path = Path(f"jupyter_config_{datetime.now():%Y%m%d_%H%M%S}.toml")
config.save(config_path)
print(f"saved {config_path}")

# --- launch training (equivalent of the CLI `main()`) ---
args = SimpleNamespace(config=config_path, tag=tag, seed=seed)
torch.manual_seed(seed)
train(args, device)

# --- alternatives (uncomment to use) ---
# eval_checkpoint(SimpleNamespace(checkpoint=Path("checkpoints/run_YYYYmmdd_HHMM_py_TAG")), device)
# vis(SimpleNamespace(dataset="Stl10Unlabeled", train_share=0.9, max_images=500, seed=42, source_points=192), device)

# Kaggle notes:
# - first run downloads 4 parquet shards (~1.77 GB total) into data/train/stl10_unlabeled/
#   and decodes them to data/train/stl10_unlabeled_full.npy (~2.77 GB); later runs start
#   instantly from the .npy cache. Re-attach a previous session's working dir as a Kaggle
#   dataset to skip the download; --max-images caps download+decode to the needed shards.
# - `unlabeled` has no test split; val comes from train_val_split over the same pool.
# - the eval/vis loop still uses 192 source points (val_step_size), a Rust-inherited
#   constant tuned for 32x32/28x28 images; scale it up for 96x96 if desired.
