import dataclasses
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
