import argparse
from pathlib import Path

from .config import (
    DATASETS,
    DECAY_MODES,
    NORM_OPTIONS,
    RESIDUAL_OPTIONS,
    TrainConfig,
)


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
