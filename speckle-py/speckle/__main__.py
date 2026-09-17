import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import dataset
from .batcher import Batcher
from .cli import apply_overrides, parse_args
from .config import DatasetConfig, TrainConfig
from .model import Model
from .muon import muon
from .utils import estimate_loss, train_val_split
from .utils.image import visualize_generation, visualize_source_points
from .utils.info import param_count, write_summary


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

    images = dataset.employ(config.dataset, True)
    h, w, c = dataset.hwc(config.dataset.dataset)

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
    images = dataset.employ(cfg, True)
    img = visualize_source_points(images, args.seed, args.source_points)
    out_path = f"vis_sample{args.seed}.png"
    img.save(out_path)
    print(f"saved {out_path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.mode == "train":
        train(args, device)
    elif args.mode == "eval":
        eval_checkpoint(args, device)
    elif args.mode == "vis":
        vis(args, device)


if __name__ == "__main__":
    main()
