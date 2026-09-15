import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speckle import dataset
from speckle.batcher import Batcher
from speckle.config import TrainConfig
from speckle.model import Model
from speckle.utils.info import param_count
from speckle.__main__ import load_safetensors

RUST_CKPT = Path(
    "../../Programming/Rust/speckle/checkpoints/run_20260913_1929_-dirty_ff-cifar-b3-h4"
)
EXPECTED_VAL = -3.5280861854553223
EXPECTED_TRAIN = -3.57356333732605
TOLERANCE = 0.2


def main():
    config = TrainConfig.load(RUST_CKPT / "config.toml")
    model = Model(config.model)
    load_safetensors(model, RUST_CKPT / "model.safetensors")
    total, trainable = param_count(model)
    print(f"loaded rust checkpoint: total={total} trainable={trainable}")

    scheme = (RUST_CKPT / "scheme.txt").read_text()
    rust_total = int(scheme.split("total parameters: ")[1].split("\n")[0])
    rust_trainable = int(scheme.split("trainable parameters: ")[1].split("\n")[0])
    assert (total, trainable) == (rust_total, rust_trainable), (total, trainable, rust_total, rust_trainable)
    print(f"param counts match rust scheme.txt ({total} total, {trainable} trainable)")

    cfg = config.dataset
    images = dataset.employ(cfg, True)
    h, w, c = dataset.hwc(cfg.dataset)
    assert images.shape[1:] == (h, w, c)
    n = int(cfg.train_share * images.shape[0])
    train_data, val_data = images[:n], images[n : images.shape[0] - 1]

    eval_iters = config.eval_iters
    with torch.no_grad():
        val_batcher = Batcher(val_data, config.model.eval_batch_size)
        losses = []
        for _ in range(eval_iters):
            batch = val_batcher.next()
            if batch is None:
                break
            sx, target = batch
            loss, _ = model.forward_with_loss(sx[:, :192], target, False)
            losses.append(loss)
        val_loss = float(torch.stack(losses).mean())
        print(f"python val loss over {len(losses)} batches: {val_loss:.4f} (rust: {EXPECTED_VAL:.4f})")
        assert abs(val_loss - EXPECTED_VAL) < TOLERANCE, val_loss

        train_batcher = Batcher(train_data, config.model.eval_batch_size)
        losses = []
        for _ in range(eval_iters):
            batch = train_batcher.next()
            if batch is None:
                break
            sx, target = batch
            loss, _ = model.forward_with_loss(sx[:, :192], target, False)
            losses.append(loss)
        train_loss = float(torch.stack(losses).mean())
        print(f"python train loss over {len(losses)} batches: {train_loss:.4f} (rust: {EXPECTED_TRAIN:.4f})")
        assert abs(train_loss - EXPECTED_TRAIN) < TOLERANCE, train_loss

    print("INTEROP TEST PASSED")


if __name__ == "__main__":
    main()
