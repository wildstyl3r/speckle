import torch

from ..batcher import Batcher
from .output import flat_to_2d, make_pictures, patch_coords


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
