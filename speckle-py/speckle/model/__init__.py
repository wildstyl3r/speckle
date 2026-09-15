import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from ..utils.output import patch_coords
from ..utils import flat_to_2d
from .block import Block, block
from .fourier_pe import PositionEncoder
from .norm import Norm, norm


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
