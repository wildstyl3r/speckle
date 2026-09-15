import torch
from torch import nn
from torch.nn import functional as F

from ..config import NORM_OPTIONS


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
