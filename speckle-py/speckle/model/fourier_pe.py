import math

import torch
from torch import nn

TAU = 2.0 * math.pi


class PositionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        proj = torch.empty(in_dim, out_dim // 2).normal_(0.0, 2.0)
        self.register_buffer("proj", proj)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        stub = (pos * TAU) @ self.proj
        return torch.cat([stub.cos(), stub.sin()], -1)
