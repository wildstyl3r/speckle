import torch
from torch import nn
from torch.nn import functional as F


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
