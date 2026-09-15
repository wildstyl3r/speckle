import torch
from torch import nn

from ..config import BlockConfig
from .mhsa import SelfAttention, self_attention
from .norm import Norm, norm
from .swiglu import SwiGLU, swiglu


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
