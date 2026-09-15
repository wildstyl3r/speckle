import torch
from torch import nn
from torch.nn import functional as F


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout: float, train: bool) -> torch.Tensor:
    att = (q @ k.transpose(-1, -2)).softmax(-1)
    att = F.dropout(att, dropout, train)
    return att @ v


class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, qk_norm: bool, dropout: float):
        super().__init__()
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_o = nn.Linear(emb_dim, emb_dim, bias=False)
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.vec_scale = (emb_dim / num_heads) ** -0.25
        self.qk_norm = qk_norm
        if qk_norm:
            self.k_norm_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(self, xs: torch.Tensor, train: bool) -> torch.Tensor:
        b, t, c = xs.shape

        q = (self.w_q(xs) * self.vec_scale).view(b, t, c // self.head_dim, self.head_dim).transpose(-2, -3)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,), None, 1e-5)

        k = (self.w_k(xs) * self.vec_scale).view(b, t, c // self.head_dim, self.head_dim).transpose(-2, -3)
        if self.qk_norm:
            k = F.rms_norm(k, (self.head_dim,), None, 1e-5) * self.k_norm_scale

        v = self.w_v(xs).view(b, t, c // self.head_dim, self.head_dim).transpose(-2, -3)

        out = attention(q, k, v, self.dropout, train).transpose(-2, -3).reshape(b, t, c)
        return F.dropout(self.w_o(out), self.dropout, train)


def self_attention(emb_dim: int, num_heads: int, qk_norm: bool, dropout: float) -> SelfAttention:
    return SelfAttention(emb_dim, num_heads, qk_norm, dropout)
