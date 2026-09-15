import math

import torch

A = 3.4445
B = -4.775
C = 2.0315


def newton_schulz(x: torch.Tensor, turbo: bool, eps: float) -> torch.Tensor:
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.transpose(-1, -2)

    if turbo:
        a = x @ x.transpose(-1, -2)
        s = a.abs().sum(-1).clamp_min(eps).rsqrt()
        a = a * s.unsqueeze(-2) * s.unsqueeze(-1)
        x = x * s.unsqueeze(-1)
        iters = 4
    else:
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
        a = x @ x.transpose(-1, -2)
        iters = 5

    for i in range(iters):
        if i != 0:
            a = x @ x.transpose(-1, -2)
        b = B * a + C * (a @ a)
        x = A * x + b @ x

    if tall:
        x = x.transpose(-1, -2)
    return x


class Muon:
    def __init__(self, parameters, lr: float, mu: float, weight_decay: float, nesterov: bool, turbo: bool, eps: float):
        self.parameters = list(parameters)
        self.momenta = [torch.zeros_like(w) for w in self.parameters]
        self.lr = lr
        self.mu = mu
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.turbo = turbo
        self.eps = eps

    def step(self):
        with torch.no_grad():
            for w, m in zip(self.parameters, self.momenta):
                g = w.grad
                if g is None:
                    continue
                m.lerp_(g, self.mu)
                if self.nesterov:
                    update_prepare = g.lerp(m, self.mu)
                else:
                    update_prepare = m
                rms_scale = 0.2 * math.sqrt(max(update_prepare.shape[-2], update_prepare.shape[-1]))
                update = newton_schulz(update_prepare.clone(), self.turbo, self.eps)
                w.sub_(self.lr * (rms_scale * update + w * self.weight_decay))

    def zero_grad(self):
        for w in self.parameters:
            w.grad = None

    def set_lr(self, lr: float):
        self.lr = lr


def muon(parameters, lr: float, mu: float, weight_decay: float, nesterov: bool, turbo: bool, eps: float) -> Muon:
    return Muon(parameters, lr, mu, weight_decay, nesterov, turbo, eps)
