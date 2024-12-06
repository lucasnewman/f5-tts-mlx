from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# global response normalization


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim))
        self.beta = mx.zeros((1, 1, dim))

    def __call__(self, x):
        Gx = mx.linalg.norm(x, ord=2, axis=1, keepdims=True)
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-v2 block


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        
        # depthwise conv
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x
