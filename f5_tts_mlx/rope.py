from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from einops.array_api import rearrange


# rotary positional embedding related


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.scale = None
            return

        scale = (mx.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.scale = scale

    def forward_from_seq_len(self, seq_len: int) -> tuple[mx.array, float]:
        t = mx.arange(seq_len)
        return self(t)

    def __call__(self, t: mx.array) -> tuple[mx.array, float]:
        max_pos = t.max() + 1

        freqs = (
            mx.einsum("i , j -> i j", t.astype(self.inv_freq.dtype), self.inv_freq)
            / self.interpolation_factor
        )
        freqs = mx.stack((freqs, freqs), axis=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if self.scale is None:
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = mx.stack((scale, scale), axis=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0
):
    freqs = 1.0 / (
        theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim)
    )
    t = mx.arange(end)
    freqs = mx.outer(t, freqs).astype(mx.float32)
    freqs_cos = freqs.cos()  # real part
    freqs_sin = freqs.sin()  # imaginary part
    return mx.concatenate([freqs_cos, freqs_sin], axis=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * mx.ones_like(start)
    pos = mx.expand_dims(start, axis=1) + (
        mx.expand_dims(mx.arange(length), axis=0) * mx.expand_dims(scale, axis=1)
    ).astype(mx.int32)
    # avoid extra long error.
    pos = mx.where(pos < max_pos, pos, max_pos - 1)
    return pos


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = [mx.squeeze(s, axis=-1) for s in mx.split(x, x.shape[-1], axis=-1)]
    x = mx.stack([-x2, x1], axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]

    freqs = freqs[-seq_len:, :]
    scale = scale[-seq_len:, :] if isinstance(scale, mx.array) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = mx.concatenate((t, t_unrotated), axis=-1)

    return out
