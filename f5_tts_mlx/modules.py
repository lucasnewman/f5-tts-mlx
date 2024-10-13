from __future__ import annotations
from functools import lru_cache
import math
import os
from typing import Union

import mlx.core as mx
import mlx.nn as nn

import numpy as np

from einops.array_api import rearrange, repeat


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

        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq

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
    t = mx.arange(end)  # type: ignore
    freqs = mx.outer(t, freqs).astype(mx.float32)  # type: ignore
    freqs_cos = freqs.cos()  # real part
    freqs_sin = freqs.sin()  # imaginary part
    return mx.concatenate([freqs_cos, freqs_sin], axis=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * mx.ones_like(start).astype(mx.float32)  # in case scale is a scalar
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


# mel spectrogram


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Saved using extract_filterbank.py
    """
    assert n_mels in {100}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join("assets", "mel_filters.npz")
    return mx.load(filename, format="npz")[f"mel_{n_mels}"]


@lru_cache(maxsize=None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


def stft(x, window, nperseg=256, noverlap=None, nfft=None, pad_mode="constant"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: Union[mx.array, np.ndarray],
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
    filterbank: mx.array | None = None,
):
    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    freqs = stft(audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
    magnitudes = freqs[:-1, :].abs()
    filters = filterbank if filterbank is not None else mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0)


class MelSpec(nn.Module):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
        filterbank: mx.array | None = None,
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.sample_rate = sample_rate
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.filterbank = filterbank

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        return log_mel_spectrogram(
            audio,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding=0,
            filterbank=self.filterbank,
        )


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = scale * mx.expand_dims(x, axis=1) * mx.expand_dims(emb, axis=0)
        emb = mx.concatenate([emb.sin(), emb.cos()], axis=-1)
        return emb


# convolutional position embedding


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: mx.array) -> mx.array:
        return rearrange(x, self.pattern)


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        if mask is not None:
            mask = mask[..., None]
            x = x * mask

        out = self.conv1d(x)

        if mask is not None:
            out = out * mask

        return out


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
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
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


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array | None = None) -> mx.array:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            emb, 6, axis=1
        )

        x = self.norm(x) * (1 + mx.expand_dims(scale_msa, axis=1)) + mx.expand_dims(
            shift_msa, axis=1
        )
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array | None = None) -> mx.array:
        emb = self.linear(self.silu(emb))
        scale, shift = mx.split(emb, 2, axis=1)

        x = self.norm(x) * (1 + mx.expand_dims(scale, axis=1)) + mx.expand_dims(
            shift, axis=1
        )
        return x


# feed forward


class FeedForward(nn.Module):
    def __init__(
        self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approx=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff(x)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        rope: mx.array | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (
                    xpos_scale,
                    xpos_scale**-1.0,
                )
                if xpos_scale is not None
                else (1.0, 1.0)
            )

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        query = rearrange(query, "b n (h d) -> b h n d", h=self.heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.heads)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = rearrange(attn_mask, "b n -> b () () n")
            attn_mask = repeat(attn_mask, "b () () n -> b h () n", h=self.heads)
        else:
            attn_mask = None

        scale_factor = 1 / math.sqrt(query.shape[-1])

        x = mx.fast.scaled_dot_product_attention(
            q=query, k=key, v=value, scale=scale_factor, mask=attn_mask
        )
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1).astype(query.dtype)

        # linear proj
        x = self.to_out(x)

        if attn_mask is not None:
            mask = rearrange(mask, "b n -> b n 1")
            x = x.masked_fill(~mask, 0.0)

        return x


# DiT block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh"
        )

    def __call__(
        self, x, t, mask=None, rope=None
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + mx.expand_dims(gate_msa, axis=1) * attn_output

        norm = self.ff_norm(x) * (
            1 + mx.expand_dims(scale_mlp, axis=1)
        ) + mx.expand_dims(shift_mlp, axis=1)
        ff_output = self.ff(norm)
        x = x + mx.expand_dims(gate_mlp, axis=1) * ff_output

        return x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def __call__(self, timestep: mx.array) -> mx.array:
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)  # b d
        return time
