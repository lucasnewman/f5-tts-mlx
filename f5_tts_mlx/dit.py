"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
import math

import mlx.core as mx
import mlx.nn as nn

from einops.array_api import rearrange, repeat

from f5_tts_mlx.convnext_v2 import ConvNeXtV2Block
from f5_tts_mlx.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

# convolutional position embedding


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
        time = self.time_mlp(time_hidden)
        return time


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

        scale_factor = 1 / mx.sqrt(query.shape[-1])

        x = mx.fast.scaled_dot_product_attention(
            q=query, k=key, v=value, scale=scale_factor, mask=attn_mask
        )
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1).astype(query.dtype)

        # linear proj
        x = self.to_out(x)

        if attn_mask is not None:
            mask = rearrange(mask, "b n -> b n 1")
            x = mx.where(mask, x, 0.0)

        return x


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(
            text_num_embeds + 1, text_dim
        )  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self._freqs_cis = precompute_freqs_cis(text_dim, self.precompute_max_pos)
            self.text_blocks = nn.Sequential(
                *[
                    ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                    for _ in range(conv_layers)
                ]
            )
        else:
            self.extra_modeling = False

    def __call__(self, text, seq_len, drop_text=False):
        batch, text_len = text.shape[0], text.shape[1]

        # use 0 as filler token. we rely on text being padded with -1 values.
        text = text + 1

        # curtail if character tokens are more than the mel spec tokens
        text = text[:, :seq_len]

        text = mx.pad(text, [(0, 0), (0, seq_len - text_len)], constant_values=0)

        # cfg for text
        text = mx.where(drop_text, mx.zeros_like(text), text)
        text = self.text_embed(text)  # b n -> b n d

        if self.extra_modeling:
            # sinus pos emb
            batch_start = mx.zeros((batch,), dtype=mx.int32)
            pos_idx = get_pos_embed_indices(
                batch_start, seq_len, max_pos=self.precompute_max_pos
            )
            text_pos_embed = self._freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnext v2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: mx.array,  # b n d
        cond: mx.array,  # b n d
        text_embed: mx.array,  # b n d
        drop_audio_cond=False,
    ):
        # cfg for cond audio
        cond = mx.where(drop_audio_cond, mx.zeros_like(cond), cond)
        x = self.proj(mx.concatenate((x, cond, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


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


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.0,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, conv_layers=conv_layers
        )
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = [
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ]

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def __call__(
        self,
        x: mx.array,  # b n d, nosied input audio
        cond: mx.array,  # b n d, masked cond audio
        text: mx.array,  # b nt, text
        time: mx.array,  # b, time step
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: mx.array | None = None,  # b n
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = repeat(time, " -> b", b=batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
