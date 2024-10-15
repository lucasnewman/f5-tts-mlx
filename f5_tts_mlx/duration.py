from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from einops.array_api import rearrange, repeat
import einx

from f5_tts_mlx.cfm import (
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    maybe_masked_mean,
)
from f5_tts_mlx.dit import DiT, TextEmbedding, TimestepEmbedding, ConvPositionEmbedding

from f5_tts_mlx.modules import (
    MelSpec,
    RotaryEmbedding,
    DiTBlock,
)

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: mx.array) -> mx.array:
        return rearrange(x, self.pattern)


class DurationInputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: float["b n d"],
        text_embed: float["b n d"],
    ):
        x = self.proj(mx.concatenate((x, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DurationTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, conv_layers=conv_layers
        )
        self.input_embed = DurationInputEmbedding(mel_dim, text_dim, dim)

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

        self.norm_out = nn.RMSNorm(dim)  # final modulation

    def __call__(
        self,
        x: float["b n d"],  # nosied input audio
        text: int["b nt"],  # text
        mask: bool["b n"] | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        
        t = self.time_embed(mx.ones((batch,), dtype=mx.float32))
        
        text_embed = self.text_embed(text, seq_len)
        
        x = self.input_embed(x, text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        x = self.norm_out(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        transformer: DiT,
        num_channels=None,
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()

        # mel spec

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = default(num_channels, self.mel_spec.n_mels)
        self.num_channels = num_channels

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        self.dim = dim

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # to prediction

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ...")
        )

    def __call__(
        self,
        inp: mx.array["b n d"] | mx.array["b nw"],  # mel or raw wave
        text: mx.array | list[str],
        *,
        lens: mx.array["b"] | None = None,
        return_loss=False,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[:2]

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map)
            else:
                text = list_str_to_tensor(text)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = mx.full((batch,), seq_len)

        mask = lens_to_mask(lens, length=seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = mx.random.uniform(0, 1, (batch,))
            rand_index = (rand_frac_index * lens).astype(mx.int32)

            seq = mx.arange(seq_len)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending

        inp = mx.where(
            repeat(mask, "b n -> b n d", d=self.num_channels), inp, mx.zeros_like(inp)
        )
        
        x = self.transformer(inp, text=text)

        x = maybe_masked_mean(x, mask)
        
        pred = self.to_pred(x)

        # return the prediction if not returning loss

        if not return_loss:
            return pred

        # loss

        duration = lens.astype(mx.float32) / SAMPLES_PER_SECOND

        return nn.losses.mse_loss(pred, duration)
