"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from random import random

import mlx.core as mx
import mlx.nn as nn

from einops.array_api import rearrange, reduce, repeat
import einx

from f5_tts_mlx.modules import MelSpec


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def lens_to_mask(
    t: mx.array,
    length: int | None = None,
) -> mx.array:  # Bool['b n']
    if not exists(length):
        length = t.max()

    seq = mx.arange(length)
    return einx.less("n, b -> b n", seq, t)


def mask_from_start_end_indices(
    seq_len: mx.array,
    start: mx.array,
    end: mx.array,
    max_length: int | None = None,
):
    max_seq_len = default(max_length, seq_len.max().item())
    seq = mx.arange(max_seq_len).astype(mx.int32)
    return einx.greater_equal("n, b -> b n", seq, start) & einx.less(
        "n, b -> b n", seq, end
    )


def mask_from_frac_lengths(
    seq_len: mx.array,
    frac_lengths: mx.array,
    max_length: int | None = None,
):
    lengths = (frac_lengths * seq_len).astype(mx.int32)
    max_start = seq_len - lengths

    rand = mx.random.uniform(0, 1, frac_lengths.shape)

    start = mx.maximum((max_start * rand).astype(mx.int32), 0)
    end = start + lengths

    out = mask_from_start_end_indices(seq_len, start, end, max_length)

    if exists(max_length):
        out = pad_to_length(out, max_length)

    return out


def maybe_masked_mean(t: mx.array, mask: mx.array | None = None) -> mx.array:
    if not exists(mask):
        return t.mean(dim=1)

    t = einx.where("b n, b n d, -> b n d", mask, t, 0.0)
    num = reduce(t, "b n d -> b d", "sum")
    den = reduce(mask.astype(mx.int32), "b n -> b", "sum")

    return einx.divide("b d, b -> b d", num, mx.maximum(den, 1))


def pad_to_length(t: mx.array, length: int, value=None):
    ndim = t.ndim
    seq_len = t.shape[-1]
    if length > seq_len:
        if ndim == 1:
            t = mx.pad(t, [(0, length - seq_len)], constant_values=value)
        elif ndim == 2:
            t = mx.pad(t, [(0, 0), (0, length - seq_len)], constant_values=value)
        elif ndim == 3:
            t = mx.pad(
                t, [(0, 0), (0, length - seq_len), (0, 0)], constant_values=value
            )
        else:
            raise ValueError(f"Unsupported padding dims: {ndim}")
    return t[..., :length]


def pad_sequence(t: mx.array, padding_value=0):
    max_len = max([i.shape[-1] for i in t])
    t = mx.array([pad_to_length(i, max_len, padding_value) for i in t])
    return t


# simple utf-8 tokenizer, since paper went character based


def list_str_to_tensor(text: list[str], padding_value=-1) -> mx.array:  # Int['b nt']:
    list_tensors = [mx.array([*bytes(t, "UTF-8")]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value=-1)
    return padded_tensor


# char tokenizer, based on custom dataset's extracted .txt file


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> mx.array:  # Int['b nt']:
    list_idx_tensors = [
        [vocab_char_map.get(c, 0) for c in t] for t in text
    ]  # pinyin or char style
    text = pad_sequence(mx.array(list_idx_tensors), padding_value=-1)
    return text


# conditional flow matching


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    def __call__(
        self,
        inp: mx.array["b n d"] | mx.array["b nw"],  # mel or raw wave
        text: mx.array["b nt"] | list[str],
        *,
        lens: mx.array["b"] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, σ1 = *inp.shape[:2], inp.dtype, self.sigma

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

        # get a random span to mask out for training conditionally
        frac_lengths = mx.random.uniform(*self.frac_lengths_mask, (batch,))
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask = rand_span_mask & mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = mx.random.normal(x1.shape)

        # time step
        time = mx.random.uniform(0, 1, (batch,), dtype=dtype)

        # sample xt (φ_t(x) in the paper)
        t = rearrange(time, "b -> b 1 1")
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = mx.where(
            rand_span_mask[..., None],
            mx.zeros_like(x1),
            x1,
        )

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )

        # flow matching loss
        loss = nn.losses.mse_loss(pred, flow, reduction="none")

        rand_span_mask = repeat(rand_span_mask, "b n -> b n d", d=self.num_channels)
        masked_loss = mx.where(rand_span_mask, loss, mx.zeros_like(loss))
        loss = mx.sum(masked_loss) / mx.maximum(mx.sum(rand_span_mask), 1e-6)

        return loss.mean(), cond

    def odeint(self, func, y0, t, **kwargs):
        """
        Solves ODE using the midpoint method.

        Parameters:
        - y0: Initial state, an MLX array of any shape.
        - t: Array of time steps, an MLX array.
        """
        ys = [y0]
        y_current = y0

        for i in range(len(t) - 1):
            t_current = t[i]
            dt = t[i + 1] - t_current

            # midpoint approximation
            k1 = func(t_current, y_current)
            mid = y_current + 0.5 * dt * k1

            # compute the next value
            k2 = func(t_current + 0.5 * dt, mid)
            y_next = y_current + dt * k2

            ys.append(y_next)
            y_current = y_next

        return mx.stack(ys)

    def sample(
        self,
        cond: mx.array["b n d"] | mx.array["b nw"],
        text: mx.array["b nt"] | list[str],
        duration: int | mx.array["b"],
        *,
        lens: mx.array["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[mx.array["b d n"]], mx.array["b nw"]] | None = None,
        no_ref_audio=False,
        t_inter=0.1,
        edit_mask=None,
    ) -> tuple[mx.array, mx.array]:
        self.eval()

        # raw wave

        if cond.ndim == 2:
            cond = rearrange(cond, "1 n -> n")
            cond = self.mel_spec(cond)
            # cond = rearrange(cond, "b d n -> b n d")
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, dtype = *cond.shape[:2], cond.dtype
        if not exists(lens):
            lens = mx.full((batch,), cond_seq_len, dtype=dtype)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map)
            else:
                text = list_str_to_tensor(text)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(axis=-1)
            lens = mx.maximum(text_lens, lens)

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = mx.full((batch,), duration, dtype=dtype)

        duration = mx.maximum(lens + 1, duration)
        duration = mx.clip(duration, 0, max_duration)
        max_duration = int(duration.max().item())

        cond = mx.pad(cond, [(0, 0), (0, max_duration - cond_seq_len), (0, 0)])
        cond_mask = mx.pad(
            cond_mask,
            [(0, 0), (0, max_duration - cond_mask.shape[-1])],
            constant_values=False,
        )
        cond_mask = rearrange(cond_mask, "... -> ... 1")
        step_cond = mx.where(
            cond_mask, cond, mx.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = mx.zeros_like(cond)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = mx.where(cond_mask, cond, mx.zeros_like(cond))

            # predict flow
            pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                mx.random.seed(seed)
            y0.append(mx.random.normal((dur, self.num_channels)))
        y0 = pad_sequence(y0, padding_value=0)

        t_start = 0

        t = mx.linspace(t_start, 1, steps)
        if exists(sway_sampling_coef):
            t = t + sway_sampling_coef * (mx.cos(mx.pi / 2 * t) - 1 + t)

        trajectory = self.odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = mx.where(cond_mask, cond, out)

        if exists(vocoder):
            out = rearrange(out, "b n d -> b d n")
            out = vocoder(out)

        return out, trajectory
