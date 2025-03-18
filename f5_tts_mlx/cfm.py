"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Literal

import mlx.core as mx
import mlx.nn as nn

from einops.array_api import rearrange, repeat

from vocos_mlx import Vocos

from f5_tts_mlx.audio import MelSpec
from f5_tts_mlx.duration import DurationPredictor, DurationTransformer
from f5_tts_mlx.dit import DiT
from f5_tts_mlx.utils import (
    exists,
    fetch_from_hub,
    default,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
    pad_sequence,
)

# ode solvers


def odeint_euler(func, y0, t):
    """
    Solves ODE using the Euler method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y).
    - y0: Initial state, an MLX array of any shape.
    - t: Array of time steps, an MLX array.
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # compute the next value
        k = func(t_current, y_current)
        y_next = y_current + dt * k

        ys.append(y_next)
        y_current = y_next

    return mx.stack(ys)


def odeint_midpoint(func, y0, t):
    """
    Solves ODE using the midpoint method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y).
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


def odeint_rk4(func, y0, t):
    """
    Solves ODE using the Runge-Kutta 4th-order (RK4) method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y).
    - y0: Initial state, an MLX array of any shape.
    - t: Array of time steps, an MLX array.
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # rk4 steps
        k1 = func(t_current, y_current)
        k2 = func(t_current + 0.5 * dt, y_current + 0.5 * dt * k1)
        k3 = func(t_current + 0.5 * dt, y_current + 0.5 * dt * k2)
        k4 = func(t_current + dt, y_current + dt * k3)

        # compute the next value
        y_next = y_current + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        ys.append(y_next)
        y_current = y_next

    return mx.stack(ys)


# conditional flow matching


class F5TTS(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str, int] | None = None,
        vocoder: Callable[[mx.array["b d n"]], mx.array["b nw"]] | None = None,
        duration_predictor: DurationPredictor | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self._mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self._mel_spec.n_mels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # vocab map for tokenization
        self._vocab_char_map = vocab_char_map

        # vocoder (optional)
        self._vocoder = vocoder

        # duration predictor (optional)
        self._duration_predictor = duration_predictor

    def __call__(
        self,
        inp: mx.array["b n d"] | mx.array["b nw"],  # mel or raw wave
        text: mx.array["b nt"] | list[str],
        *,
        lens: mx.array["b"] | None = None,
    ) -> mx.array:
        # handle raw wave
        if inp.ndim == 2:
            inp = self._mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype = *inp.shape[:2], inp.dtype

        # handle text as string
        if isinstance(text, list):
            if exists(self._vocab_char_map):
                text = list_str_to_idx(text, self._vocab_char_map)
            else:
                text = list_str_to_tensor(text)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = mx.full((batch,), seq_len)

        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = mx.random.uniform(*self.frac_lengths_mask, (batch,))
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, max_length=seq_len)

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

        rand_audio_drop = mx.random.uniform(0, 1, (1,))
        rand_cond_drop = mx.random.uniform(0, 1, (1,))
        drop_audio_cond = rand_audio_drop < self.audio_drop_prob
        drop_text = rand_cond_drop < self.cond_drop_prob
        drop_audio_cond = drop_audio_cond | drop_text

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

        return loss.mean()

    def predict_duration(
        self,
        cond: mx.array["b n d"],
        text: mx.array["b nt"],
        speed: float = 1.0,
    ) -> int:
        duration_in_sec = self._duration_predictor(cond, text)
        frame_rate = self._mel_spec.sample_rate // self._mel_spec.hop_length
        duration = (duration_in_sec * frame_rate / speed).astype(mx.int32)
        return duration

    def sample(
        self,
        cond: mx.array["b n d"] | mx.array["b nw"],
        text: mx.array["b nt"] | list[str],
        duration: int | mx.array["b"] | None = None,
        *,
        lens: mx.array["b"] | None = None,
        steps=8,
        method: Literal["euler", "midpoint", "rk4"] = "rk4",
        cfg_strength=2.0,
        speed=1.0,
        sway_sampling_coef=-1.0,
        seed: int | None = None,
        max_duration=4096,
    ) -> tuple[mx.array, mx.array]:
        self.eval()

        # raw wave

        if cond.ndim == 2:
            cond = rearrange(cond, "1 n -> n")
            cond = self._mel_spec(cond)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, dtype = *cond.shape[:2], cond.dtype
        if not exists(lens):
            lens = mx.full((batch,), cond_seq_len, dtype=dtype)

        # text

        if isinstance(text, list):
            if exists(self._vocab_char_map):
                text = list_str_to_idx(text, self._vocab_char_map)
            else:
                text = list_str_to_tensor(text)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(axis=-1)
            lens = mx.maximum(text_lens, lens)

        # duration

        if duration is None and self._duration_predictor is not None:
            duration = self.predict_duration(cond, text, speed)
        elif duration is None:
            raise ValueError("Duration must be provided or a duration predictor must be set.")

        cond_mask = lens_to_mask(lens)

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

        # at each step, conditioning is fixed

        step_cond = mx.where(cond_mask, cond, mx.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # neural ode

        def fn(t, x):
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
            output = pred + (pred - null_pred) * cfg_strength
            return output

        # noise input

        y0 = []
        for dur in duration:
            if exists(seed):
                mx.random.seed(seed)
            y0.append(mx.random.normal((self.num_channels, dur)))
        y0 = pad_sequence(y0, padding_value=0)
        y0 = rearrange(y0, "b d n -> b n d")

        t_start = 0

        t = mx.linspace(t_start, 1, steps)
        if exists(sway_sampling_coef):
            t = t + sway_sampling_coef * (mx.cos(mx.pi / 2 * t) - 1 + t)

        if method == "midpoint":
            ode_step_fn = odeint_midpoint
        elif method == "euler":
            ode_step_fn = odeint_euler
        elif method == "rk4":
            ode_step_fn = odeint_rk4
        else:
            raise ValueError(f"Unknown method: {method}")

        fn = mx.compile(fn)
        trajectory = ode_step_fn(fn, y0, t)

        sampled = trajectory[-1]
        out = sampled
        out = mx.where(cond_mask, cond, out)

        if exists(self._vocoder):
            out = self._vocoder(out)

        return out, trajectory

    @classmethod
    def from_pretrained(
        cls,
        hf_model_name_or_path: str,
        convert_weights=True,
        quantization_bits: int | None = None,
    ) -> F5TTS:
        path = fetch_from_hub(hf_model_name_or_path, quantization_bits=quantization_bits)

        if path is None:
            raise ValueError(f"Could not find model {hf_model_name_or_path}")

        # vocab

        vocab_path = path / "vocab.txt"
        vocab = {v: i for i, v in enumerate(Path(vocab_path).read_text().split("\n"))}
        if len(vocab) == 0:
            raise ValueError(f"Could not load vocab from {vocab_path}")

        # duration predictor

        duration_model_path = path / "duration_v2.safetensors"
        duration_predictor = None

        if duration_model_path.exists():
            duration_predictor = DurationPredictor(
                transformer=DurationTransformer(
                    dim=512,
                    depth=8,
                    heads=8,
                    text_dim=512,
                    ff_mult=2,
                    conv_layers=2,
                    text_num_embeds=len(vocab) - 1,
                ),
                vocab_char_map=vocab,
            )
            weights = mx.load(duration_model_path.as_posix(), format="safetensors")
            duration_predictor.load_weights(list(weights.items()))

        # vocoder

        vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

        # model

        model_filename = "model_v1.safetensors"
        if exists(quantization_bits):
            model_filename = f"model_v1_{quantization_bits}b.safetensors"

        model_path = path / model_filename

        f5tts = F5TTS(
            transformer=DiT(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4,
                text_num_embeds=len(vocab) - 1,
                text_mask_padding=True,
            ),
            vocab_char_map=vocab,
            vocoder=vocos.decode,
            duration_predictor=duration_predictor,
        )

        weights = mx.load(model_path.as_posix(), format="safetensors")

        if convert_weights:
            new_weights = {}
            for k, v in weights.items():
                k = k.replace("ema_model.", "")

                # rename layers
                if len(k) < 1 or "mel_spec." in k or k in ("initted", "step"):
                    continue
                elif ".to_out" in k:
                    k = k.replace(".to_out", ".to_out.layers")
                elif ".text_blocks" in k:
                    k = k.replace(".text_blocks", ".text_blocks.layers")
                elif ".ff.ff.0.0" in k:
                    k = k.replace(".ff.ff.0.0", ".ff.ff.layers.0.layers.0")
                elif ".ff.ff.2" in k:
                    k = k.replace(".ff.ff.2", ".ff.ff.layers.2")
                elif ".time_mlp" in k:
                    k = k.replace(".time_mlp", ".time_mlp.layers")
                elif ".conv1d" in k:
                    k = k.replace(".conv1d", ".conv1d.layers")

                # reshape weights
                if ".dwconv.weight" in k:
                    v = v.swapaxes(1, 2)
                elif ".conv1d.layers.0.weight" in k:
                    v = v.swapaxes(1, 2)
                elif ".conv1d.layers.2.weight" in k:
                    v = v.swapaxes(1, 2)

                new_weights[k] = v

            weights = new_weights

        if quantization_bits is not None:
            nn.quantize(
                f5tts,
                bits=quantization_bits,
                class_predicate=lambda p, m: (isinstance(m, nn.Linear) and m.weight.shape[1] % 64 == 0),
            )

        f5tts.load_weights(list(weights.items()))
        mx.eval(f5tts.parameters())

        return f5tts
