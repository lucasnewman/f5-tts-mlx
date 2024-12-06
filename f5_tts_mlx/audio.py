from __future__ import annotations
from functools import lru_cache
import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

import numpy as np


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: Optional[float] = None,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> mx.array:
    """
    Compute torch-compatible mel filterbanks.

    Args:
        sample_rate: Sampling rate of the audio.
        n_fft: Number of FFT points.
        n_mels: Number of mel bands.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        norm: Normalization mode.
        mel_scale: Mel scale type.

    Returns:
        mx.array of shape (n_mels, n_fft // 2 + 1) containing mel filterbanks.
    """

    def hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * mx.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    f_max = f_max or sample_rate / 2

    # generate frequency points

    n_freqs = n_fft // 2 + 1
    all_freqs = mx.linspace(0, sample_rate // 2, n_freqs)

    # convert frequencies to mel and back to hz

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = mx.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    # compute slopes for filterbank

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    # calculate overlapping triangular filters

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(
        mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes)
    )

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    filterbank = filterbank.moveaxis(0, 1)
    return filterbank


@lru_cache(maxsize=None)
def hanning(size):
    """
    Compute the Hanning window.

    Args:
        size: Size of the window.

        Returns:
            mx.array of shape (size,) containing the Hanning window.
    """
    return mx.array(np.hanning(size + 1)[:-1])


def stft(
    x,
    window,
    nperseg=256,
    noverlap=None,
    nfft=None,
    pad_mode="constant",
):
    """
    Compute the short-time Fourier transform of a signal.

    Args:
        x: mx.array of shape (t,) containing the input signal.
        window: mx.array of shape (nperseg,) containing the window function.
        nperseg: Number of samples per segment.
        noverlap: Number of overlapping samples.
        nfft: Number of FFT points.
        pad_mode: Padding mode.

    Returns:
        mx.array of shape (t, nfft // 2 + 1) containing the short-time Fourier transform.
    """
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
    audio: mx.array,
    sample_rate: int = 24_000,
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
):
    """
    Compute log-mel spectrograms for a batch of audio inputs.

    Args:
        audio: mx.array of shape [t] or [b, t] containing audio samples.
        sample_rate: Sampling rate of the audio.
        n_mels: Number of mel bands.
        n_fft: Number of FFT points.
        hop_length: Hop length between frames.
        padding: Amount of padding to add to each audio signal.

    Returns:
        mx.array of shape (batch_size, n_mels, frames) containing log-mel spectrograms.
    """

    if audio.ndim == 1:
        audio = mx.expand_dims(audio, axis=0)

    filters = mel_filters(
        sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, norm=None, mel_scale="htk"
    )

    batch = audio.shape[0]
    outputs = []

    for i in range(batch):
        one_audio = audio[i]

        if padding > 0:
            one_audio = mx.pad(one_audio, (0, padding))

        freqs = stft(one_audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
        magnitudes = mx.abs(freqs[:-1, :])

        mel_spec = mx.matmul(magnitudes, filters.T)
        log_spec = mx.maximum(mel_spec, 1e-5).log()
        outputs.append(log_spec)

    max_seq_len = max([x.shape[1] for x in outputs])
    outputs = [mx.pad(x, (0, max_seq_len - x.shape[1])) for x in outputs]
    return mx.stack(outputs, axis=0)


class MelSpec(nn.Module):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        return log_mel_spectrogram(
            audio, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
