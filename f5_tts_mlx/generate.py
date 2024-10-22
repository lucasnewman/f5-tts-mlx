import argparse
import datetime
import pkgutil
from typing import Literal, Optional

import mlx.core as mx

import numpy as np

from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.utils import convert_char_to_pinyin

import soundfile as sf

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH
TARGET_RMS = 0.1


def generate(
    generation_text: str,
    duration: Optional[float] = None,
    model_name: str = "lucasnewman/f5-tts-mlx",
    ref_audio_path: Optional[str] = None,
    ref_audio_text: Optional[str] = None,
    steps: int = 32,
    method: Literal["euler", "midpoint"] = "euler",
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 0.8,  # used when duration is None as part of the duration heuristic
    seed: Optional[int] = None,
    output_path: str = "output.wav",
):
    f5tts = F5TTS.from_pretrained(model_name)

    if ref_audio_path is None:
        data = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")

        # write to a temp file
        tmp_ref_audio_file = "/tmp/ref.wav"
        with open(tmp_ref_audio_file, "wb") as f:
            f.write(data)

        if data is not None:
            audio, sr = sf.read(tmp_ref_audio_file)
            ref_audio_text = "Some call me nature, others call me mother nature."
    else:
        # load reference audio
        audio, sr = sf.read(ref_audio_path)
        if sr != SAMPLE_RATE:
            raise ValueError("Reference audio must have a sample rate of 24kHz")

    audio = mx.array(audio)
    ref_audio_duration = audio.shape[0] / SAMPLE_RATE
    print(f"Got reference audio with duration: {ref_audio_duration:.2f} seconds")

    rms = mx.sqrt(mx.mean(mx.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms

    # generate the audio for the given text
    text = convert_char_to_pinyin([ref_audio_text + " " + generation_text])

    start_date = datetime.datetime.now()
    
    if duration is not None:
        duration = int(duration * FRAMES_PER_SEC)

    wave, _ = f5tts.sample(
        mx.expand_dims(audio, axis=0),
        text=text,
        duration=duration,
        steps=steps,
        method=method,
        speed=speed,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        seed=seed
    )

    # trim the reference audio
    wave = wave[audio.shape[0] :]
    generated_duration = wave.shape[0] / SAMPLE_RATE
    elapsed_time = datetime.datetime.now() - start_date

    print(f"Generated {generated_duration:.2f} seconds of audio in {elapsed_time}.")

    sf.write(output_path, np.array(wave), SAMPLE_RATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio from text using f5-tts-mlx"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lucasnewman/f5-tts-mlx",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text to generate speech from"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration of the generated audio in seconds",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to the reference audio file",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Text spoken in the reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Path to save the generated audio output",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of steps to take when sampling the neural ODE",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="euler",
        choices=["euler", "midpoint"],
        help="Method to use for sampling the neural ODE",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help="Strength of classifer free guidance",
    )
    parser.add_argument(
        "--sway-coef",
        type=float,
        default=-1.0,
        help="Coefficient for sway sampling",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.8,
        help="Speed factor for the duration heuristic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for noise generation",
    )

    args = parser.parse_args()

    generate(
        generation_text=args.text,
        duration=args.duration,
        model_name=args.model,
        ref_audio_path=args.ref_audio,
        ref_audio_text=args.ref_text,
        steps=args.steps,
        method=args.method,
        cfg_strength=args.cfg,
        sway_sampling_coef=args.sway_coef,
        speed=args.speed,
        seed=args.seed,
        output_path=args.output,
    )
