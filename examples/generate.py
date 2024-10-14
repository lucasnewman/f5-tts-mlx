import argparse
import datetime
from pathlib import Path
from typing import Optional

import mlx.core as mx

import numpy as np

from f5_tts_mlx.cfm import CFM
from f5_tts_mlx.utils import convert_char_to_pinyin

from vocos_mlx import Vocos

import soundfile as sf

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH
TARGET_RMS = 0.1


def generate(
    generation_text: str,
    duration: float,
    model_name: str = "lucasnewman/f5-tts-mlx",
    vocab_path: str = "data/Emilia_ZH_EN_pinyin/vocab.txt",
    ref_audio_path: str = "tests/test_en_1_ref_short.wav",
    ref_audio_text: str = "Some call me nature, others call me mother nature.",
    sway_sampling_coef: float = 0.0,
    seed: Optional[int] = None,
    output_path: str = "output.wav",
):
    vocab = {v: i for i, v in enumerate(Path(vocab_path).read_text().split("\n"))}

    f5tts = CFM.from_pretrained(model_name, vocab)

    # load reference audio
    audio, sr = sf.read(ref_audio_path)
    audio = mx.array(audio)
    ref_audio_duration = audio.shape[0] / SAMPLE_RATE

    rms = mx.sqrt(mx.mean(mx.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms

    # generate the audio for the given text
    text = convert_char_to_pinyin([ref_audio_text + " " + generation_text])

    frame_duration = int((ref_audio_duration + duration) * FRAMES_PER_SEC)
    print(f"Generating {frame_duration} total frames of audio...")

    start_date = datetime.datetime.now()
    vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

    wave, _ = f5tts.sample(
        mx.expand_dims(audio, axis=0),
        text=text,
        duration=frame_duration,
        steps=32,
        cfg_strength=1,
        sway_sampling_coef=sway_sampling_coef,
        seed=seed,
        vocoder=vocos.decode,
    )

    # trim the reference audio
    wave = wave[audio.shape[0]:]
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
        required=True,
        help="Duration of the generated audio in seconds",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/Emilia_ZH_EN_pinyin/vocab.txt",
        help="Path to the vocab file",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default="tests/test_en_1_ref_short.wav",
        help="Path to the reference audio file",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Some call me nature, others call me mother nature.",
        help="Text spoken in the reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Path to save the generated audio output",
    )
    
    parser.add_argument(
        "--sway-coef",
        type=float,
        default="0.0",
        help="Coefficient for sway sampling",
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
        vocab_path=args.vocab,
        ref_audio_path=args.ref_audio,
        ref_audio_text=args.ref_text,
        sway_sampling_coef=args.sway_coef,
        seed=args.seed,
        output_path=args.output,
    )
