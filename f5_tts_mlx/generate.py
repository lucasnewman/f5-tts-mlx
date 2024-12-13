import argparse
from collections import deque
import datetime
import pkgutil
import re
import sys
from threading import Event, Lock
from typing import Literal, Optional

import mlx.core as mx
import numpy as np

from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.utils import convert_char_to_pinyin

import sounddevice as sd
import soundfile as sf

from tqdm import tqdm

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH
TARGET_RMS = 0.1


# utilities


def split_sentences(text):
    sentence_endings = re.compile(r"([.!?;:])")
    sentences = sentence_endings.split(text)
    sentences = [
        sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)
    ]
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# playback


class AudioPlayer:
    def __init__(self, sample_rate=24000, buffer_size=2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.playing = False
        self.drain_event = Event()

    def callback(self, outdata, frames, time, status):
        with self.buffer_lock:
            if len(self.audio_buffer) > 0:
                available = min(frames, len(self.audio_buffer[0]))
                chunk = self.audio_buffer[0][:available].copy()
                self.audio_buffer[0] = self.audio_buffer[0][available:]

                if len(self.audio_buffer[0]) == 0:
                    self.audio_buffer.popleft()
                    if len(self.audio_buffer) == 0:
                        self.drain_event.set()

                outdata[:, 0] = np.zeros(frames)
                outdata[:available, 0] = chunk
            else:
                outdata[:, 0] = np.zeros(frames)
                self.drain_event.set()

    def play(self):
        if not self.playing:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                blocksize=self.buffer_size,
            )
            self.stream.start()
            self.playing = True
            self.drain_event.clear()

    def queue_audio(self, samples):
        with self.buffer_lock:
            self.audio_buffer.append(np.array(samples))
        if not self.playing:
            self.play()

    def wait_for_drain(self):
        return self.drain_event.wait()

    def stop(self):
        if self.playing:
            self.wait_for_drain()
            sd.sleep(100)
            
            self.stream.stop()
            self.stream.close()
            self.playing = False


# generation


def generate(
    generation_text: str,
    duration: Optional[float] = None,
    model_name: str = "lucasnewman/f5-tts-mlx",
    ref_audio_path: Optional[str] = None,
    ref_audio_text: Optional[str] = None,
    steps: int = 8,
    method: Literal["euler", "midpoint"] = "rk4",
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,  # used when duration is None as part of the duration heuristic
    seed: Optional[int] = None,
    quantization_bits: Optional[int] = None,
    output_path: Optional[str] = None,
):
    player = AudioPlayer(sample_rate=SAMPLE_RATE) if output_path is None else None

    # the default model already has converted weights
    convert_weights = model_name != "lucasnewman/f5-tts-mlx"

    f5tts = F5TTS.from_pretrained(model_name, convert_weights=convert_weights, quantization_bits=quantization_bits)

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
    
    sentences = split_sentences(generation_text)
    is_single_generation = len(sentences) <= 1 or duration is not None

    if is_single_generation:
        generation_text = convert_char_to_pinyin(
            [ref_audio_text + " " + generation_text]
        )

        if duration is not None:
            duration = int(duration * FRAMES_PER_SEC)

        start_date = datetime.datetime.now()

        wave, _ = f5tts.sample(
            mx.expand_dims(audio, axis=0),
            text=generation_text,
            duration=duration,
            steps=steps,
            method=method,
            speed=speed,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
        )

        wave = wave[audio.shape[0] :]
        mx.eval(wave)

        generated_duration = wave.shape[0] / SAMPLE_RATE
        print(
            f"Generated {generated_duration:.2f}s of audio in {datetime.datetime.now() - start_date}."
        )

        if player is not None:
            player.queue_audio(wave)

        if output_path is not None:
            sf.write(output_path, np.array(wave), SAMPLE_RATE)

        if player is not None:
            player.stop()
    else:
        start_date = datetime.datetime.now()

        output = []

        for sentence_text in tqdm(split_sentences(generation_text)):
            text = convert_char_to_pinyin([ref_audio_text + " " + sentence_text])

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
                seed=seed,
            )

            # trim the reference audio
            wave = wave[audio.shape[0] :]
            mx.eval(wave)

            output.append(wave)

            if player is not None:
                mx.eval(wave)
                player.queue_audio(wave)

        wave = mx.concatenate(output, axis=0)

        generated_duration = wave.shape[0] / SAMPLE_RATE
        print(
            f"Generated {generated_duration:.2f}s of audio in {datetime.datetime.now() - start_date}."
        )

        if output_path is not None:
            sf.write(output_path, np.array(wave), SAMPLE_RATE)

        if player is not None:
            player.stop()


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
        "--text",
        type=str,
        default=None,
        help="Text to generate speech from (leave blank to input via stdin)",
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
        default=None,
        help="Path to save the generated audio output",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of steps to take when sampling the neural ODE",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rk4",
        choices=["euler", "midpoint", "rk4"],
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
        default=1.0,
        help="Speed factor for the duration heuristic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for noise generation",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=None,
        help="Number of bits to use for quantization. 4 and 8 are supported.",
    )

    args = parser.parse_args()

    if args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            print("Please enter the text to generate:")
            args.text = input("> ").strip()

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
        quantization_bits=args.q,
        output_path=args.output,
    )
