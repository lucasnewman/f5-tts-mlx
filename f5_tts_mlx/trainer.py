from __future__ import annotations
import datetime
from functools import partial
import io
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import (
    AdamW,
    linear_schedule,
    cosine_decay,
    join_schedules,
    clip_grad_norm,
)
from mlx.utils import tree_flatten

from einops.array_api import rearrange

from f5_tts_mlx.audio import MelSpec
from f5_tts_mlx.cfm import F5TTS

import soundfile as sf

from PIL import Image
import matplotlib.pyplot as plt

import wandb


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# trainer

TARGET_RMS = 0.1

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH


class F5TTSTrainer:
    def __init__(
        self,
        model: F5TTS,
        num_warmup_steps=1000,
        max_grad_norm=1.0,
        sample_rate=24_000,
        log_with_wandb=False,
    ):
        self.model = model
        self.num_warmup_steps = num_warmup_steps
        self.mel_spectrogram = MelSpec(sample_rate=sample_rate)
        self.max_grad_norm = max_grad_norm
        self.log_with_wandb = log_with_wandb

    def save_checkpoint(self, step, finetune=False):
        if Path("results").exists() is False:
            os.makedirs("results")

        mx.save_safetensors(
            f"results/f5tts_{step}",
            dict(tree_flatten(self.model.trainable_parameters())),
        )

    def load_checkpoint(self, step):
        params = mx.load(f"results/f5tts_{step}.safetensors")
        self.model.load_weights(list(params.items()))
        self.model.eval()

    def generate_sample(
        self,
        sample_audio: str,
        sample_ref_text: str,
        sample_generation_text: str,
        sample_generation_duration: float,
        step: int,
    ):
        audio, _ = sf.read(sample_audio)
        audio = mx.array(audio)
        ref_audio_duration = audio.shape[0] / SAMPLE_RATE

        rms = mx.sqrt(mx.mean(mx.square(audio)))
        if rms < TARGET_RMS:
            audio = audio * TARGET_RMS / rms

        audio = mx.expand_dims(audio, axis=0)
        text = [sample_ref_text + " " + sample_generation_text]

        self.model.eval()

        start_date = datetime.datetime.now()

        wave, trajectories = self.model.sample(
            audio,
            text=text,
            duration=int(
                (ref_audio_duration + sample_generation_duration) * FRAMES_PER_SEC
            ),
            method="rk4",
            steps=8,
            cfg_strength=2,
            speed=1,
            sway_sampling_coef=-1.0,
        )
        mx.eval([wave, trajectories])

        elapsed_time = (datetime.datetime.now() - start_date).total_seconds()
        print(f"Generated sample at step {step} in {elapsed_time:0.1f}s")

        # save the generated audio

        wave = wave[audio.shape[1] :]
        os.makedirs("samples/audio", exist_ok=True)
        sf.write(
            f"samples/audio/step_{step}.wav", np.array(wave), samplerate=SAMPLE_RATE
        )

        # save a visualization of the trajectory

        frames = []
        
        ref_audio_frame_len = audio.shape[1] // HOP_LENGTH

        for trajectory in trajectories:
            plt.figure(figsize=(10, 4))
            plt.imshow(
                np.array(trajectory[0, ref_audio_frame_len:]).T,
                aspect="auto",
                origin="lower",
                interpolation="none",
            )
            plt.yticks([])

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)

            frames.append(Image.open(buf))
            plt.close()

        os.makedirs("samples/viz", exist_ok=True)
        frames[0].save(
            f"samples/viz/step_{step}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
        )

        self.model.train()

    def train(
        self,
        train_dataset,
        learning_rate=1e-4,
        weight_decay=1e-2,
        total_steps=1_000_000,
        save_every=10_000,
        sample_every=5_000,
        sample_reference_audio: str | None = None,
        sample_reference_text: str | None = None,
        sample_generation_text: str | None = None,
        sample_generation_duration: float | None = None,
        checkpoint: int | None = None,
    ):
        if self.log_with_wandb:
            wandb.init(
                project="f5tts",
                config=dict(
                    learning_rate=learning_rate,
                    total_steps=total_steps,
                ),
            )

        decay_steps = total_steps - self.num_warmup_steps

        warmup_scheduler = linear_schedule(
            init=1e-8,
            end=learning_rate,
            steps=self.num_warmup_steps,
        )
        decay_scheduler = cosine_decay(init=learning_rate, decay_steps=decay_steps)
        scheduler = join_schedules(
            schedules=[warmup_scheduler, decay_scheduler],
            boundaries=[self.num_warmup_steps],
        )
        self.optimizer = AdamW(learning_rate=scheduler, weight_decay=weight_decay)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
            start_step = checkpoint
        else:
            start_step = 0

        global_step = start_step

        if global_step != 0:
            print(f"Starting training at step {global_step}")

        def loss_fn(model, mel_spec, text, lens):
            return model(mel_spec, text=text, lens=lens)

        state = [self.model.state, self.optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def train_step(mel_spec, text_inputs, mel_lens):
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(
                self.model,
                mel_spec,
                text=text_inputs,
                lens=mel_lens,
            )

            if self.max_grad_norm > 0:
                grads, _ = clip_grad_norm(grads, max_norm=self.max_grad_norm)

            self.optimizer.update(self.model, grads)

            return loss

        training_start_date = datetime.datetime.now()

        self.model.train()

        pbar = tqdm(
            initial=start_step, total=total_steps, desc="", unit="step"
        )

        for step, batch in enumerate(train_dataset):
            mel_spec = rearrange(mx.array(batch["mel_spec"]), "b 1 n c -> b n c")
            mel_lens = mx.array(batch["mel_len"], dtype=mx.int32)

            # pad text to sequence length with -1
            seq_len = mel_spec.shape[1]
            text = mx.array(batch["transcript"]).squeeze(-1)
            text = mx.pad(
                text, [(0, 0), (0, seq_len - text.shape[-1])], constant_values=-1
            )

            loss = train_step(mel_spec, text, mel_lens)
            mx.eval(state)
            # mx.eval(self.model.parameters(), self.optimizer.state)

            if self.log_with_wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": self.optimizer.learning_rate.item(),
                        "batch_len": mel_lens.sum().item(),
                    },
                    step=global_step,
                )

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "batch_len": f"{mel_lens.sum().item():04d}",
                }
            )

            global_step += 1

            if global_step % save_every == 0:
                self.save_checkpoint(global_step)

            if (
                global_step % sample_every == 0
                and sample_reference_audio is not None
                and sample_reference_text is not None
                and sample_generation_text is not None
                and sample_generation_duration is not None
            ):
                self.generate_sample(sample_reference_audio, sample_reference_text, sample_generation_text, sample_generation_duration, global_step)

            if global_step >= total_steps:
                break

        pbar.close()
        if self.log_with_wandb:
            wandb.finish()

        print(f"Training complete in {datetime.datetime.now() - training_start_date}")
