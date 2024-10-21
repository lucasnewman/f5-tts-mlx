from __future__ import annotations
import datetime

from einops.array_api import rearrange

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

from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.duration import DurationPredictor
from f5_tts_mlx.modules import MelSpec

import wandb


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# trainer


class DurationTrainer:
    def __init__(
        self,
        model: DurationPredictor,
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
        mx.save_safetensors(
            f"f5tts_duration_{step}",
            dict(tree_flatten(self.model.trainable_parameters())),
        )

    def load_checkpoint(self, step):
        params = mx.load(f"f5tts_duration_{step}.saftensors")
        self.model.load_weights(params)
        self.model.eval()

    def train(
        self,
        train_dataset,
        learning_rate=1e-4,
        weight_decay=1e-2,
        total_steps=100_000,
        batch_size=8,
        log_every=10,
        save_every=1000,
        checkpoint: int | None = None,
    ):
        if self.log_with_wandb:
            wandb.init(
                project="f5tts_duration",
                config=dict(
                    learning_rate=learning_rate,
                    total_steps=total_steps,
                    batch_size=batch_size,
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

        def loss_fn(model: F5TTS, mel_spec, text, lens):
            loss = model(mel_spec, text=text, lens=lens, return_loss=True)
            return loss

        # state = [self.model.state, self.optimizer.state, mx.random.state]

        # @partial(mx.compile, inputs=state, outputs=state)
        def train_step(mel_spec, text_inputs, mel_lens):
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(
                self.model, mel_spec, text=text_inputs, lens=mel_lens
            )

            if self.max_grad_norm > 0:
                grads, _ = clip_grad_norm(grads, max_norm=self.max_grad_norm)

            self.optimizer.update(self.model, grads)

            return loss

        training_start_date = datetime.datetime.now()
        log_start_date = datetime.datetime.now()

        batched_dataset = (
            train_dataset.repeat(1_000_000)  # repeat indefinitely
            .shuffle(1000)
            .prefetch(prefetch_size=batch_size, num_threads=4)
            .batch(batch_size)
        )

        for batch in batched_dataset:
            effective_batch_size = batch["transcript"].shape[0]
            text_inputs = [
                bytes(batch["transcript"][i]).decode("utf-8")
                for i in range(effective_batch_size)
            ]

            mel_spec = rearrange(mx.array(batch["mel_spec"]), "b 1 n c -> b n c")
            mel_lens = mx.array(batch["mel_len"], dtype=mx.int32)

            loss = train_step(mel_spec, text_inputs, mel_lens)
            # mx.eval(state)
            mx.eval(self.model.parameters(), self.optimizer.state)

            if self.log_with_wandb:
                wandb.log(
                    {"loss": loss.item(), "lr": self.optimizer.learning_rate.item()},
                    step=global_step,
                )

            if global_step > 0 and global_step % log_every == 0:
                elapsed_time = datetime.datetime.now() - log_start_date
                log_start_date = datetime.datetime.now()

                print(
                    f"step {global_step}: loss = {loss.item():.4f}, sec per step = {(elapsed_time.seconds / log_every):.2f}"
                )

            global_step += 1

            if global_step % save_every == 0:
                self.save_checkpoint(global_step)

            if global_step >= total_steps:
                break

        if self.log_with_wandb:
            wandb.finish()

        print(f"Training complete in {datetime.datetime.now() - training_start_date}")

class F5TTSTrainer:
    def __init__(
        self,
        model: DurationPredictor,
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
        mx.save_safetensors(
            f"f5tts_{step}",
            dict(tree_flatten(self.model.trainable_parameters())),
        )

    def load_checkpoint(self, step):
        params = mx.load(f"f5tts_{step}.saftensors")
        self.model.load_weights(params)
        self.model.eval()

    def train(
        self,
        train_dataset,
        learning_rate=1e-4,
        weight_decay=1e-2,
        total_steps=100_000,
        batch_size=8,
        log_every=10,
        save_every=1000,
        checkpoint: int | None = None,
    ):
        if self.log_with_wandb:
            wandb.init(
                project="f5tts",
                config=dict(
                    learning_rate=learning_rate,
                    total_steps=total_steps,
                    batch_size=batch_size,
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

        def loss_fn(model: F5TTS, mel_spec, text, lens):
            return model(mel_spec, text=text, lens=lens)

        # state = [self.model.state, self.optimizer.state, mx.random.state]

        # @partial(mx.compile, inputs=state, outputs=state)
        def train_step(mel_spec, text_inputs, mel_lens):
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(
                self.model, mel_spec, text=text_inputs, lens=mel_lens
            )

            if self.max_grad_norm > 0:
                grads, _ = clip_grad_norm(grads, max_norm=self.max_grad_norm)

            self.optimizer.update(self.model, grads)

            return loss

        training_start_date = datetime.datetime.now()
        log_start_date = datetime.datetime.now()

        batched_dataset = (
            train_dataset.repeat(1_000_000)  # repeat indefinitely
            .shuffle(1000)
            .prefetch(prefetch_size=batch_size, num_threads=4)
            .batch(batch_size)
        )

        for batch in batched_dataset:
            effective_batch_size = batch["transcript"].shape[0]
            text_inputs = [
                bytes(batch["transcript"][i]).decode("utf-8")
                for i in range(effective_batch_size)
            ]

            mel_spec = rearrange(mx.array(batch["mel_spec"]), "b 1 n c -> b n c")
            mel_lens = mx.array(batch["mel_len"], dtype=mx.int32)

            loss = train_step(mel_spec, text_inputs, mel_lens)
            # mx.eval(state)
            mx.eval(self.model.parameters(), self.optimizer.state)

            if self.log_with_wandb:
                wandb.log(
                    {"loss": loss.item(), "lr": self.optimizer.learning_rate.item()},
                    step=global_step,
                )

            if global_step > 0 and global_step % log_every == 0:
                elapsed_time = datetime.datetime.now() - log_start_date
                log_start_date = datetime.datetime.now()

                print(
                    f"step {global_step}: loss = {loss.item():.4f}, sec per step = {(elapsed_time.seconds / log_every):.2f}"
                )

            global_step += 1

            if global_step % save_every == 0:
                self.save_checkpoint(global_step)

            if global_step >= total_steps:
                break

        if self.log_with_wandb:
            wandb.finish()

        print(f"Training complete in {datetime.datetime.now() - training_start_date}")
