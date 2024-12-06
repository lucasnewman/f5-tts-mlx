from pathlib import Path

from mlx.utils import tree_flatten

from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.dit import DiT
from f5_tts_mlx.trainer import F5TTSTrainer, FRAMES_PER_SEC
from f5_tts_mlx.data import load_libritts_r

from vocos_mlx import Vocos

vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")

vocab = {chr(i): i for i in range(256)}

f5tts = F5TTS(
    transformer=DiT(
        dim=768,
        depth=16,
        heads=8,
        ff_mult=2,
        text_dim=384,
        conv_layers=4,
        text_num_embeds=len(vocab),
    ),
    vocab_char_map=vocab,
    vocoder=vocos.decode,
)

num_trainable_params = sum(
    [p[1].size for p in tree_flatten(f5tts.trainable_parameters())]
)
print(f"Using {num_trainable_params:,} trainable parameters.")

epochs = 100
max_duration = 10

dataset, path = load_libritts_r(max_duration = max_duration)

max_batch_duration = 40
batch_size = int(max_batch_duration / max_duration)
max_data_size = int(max_batch_duration * FRAMES_PER_SEC) * f5tts._mel_spec.n_mels

batched_dataset = (
    dataset
    .repeat(epochs)
    .shuffle(500)
    .prefetch(prefetch_size = batch_size, num_threads = 6)
    .batch(batch_size, pad=dict(mel_spec=0.0, transcript=-1))
    # .dynamic_batch(buffer_size = batch_size * 2, key = "mel_spec", max_data_size = max_data_size, shuffle = True)
    .pad_to_multiple("mel_spec", dim=2, pad_multiple=256, pad_value=0.0)
)

trainer = F5TTSTrainer(
    model=f5tts,
    num_warmup_steps=1000,
    max_grad_norm=1,
    log_with_wandb=False
)

sample_path = "tests/test_en_1_ref_short.wav"
sample_text = "Some call me nature, others call me mother nature."

trainer.train(
    train_dataset=batched_dataset,
    learning_rate=1e-4,
    total_steps=1_000_000,
    save_every=10_000,
    checkpoint=100_000,
    sample_every=100,
    sample_reference_audio=sample_path,
    sample_reference_text=sample_text,
    sample_generation_duration=3.5,
    sample_generation_text="The quick brown fox jumped over the lazy dog.",
)
