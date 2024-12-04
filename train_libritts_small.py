from mlx.utils import tree_flatten

from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.dit import DiT
from f5_tts_mlx.trainer import F5TTSTrainer
from f5_tts_mlx.data import load_libritts_r

vocab = {chr(i): i for i in range(256)}

f5tts = F5TTS(
    transformer=DiT(
        dim=256,
        depth=8,
        heads=8,
        ff_mult=2,
        text_dim=128,
        conv_layers=2,
        text_num_embeds=len(vocab),
    ),
    vocab_char_map=vocab,
)

num_trainable_params = sum(
    [p[1].size for p in tree_flatten(f5tts.trainable_parameters())]
)
print(f"Using {num_trainable_params:,} trainable parameters.")

dataset = load_libritts_r(max_duration = 10)

batched_dataset = (
    dataset
    .repeat(1_000_000)  # repeat indefinitely
    .shuffle(1000)
    .prefetch(prefetch_size = 4, num_threads = 1)
    .batch(4)
    .pad_to_multiple("mel_spec", dim=2, pad_multiple=256, pad_value=0.0)
)

trainer = F5TTSTrainer(
    model=f5tts,
    num_warmup_steps=1000,
    max_grad_norm=1,
    log_with_wandb=False
)

trainer.train(
    train_dataset=batched_dataset,
    learning_rate=1e-4,
    log_every=10,
    save_every=10_000,
    total_steps=1_000_000,
)
