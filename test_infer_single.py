import datetime
from pathlib import Path

import mlx.core as mx

import numpy as np

from f5_tts_mlx.cfm import CFM
from f5_tts_mlx.dit import DiT
from f5_tts_mlx.utils import convert_char_to_pinyin

import torch
import torchaudio

from vocos_mlx import Vocos

vocab_path = Path("data/Emilia_ZH_EN_pinyin/vocab.txt")
vocab = {v: i for i, v in enumerate(vocab_path.read_text().split("\n"))}

f5tts = DiT(dim = 1024, depth = 22, heads = 16, ff_mult = 2, text_dim = 512, conv_layers = 4, text_num_embeds = len(vocab) - 1)
cfm = CFM(transformer=f5tts, vocab_char_map=vocab)

ckpt_path = Path("model_1200000.pt")
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)['ema_model_state_dict']

# load weights

new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace('ema_model.', '')
    v = mx.array(v.numpy())
    
    # rename layers
    if len(k) < 1 or 'mel_spec.' in k or k in ('initted', 'step'):
        continue
    elif '.to_out' in k:
        k = k.replace('.to_out', '.to_out.layers')
    elif '.text_blocks' in k:
        k = k.replace('.text_blocks', '.text_blocks.layers')
    elif '.ff.ff.0.0' in k:
        k = k.replace('.ff.ff.0.0', '.ff.ff.layers.0.layers.0')
    elif '.ff.ff.2' in k:
        k = k.replace('.ff.ff.2', '.ff.ff.layers.2')
    elif '.time_mlp' in k:
        k = k.replace('.time_mlp', '.time_mlp.layers')
    elif '.conv1d' in k:
        k = k.replace('.conv1d', '.conv1d.layers')
    
    # reshape weights
    if '.dwconv.weight' in k:
        v = v.swapaxes(1, 2)
    elif '.conv1d.layers.0.weight' in k:
        v = v.swapaxes(1, 2)
    elif '.conv1d.layers.2.weight' in k:
        v = v.swapaxes(1, 2)
    
    new_state_dict[k] = v

cfm.load_weights(list(new_state_dict.items()))
cfm.eval()

target_rms = 0.1
path = Path("tests/test_en_1_ref_short")
audio, sr = torchaudio.load(Path(f"{path}.wav").expanduser())
audio = mx.array(audio.numpy())

rms = mx.sqrt(mx.mean(mx.square(audio)))
if rms < target_rms:
    audio = audio * target_rms / rms

ref_text = "Some call me nature, others call me mother nature."
gen_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."
text = [ref_text + " " + gen_text]
text = convert_char_to_pinyin(text)

sample_rate = 24_000
hop_length = 256
frames_per_second = sample_rate * hop_length

duration = int(27 * frames_per_second)
print(f"Generating {duration} seconds of audio...")

start_date = datetime.datetime.now()

mel, _ = cfm.sample(audio, text=text, duration=duration, steps=32, cfg_strength=1, sway_sampling_coef=None, seed=1234)

vocos = Vocos.from_pretrained("lucasnewman/vocos-mel-24khz")
wave = vocos.decode(mel)

torchaudio.save("tests/output.wav", torch.Tensor(np.array(wave)).unsqueeze(0), 24000)
