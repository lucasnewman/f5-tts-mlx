![F5 TTS diagram](f5tts.jpg)

# F5 TTS — MLX

Implementation of [F5-TTS](https://arxiv.org/abs/2410.06885), with the [MLX](https://github.com/ml-explore/mlx) framework.

F5 TTS is a non-autoregressive, zero-shot text-to-speech system using a flow-matching mel spectrogram generator with a diffusion transformer (DiT).

You can listen to a [sample here](https://s3.amazonaws.com/lucasnewman.datasets/f5tts/sample.wav) that was generated in ~11 seconds on an M3 Max MacBook Pro.

F5 is an evolution of [E2 TTS](https://arxiv.org/abs/2406.18009v2) and improves performance with ConvNeXT v2 blocks for the learned text alignment. This repository is based on the original Pytorch implementation available [here](https://github.com/SWivid/F5-TTS).

## Installation

```bash
pip install f5-tts-mlx
```

## Basic Usage

```bash
python -m f5_tts_mlx.generate --text "The quick brown fox jumped over the lazy dog."
```

You can also use a pipe to generate speech from the output of another process, for instance from a language model:

```bash
mlx_lm.generate --model mlx-community/Llama-3.2-1B-Instruct-4bit --verbose false \
--prompt "Write a concise paragraph explaning wavelets as used in signal processing." \
| python -m f5_tts_mlx.generate
```

## Voice Matching

If you want to use your own reference audio sample, make sure it's a mono, 24kHz wav file of around 5-10 seconds:

```bash
python -m f5_tts_mlx.generate \
--text "The quick brown fox jumped over the lazy dog."
--ref-audio /path/to/audio.wav
--ref-text "This is the caption for the reference audio."
```

You can convert an audio file to the correct format with ffmpeg like this:

```bash
ffmpeg -i /path/to/audio.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 /path/to/output_audio.wav
```

See [here](./f5_tts_mlx) for more options to customize generation.

## From Python

You can load a pretrained model from Python:

```python
from f5_tts_mlx.generate import generate

audio = generate(text = "Hello world.", ...)
```

Pretrained model weights are also available [on Hugging Face](https://huggingface.co/lucasnewman/f5-tts-mlx).

## Appreciation

[Yushen Chen](https://github.com/SWivid) for the original Pytorch implementation of F5 TTS and pretrained model.

[Phil Wang](https://github.com/lucidrains) for the E2 TTS implementation that this model is based on.

## Citations

```bibtex
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```

```bibtex
@inproceedings{Eskimez2024E2TE,
    title   = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author  = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270738197}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
