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

Pretrained model weights are available [on Hugging Face](https://huggingface.co/lucasnewman/f5-tts-mlx).

You'll also need a vocabulary (see the `data` folder) and mel filterbanks (see the `assets` folder) to preprocess the text & audio before generating speech.

## Usage

See `examples/generate.py` for an example of generation.

You can load a pretrained model from Python like this:

```python
import mlx.core as mx

from f5_tts_mlx.cfm import CFM

vocab = ...
f5tts = CFM.from_pretrained("lucasnewman/f5-tts-mlx", vocab)
audio = f5tts.sample(...)
```

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
