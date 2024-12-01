"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from pathlib import Path

import mlx.core as mx

from einops.array_api import reduce
import einx

from huggingface_hub import snapshot_download

import jieba
from pypinyin import lazy_pinyin, Style

jieba.setLogLevel(20)

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def lens_to_mask(
    t: mx.array,
    length: int | None = None,
) -> mx.array:  # Bool['b n']
    if not exists(length):
        length = t.max()

    seq = mx.arange(length)
    return einx.less("n, b -> b n", seq, t)


def mask_from_start_end_indices(
    seq_len: mx.array,
    start: mx.array,
    end: mx.array,
    max_length: int | None = None,
):
    max_seq_len = max_length # default(max_length, seq_len.max().item())
    seq = mx.arange(max_seq_len).astype(mx.int32)
    return einx.greater_equal("n, b -> b n", seq, start) & einx.less(
        "n, b -> b n", seq, end
    )


def mask_from_frac_lengths(
    seq_len: mx.array,
    frac_lengths: mx.array,
    max_length: int | None = None,
):
    lengths = (frac_lengths * seq_len).astype(mx.int32)
    max_start = seq_len - lengths

    rand = mx.random.uniform(0, 1, frac_lengths.shape)

    start = mx.maximum((max_start * rand).astype(mx.int32), 0)
    end = start + lengths

    out = mask_from_start_end_indices(seq_len, start, end, max_length)

    if exists(max_length):
        out = pad_to_length(out, max_length)

    return out


def maybe_masked_mean(t: mx.array, mask: mx.array | None = None) -> mx.array:
    if not exists(mask):
        return t.mean(dim=1)

    t = einx.where("b n, b n d, -> b n d", mask, t, 0.0)
    num = reduce(t, "b n d -> b d", "sum")
    den = reduce(mask.astype(mx.int32), "b n -> b", "sum")

    return einx.divide("b d, b -> b d", num, mx.maximum(den, 1))


def pad_to_length(t: mx.array, length: int, value=0):
    ndim = t.ndim
    seq_len = t.shape[-1]
    if length > seq_len:
        if ndim == 1:
            t = mx.pad(t, [(0, length - seq_len)], constant_values=value)
        elif ndim == 2:
            t = mx.pad(t, [(0, 0), (0, length - seq_len)], constant_values=value)
        else:
            raise ValueError(f"Unsupported padding dims: {ndim}")
    return t[..., :length]


def pad_sequence(t: mx.array, padding_value=0):
    max_len = max([i.shape[-1] for i in t])
    t = mx.array([pad_to_length(i, max_len, padding_value) for i in t])
    return t


# simple utf-8 tokenizer, since paper went character based


def list_str_to_tensor(text: list[str], padding_value=-1) -> mx.array:  # Int['b nt']:
    list_tensors = [mx.array([*bytes(t, "UTF-8")]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value=-1)
    return padded_tensor


# char tokenizer, based on custom dataset's extracted .txt file


def list_str_to_idx(
    text: list[str],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> mx.array:  # Int['b nt']:
    list_idx_tensors = [
        [vocab_char_map.get(c, 0) for c in t] for t in text
    ]  # pinyin or char style

    list_idx_tensors = [mx.array(t) for t in list_idx_tensors]
    text = pad_sequence(list_idx_tensors, padding_value=padding_value)
    return text


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # in case librispeech (orig no-pc) test-clean
    custom_trans = str.maketrans({";": ","})  # add custom trans here, to address oov
    for text in text_list:
        char_list = []
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(
                seg
            ):  # if pure chinese characters
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:  # if mixed chinese characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(
                                lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                            )
                        else:  # if is zh punc
                            char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# fetch model from hub


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.txt"],
        )
    )
    return model_path
