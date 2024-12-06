from functools import partial
import hashlib
from pathlib import Path
import tarfile

import mlx.core as mx
import mlx.data as dx
import numpy as np
import os

from mlx.data.datasets.common import (
    CACHE_DIR,
    ensure_exists,
    urlretrieve_with_progress,
    file_digest,
    gzip_decompress,
)

from f5_tts_mlx.audio import log_mel_spectrogram
from f5_tts_mlx.utils import list_str_to_idx

SAMPLE_RATE = 24_000

# utilities


def files_with_extensions(dir: Path, extensions: list = ["wav"]):
    files = []
    for ext in extensions:
        files.extend(list(dir.rglob(f"*.{ext}")))
    files = sorted(files)

    return [{"file": mx.array(f.as_posix().encode("utf-8"))} for f in files]


def calculate_wav_duration(file_path):
    # assumptions
    bit_depth = 16
    num_channels = 1

    bytes_per_sample = bit_depth // 8
    bytes_per_second = SAMPLE_RATE * num_channels * bytes_per_sample

    file_size = os.path.getsize(file_path)
    duration_seconds = file_size / bytes_per_second

    return duration_seconds


# transforms

vocab = {chr(i): i for i in range(256)}


def _load_transcript(sample):
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))
    if not audio_file.suffix == ".wav":
        return dict()

    transcript_file = audio_file.with_suffix(".normalized.txt")
    if not transcript_file.exists():
        return dict()

    text = transcript_file.read_text().strip()
    sample["transcript"] = mx.array(list_str_to_idx(text, vocab))
    return sample


def _load_audio_file(sample, max_duration=10):
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))

    duration = calculate_wav_duration(audio_file)
    if duration > max_duration:
        return dict()

    audio = np.array(list(audio_file.read_bytes()), dtype=np.uint8)
    sample["audio"] = audio
    return sample


def _to_mel_spec(sample):
    audio = mx.squeeze(mx.array(sample["audio"]), axis=-1)
    mel_spec = log_mel_spectrogram(audio)
    sample["mel_spec"] = mel_spec
    sample["mel_len"] = mel_spec.shape[1]
    return sample


# dataset loading

SPLITS = {
    "dev-clean": (
        "https://www.openslr.org/resources/141/dev_clean.tar.gz",
        "2c1f5312914890634cc2d15783032ff3",
    ),
    "dev-other": (
        "https://www.openslr.org/resources/141/dev_other.tar.gz",
        "62d3a80ad8a282b6f31b3904f0507e4f",
    ),
    "test-clean": (
        "https://www.openslr.org/resources/141/test_clean.tar.gz",
        "4d373d453eb96c0691e598061bbafab7",
    ),
    "test-other": (
        "https://www.openslr.org/resources/141/test_other.tar.gz",
        "dbc0959d8bdb6d52200595cabc9995ae",
    ),
    "train-clean-100": (
        "https://www.openslr.org/resources/141/train_clean_100.tar.gz",
        "6df668d8f5f33e70876bfa33862ad02b",
    ),
    "train-clean-360": (
        "https://www.openslr.org/resources/141/train_clean_360.tar.gz",
        "382eb3e64394b3da6a559f864339b22c",
    ),
    "train-other-500": (
        "https://www.openslr.org/resources/141/train_other_500.tar.gz",
        "a37a8e9f4fe79d20601639bf23d1add8",
    ),
}


def load_libritts_r_tarfile(
    root=None, split="dev-clean", quiet=False, validate_download=True
):
    """Fetch the libritts_r TAR archive and return the path to it for manual processing.

    Args:
        root (Path or str, optional): The The directory to load/save the data. If
            none is given the ``~/.cache/mlx.data/libritts_r`` is used.
        split (str): The split to use. It should be one of dev-clean,
            dev-other, test-clean, test-other, train-clean-100,
            train-clean-360, train-other-500 .
        quiet (bool): If true do not show download (and possibly decompression)
            progress.
    """
    if split not in SPLITS:
        raise ValueError(
            f"Unknown libritts_r split '{split}'. It should be one of [{', '.join(SPLITS.keys())}]"
        )

    if root is None:
        root = CACHE_DIR / "libritts_r"
    else:
        root = Path(root)
    ensure_exists(root)

    url, target_hash = SPLITS[split]
    filename = Path(url).name
    target_compressed = root / filename
    target = root / filename.replace(".gz", "")

    if not target.is_file():
        if not target_compressed.is_file():
            urlretrieve_with_progress(url, target_compressed, quiet=quiet)
            if validate_download:
                h = file_digest(target_compressed, hashlib.md5(), quiet=quiet)
                if h.hexdigest() != target_hash:
                    raise RuntimeError(
                        f"[libritts_r] File download corrupted md5sums don't match. Please manually delete {str(target_compressed)}."
                    )

        gzip_decompress(target_compressed, target, quiet=quiet)
        target_compressed.unlink()

    return target


def load_libritts_r(
    root=None, split="dev-clean", quiet=False, validate_download=True, max_duration=30
):
    """Load the libritts_r dataset directly from the TAR archive.

    Args:
        root (Path or str, optional): The The directory to load/save the data. If
            none is given the ``~/.cache/mlx.data/libritts_r`` is used.
        split (str): The split to use. It should be one of dev-clean,
            dev-other, test-clean, test-other, train-clean-100,
            train-clean-360, train-other-500 .
        quiet (bool): If true do not show download (and possibly decompression)
            progress.
    """

    target = load_libritts_r_tarfile(
        root=root, split=split, quiet=quiet, validate_download=validate_download
    )

    path = Path(target.parent) / "LibriTTS_R" / split

    tar = tarfile.open(target)
    tar.extractall(path=target.parent)
    tar.close()

    return load_dir(path, max_duration=max_duration), path


def load_dir(dir=None, max_duration=30):
    path = Path(dir).expanduser()

    files = files_with_extensions(path)
    print(f"Found {len(files)} files at {path}")

    dset = (
        dx.buffer_from_vector(files)
        .to_stream()
        .sample_transform(lambda s: s if bytes(s["file"]).endswith(b".wav") else dict())
        .sample_transform(_load_transcript)
        .sample_transform(partial(_load_audio_file, max_duration=max_duration))
        .load_audio("audio", from_memory=True)
        .sample_transform(_to_mel_spec)
    )

    return dset
