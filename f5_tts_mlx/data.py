from functools import partial
import hashlib
from pathlib import Path

import mlx.core as mx
import mlx.data as dx
import numpy as np

from einops.array_api import rearrange

from mlx.data.datasets.common import (
    CACHE_DIR,
    ensure_exists,
    urlretrieve_with_progress,
    file_digest,
    gzip_decompress,
)

from f5_tts_mlx.modules import log_mel_spectrogram

# utilities


def files_with_extensions(dir: Path, extensions: list = ["wav"]):
    files = []
    for ext in extensions:
        files.extend(list(dir.rglob(f"*.{ext}")))
    files = sorted(files)

    return [{"file": f.as_posix().encode("utf-8")} for f in files]


# transforms


def _load_transcript_file(sample):
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))
    transcript_file = audio_file.with_suffix(".normalized.txt")
    sample["transcript_file"] = transcript_file.as_posix().encode("utf-8")
    return sample


def _load_transcript(sample):
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))
    transcript_file = audio_file.with_suffix(".normalized.txt")
    if not transcript_file.exists():
        return dict()

    transcript = np.array(
        list(transcript_file.read_text().strip().encode("utf-8")), dtype=np.int8
    )
    sample["transcript"] = transcript
    return sample


def _load_cached_mel_spec(sample, max_duration=5):
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))
    mel_file = audio_file.with_suffix(".mel.npy.npz")
    mel_spec = mx.load(mel_file.as_posix())["arr_0"]
    mel_len = mel_spec.shape[1]

    if mel_len > int(max_duration * 93.75):
        return dict()

    sample["mel_spec"] = mel_spec
    sample["mel_len"] = mel_len
    del sample["file"]
    return sample


def _load_audio_file(sample):
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))
    audio = np.array(list(audio_file.read_bytes()), dtype=np.int8)
    sample["audio"] = audio
    return sample


def _to_mel_spec(sample):
    audio = rearrange(mx.array(sample["audio"]), "t 1 -> t")
    mel_spec = log_mel_spectrogram(audio)
    sample["mel_spec"] = mel_spec
    sample["mel_len"] = mel_spec.shape[1]
    return sample


def _with_max_duration(sample, sample_rate=24_000, max_duration=30):
    audio_duration = sample["audio"].shape[0] / sample_rate
    if audio_duration > max_duration:
        return dict()
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
    target = str(target)

    dset = (
        dx.files_from_tar(target)
        .to_stream()
        .sample_transform(lambda s: s if bytes(s["file"]).endswith(b".wav") else dict())
        .sample_transform(_load_transcript_file)
        .read_from_tar(target, "transcript_file", "transcript")
        .read_from_tar(target, "file", "audio")
        .load_audio("audio", from_memory=True)
        .sample_transform(partial(_with_max_duration, max_duration=max_duration))
        .sample_transform(_to_mel_spec)
    )

    return dset


def load_dir(dir=None, max_duration=30):
    path = Path(dir).expanduser()

    files = files_with_extensions(path)
    print(f"Found {len(files)} files at {path}")

    dset = (
        dx.buffer_from_vector(files)
        .to_stream()
        .sample_transform(lambda s: s if bytes(s["file"]).endswith(b".wav") else dict())
        .sample_transform(_load_transcript)
        .sample_transform(partial(_load_cached_mel_spec, max_duration=max_duration))
        # .pad_to_multiple("mel_spec", dim=1, pad_multiple=1024, pad_value=0.0)
    )

    return dset
