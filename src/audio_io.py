import io
import os
from typing import Tuple, Union

import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf


VALID_EXTS = (".wav", ".mp3", ".flac", ".m4a")


def is_valid_audio_file(path: str) -> bool:
    """Check if path has a valid audio extension."""
    return path.lower().endswith(VALID_EXTS)


def load_audio_16k_mono(
    file_or_bytes: Union[str, bytes, io.BytesIO],
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32, resampled to target_sr.
    Accepts:
      - file path (str)
      - raw bytes (e.g. from Streamlit upload)
      - file-like object
    Returns: (audio np.ndarray [n_samples], sr)
    """
    if isinstance(file_or_bytes, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(file_or_bytes), dtype="float32")
    elif hasattr(file_or_bytes, "read"):  # file-like
        data, sr = sf.read(file_or_bytes, dtype="float32")
    else:  # assume string path
        data, sr = sf.read(str(file_or_bytes), dtype="float32")

    if data.ndim > 1:
        data = data.mean(axis=1)  # mono

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32), target_sr


def to_tensor(wav: np.ndarray) -> tf.Tensor:
    """
    Convert waveform np.ndarray to TF float32 tensor.
    """
    return tf.convert_to_tensor(wav, dtype=tf.float32)
