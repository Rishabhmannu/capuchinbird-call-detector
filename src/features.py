from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf


def get_params_from_cfg(cfg: Dict) -> Tuple[int, int, int, int]:
    """
    Pull STFT + windowing params from the loaded config.
    Returns: (sr, window_samples, hop_samples, (frame_len, frame_step))
    """
    sr = int(cfg["samplerate"])
    win = int(cfg["window_samples"])
    hop = int(cfg["hop_samples"])
    frame_len = int(cfg["stft_frame_length"])
    frame_step = int(cfg["stft_frame_step"])
    return sr, win, hop, frame_len, frame_step


def stft_mag(
    wav: np.ndarray,
    frame_length: int,
    frame_step: int,
) -> tf.Tensor:
    """
    Compute magnitude STFT, returning a tensor of shape (frames, bins, 1).
    """
    # Ensure tf float tensor
    wav_tf = tf.convert_to_tensor(wav, dtype=tf.float32)
    spec = tf.signal.stft(
        wav_tf,
        frame_length=frame_length,
        frame_step=frame_step,
    )
    mag = tf.abs(spec)
    mag = tf.expand_dims(mag, axis=-1)  # (frames, bins, 1)
    return mag


def pad_or_trim(wav: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad with zeros or trim to exactly target_len samples.
    """
    n = wav.shape[0]
    if n == target_len:
        return wav
    if n > target_len:
        return wav[:target_len]
    # pad
    out = np.zeros((target_len,), dtype=np.float32)
    out[:n] = wav
    return out


def make_windows(wav: np.ndarray, window_samples: int, hop_samples: int) -> List[np.ndarray]:
    """
    Slice a long waveform into overlapping windows (last one padded if needed).
    Returns a list of arrays each length == window_samples.
    """
    n = len(wav)
    if n <= 0:
        return []
    if n <= window_samples:
        return [pad_or_trim(wav, window_samples)]

    windows: List[np.ndarray] = []
    start = 0
    while start + window_samples <= n:
        windows.append(wav[start: start + window_samples])
        start += hop_samples

    # tail (pad if leftover)
    if start < n:
        windows.append(pad_or_trim(wav[start:], window_samples))
    return windows


def windows_to_specs(
    windows: Iterable[np.ndarray],
    frame_length: int,
    frame_step: int,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """
    Convert an iterable of waveform windows into a batched tf.data.Dataset of spectrograms.
    """
    specs = []
    for w in windows:
        specs.append(stft_mag(w, frame_length, frame_step))
    ds = tf.data.Dataset.from_tensor_slices(specs).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
