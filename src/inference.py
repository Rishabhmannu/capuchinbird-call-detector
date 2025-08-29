from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from audio_io import load_audio_16k_mono
from features import (
    get_params_from_cfg,
    make_windows,
    windows_to_specs,
)
from postprocess import predict_to_events, count_events
from caching import get_artifacts_and_model


@dataclass
class DetectionParams:
    threshold: float
    k_consecutive: int
    min_gap_seconds: float
    samplerate: int
    window_samples: int
    hop_samples: int
    stft_frame_length: int
    stft_frame_step: int


def _params_from_cfg(cfg: Dict) -> DetectionParams:
    sr, win, hop, frame_len, frame_step = get_params_from_cfg(cfg)
    return DetectionParams(
        threshold=float(cfg.get("threshold", 0.68)),
        k_consecutive=int(cfg.get("k_consecutive", 2)),
        min_gap_seconds=float(cfg.get("min_gap_seconds", 0.40)),
        samplerate=int(sr),
        window_samples=int(win),
        hop_samples=int(hop),
        stft_frame_length=int(frame_len),
        stft_frame_step=int(frame_step),
    )


def _prepare_dataset(
    wav: np.ndarray,
    P: DetectionParams,
    batch_size: int = 64,
) -> Tuple[tf.data.Dataset, float, float]:
    """
    Create a batched spectrogram dataset from the waveform.
    Returns: (tf.data.Dataset, hop_sec, win_sec)
    """
    windows = make_windows(wav, P.window_samples, P.hop_samples)
    ds = windows_to_specs(
        windows,
        frame_length=P.stft_frame_length,
        frame_step=P.stft_frame_step,
        batch_size=batch_size,
    )
    hop_sec = P.hop_samples / float(P.samplerate)
    win_sec = P.window_samples / float(P.samplerate)
    return ds, hop_sec, win_sec


def _infer_probs(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
) -> np.ndarray:
    """
    Run model inference on a batched dataset of spectrograms.
    Returns a 1D numpy array of probabilities.
    """
    probs = model.predict(ds, verbose=0).ravel()
    return probs.astype(np.float32)


def run_detection(
    wav: np.ndarray,
    cfg: Dict,
    model: tf.keras.Model,
    threshold: Optional[float] = None,
    k_consecutive: Optional[int] = None,
    min_gap_seconds: Optional[float] = None,
    batch_size: int = 64,
) -> Dict:
    """
    Core detection pipeline from waveform array to events.

    Returns dict with:
      - 'events': List[[start_s, end_s], ...]
      - 'count': int
      - 'probs': list of per-window probabilities
      - 'hop_sec', 'win_sec'
      - 'params': dict of effective parameters used
    """
    P = _params_from_cfg(cfg)
    if threshold is not None:
        P.threshold = float(threshold)
    if k_consecutive is not None:
        P.k_consecutive = int(k_consecutive)
    if min_gap_seconds is not None:
        P.min_gap_seconds = float(min_gap_seconds)

    ds, hop_sec, win_sec = _prepare_dataset(wav, P, batch_size=batch_size)
    probs = _infer_probs(model, ds)

    events = predict_to_events(
        probs=probs,
        threshold=P.threshold,
        hop_sec=hop_sec,
        win_sec=win_sec,
        k_consecutive=P.k_consecutive,
        min_gap_s=P.min_gap_seconds,
    )
    return {
        "events": events,
        "count": count_events(events),
        "probs": probs.tolist(),
        "hop_sec": hop_sec,
        "win_sec": win_sec,
        "params": {
            "threshold": P.threshold,
            "k_consecutive": P.k_consecutive,
            "min_gap_seconds": P.min_gap_seconds,
            "samplerate": P.samplerate,
            "window_samples": P.window_samples,
            "hop_samples": P.hop_samples,
            "stft_frame_length": P.stft_frame_length,
            "stft_frame_step": P.stft_frame_step,
        },
    }


def detect_from_path(
    path: str,
    threshold: Optional[float] = None,
    k_consecutive: Optional[int] = None,
    min_gap_seconds: Optional[float] = None,
    batch_size: int = 64,
) -> Dict:
    """
    Convenience wrapper: load audio from a filesystem path and run detection.
    """
    cfg, paths, model = get_artifacts_and_model()
    wav, sr = load_audio_16k_mono(path, target_sr=int(cfg["samplerate"]))
    return run_detection(
        wav=wav,
        cfg=cdf(cfg=cfg),  # see helper below
        model=model,
        threshold=threshold,
        k_consecutive=k_consecutive,
        min_gap_seconds=min_gap_seconds,
        batch_size=batch_size,
    )


def detect_from_bytes(
    data: bytes,
    threshold: Optional[float] = None,
    k_consecutive: Optional[int] = None,
    min_gap_seconds: Optional[float] = None,
    batch_size: int = 64,
) -> Dict:
    """
    Convenience wrapper: load audio from raw bytes (e.g., Streamlit upload) and run detection.
    """
    cfg, paths, model = get_artifacts_and_model()
    wav, sr = load_audio_16k_mono(data, target_sr=int(cfg["samplerate"]))
    return run_detection(
        wav=wav,
        cfg=cdf(cfg=cfg),
        model=model,
        threshold=threshold,
        k_consecutive=k_consecutive,
        min_gap_seconds=min_gap_seconds,
        batch_size=batch_size,
    )


# --- Helper to guard against accidental mutation of the loaded config ---
def cdf(cfg: Dict) -> Dict:
    """Return a shallow copy of config dict (avoid accidental mutation)."""
    return dict(cfg)
