from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def _overlay_events(ax, events: Iterable[Tuple[float, float]], color="tab:orange", alpha=0.25):
    """
    Shade time spans on an existing axis for each event [start_s, end_s].
    """
    for (st, en) in events:
        ax.axvspan(st, en, color=color, alpha=alpha, lw=0)


def plot_waveform_with_events(
    wav: np.ndarray,
    sr: int,
    events: Optional[List[Tuple[float, float]]] = None,
    title: str = "Waveform",
    figsize=(10, 3),
):
    """
    Returns a matplotlib Figure for the waveform with optional event overlays.
    """
    dur = wav.shape[0] / float(sr) if sr else len(wav)
    t = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, wav, linewidth=0.8)
    ax.set_xlim(0, max(1e-6, dur))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if events:
        _overlay_events(ax, events)
    fig.tight_layout()
    return fig


def compute_stft_mag(
    wav: np.ndarray,
    frame_length: int,
    frame_step: int,
) -> np.ndarray:
    """
    Compute |STFT| magnitude (frames x bins) as numpy array (no channel dim),
    for visualization only.
    """
    x = tf.convert_to_tensor(wav, dtype=tf.float32)
    spec = tf.signal.stft(x, frame_length=frame_length, frame_step=frame_step)
    mag = tf.abs(spec).numpy()
    return mag


def plot_spectrogram_with_events(
    wav: np.ndarray,
    sr: int,
    frame_length: int,
    frame_step: int,
    events: Optional[List[Tuple[float, float]]] = None,
    title: str = "Spectrogram (log-magnitude)",
    figsize=(10, 4),
):
    """
    Returns a matplotlib Figure showing a log-magnitude spectrogram with optional event overlays.
    X-axis in seconds, Y-axis is frequency bins.
    """
    mag = compute_stft_mag(wav, frame_length, frame_step)  # (frames, bins)
    # log for display
    mag_log = np.log1p(mag)

    # time axis (frame centers)
    hop_sec = frame_step / float(sr)
    win_sec = frame_length / float(sr)
    n_frames = mag_log.shape[0]
    times = np.arange(n_frames) * hop_sec + win_sec / 2.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mag_log.T,
        origin="lower",
        aspect="auto",
        extent=[times.min(), times.max(), 0, mag_log.shape[1]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if events:
        # draw vertical spans matching event boundaries
        for (st, en) in events:
            ax.axvspan(st, en, color="white", alpha=0.15, lw=0)

    fig.tight_layout()
    return fig
