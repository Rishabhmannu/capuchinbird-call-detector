from __future__ import annotations

from itertools import groupby
from typing import List, Tuple

import numpy as np


def rle_runs(binary_array: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Run-length encode a binary array.
    Returns a list of (val, start_idx, end_idx) for each run.
    """
    a = np.asarray(binary_array, dtype=np.int32)
    if a.size == 0:
        return []
    changes = np.diff(a)
    starts = np.r_[0, np.where(changes != 0)[0] + 1]
    ends = np.r_[starts[1:], a.size]
    values = a[starts]
    return list(zip(values.tolist(), starts.tolist(), ends.tolist()))


def collapse_events(
    preds: np.ndarray,
    hop_sec: float,
    win_sec: float,
    k_consecutive: int = 2,
    min_gap_s: float = 0.4,
) -> List[Tuple[float, float]]:
    """
    Collapse window-level predictions into merged event time ranges.

    preds: 1D binary array (0/1)
    hop_sec: stride between window starts in seconds
    win_sec: window length in seconds
    k_consecutive: require at least k consecutive positive windows to form an event
    min_gap_s: merge events closer than this (in seconds)

    Returns: list of [start_s, end_s] ranges
    """
    events: List[Tuple[float, float]] = []
    for val, s, e in rle_runs(preds):
        if val == 1 and (e - s) >= k_consecutive:
            start_time = s * hop_sec
            end_time = (e - 1) * hop_sec + win_sec
            events.append([start_time, end_time])

    if not events:
        return []

    # merge close events
    merged = [events[0]]
    for st, en in events[1:]:
        prev_st, prev_en = merged[-1]
        if st - prev_en < min_gap_s:
            merged[-1][1] = max(prev_en, en)
        else:
            merged.append([st, en])
    return merged


def predict_to_events(
    probs: np.ndarray,
    threshold: float,
    hop_sec: float,
    win_sec: float,
    k_consecutive: int,
    min_gap_s: float,
) -> List[Tuple[float, float]]:
    """
    Convert probability outputs to event ranges using threshold + postprocessing.
    """
    preds = (probs >= threshold).astype(np.int32)
    return collapse_events(preds, hop_sec, win_sec, k_consecutive, min_gap_s)


def count_events(events: List[Tuple[float, float]]) -> int:
    """Return number of events detected."""
    return len(events)
