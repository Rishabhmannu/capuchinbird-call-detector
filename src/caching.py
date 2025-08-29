from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import tensorflow as tf

from config import load_inference_config, resolve_model_paths


def _hash_bytes(data: bytes) -> str:
    """Return a short hex hash for caching uploaded files."""
    return hashlib.sha256(data).hexdigest()[:16]


@st.cache_resource(show_spinner="Loading artifacts and model...")
def get_artifacts_and_model(
    artifacts_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Optional[str]], tf.keras.Model]:
    """
    Load inference_config and a Keras model once, cache across reruns.
    Returns: (config_dict, resolved_paths, keras_model)
    """
    cfg, art_dir = load_inference_config(artifacts_dir)
    paths = resolve_model_paths(cfg, art_dir)

    keras_path = paths.get("keras_model_path")
    if not keras_path:
        raise FileNotFoundError(
            "Keras model path not found or missing in artifacts.")

    # Load the .keras file (Keras 3 format). This is GPU/MPS compatible on Apple silicon.
    model = tf.keras.models.load_model(keras_path)

    return cfg, paths, model


@st.cache_data(show_spinner=False)
def cache_config_snapshot(cfg: Dict[str, Any]) -> str:
    """
    Cache a JSON snapshot of the loaded config for quick debug.
    Returns a short hash string representing the config content.
    """
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def cache_uploaded_audio(bytes_data: bytes) -> str:
    """
    Cache the raw bytes of an uploaded file. We don't persist to disk here; this
    is only to deduplicate processing when the same file is re-used in-session.
    Returns a short content hash key.
    """
    return _hash_bytes(bytes_data)


def read_bytes_like(file_obj_or_bytes) -> bytes:
    """
    Utility: get raw bytes whether input is `bytes` or a file-like object.
    """
    if isinstance(file_obj_or_bytes, bytes):
        return file_obj_or_bytes
    if hasattr(file_obj_or_bytes, "getvalue"):
        # e.g., Streamlit's UploadedFile
        return file_obj_or_bytes.getvalue()
    # Generic file-like
    data = file_obj_or_bytes.read()
    if isinstance(data, bytes):
        return data
    # Fallback: convert str -> bytes
    return str(data).encode("utf-8")
