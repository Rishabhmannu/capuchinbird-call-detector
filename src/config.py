import json
import os
from typing import Dict, Any, Optional, Tuple

DEFAULT_CANDIDATE_DIRS = ("artifacts", "artifacts_capuchin")


def find_artifacts_dir(base_dir: Optional[str] = None) -> str:
    """
    Return the first existing artifacts directory from known candidates.
    If base_dir is provided, search inside it; otherwise use CWD.
    """
    root = base_dir or os.getcwd()
    for d in DEFAULT_CANDIDATE_DIRS:
        cand = os.path.join(root, d)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find artifacts directory. Tried: {DEFAULT_CANDIDATE_DIRS} under {root}"
    )


def load_inference_config(artifacts_dir: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Load inference_config.json from the artifacts directory.
    Returns (config_dict, artifacts_dir_used).
    """
    art_dir = artifacts_dir or find_artifacts_dir()
    cfg_path = os.path.join(art_dir, "inference_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"inference_config.json not found at: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    return cfg, art_dir


def resolve_model_paths(cfg: Dict[str, Any], artifacts_dir: str) -> Dict[str, Optional[str]]:
    """
    Resolve model paths (keras and SavedModel export) relative to artifacts_dir if needed.
    Returns a dict with absolute paths and flags indicating existence.
    """
    keras_rel = cfg.get("keras_model_path")
    saved_rel = cfg.get("savedmodel_dir")

    keras_path = (
        keras_rel if os.path.isabs(keras_rel) else os.path.join(
            artifacts_dir, os.path.basename(keras_rel))
    ) if keras_rel else None

    saved_dir = (
        saved_rel if (saved_rel and os.path.isabs(saved_rel)) else
        (os.path.join(artifacts_dir, os.path.basename(saved_rel)) if saved_rel else None)
    )

    return {
        "keras_model_path": keras_path if (keras_path and os.path.isfile(keras_path)) else None,
        "savedmodel_dir": saved_dir if (saved_dir and os.path.isdir(saved_dir)) else None,
    }


def summarize_environment() -> Dict[str, str]:
    """
    Minimal environment summary for display in the UI.
    """
    try:
        import tensorflow as tf  # noqa
        tf_ver = tf.__version__
    except Exception:
        tf_ver = "unknown"
    try:
        import numpy as np  # noqa
        np_ver = np.__version__
    except Exception:
        np_ver = "unknown"
    try:
        import librosa  # noqa
        librosa_ver = librosa.__version__
    except Exception:
        librosa_ver = "unknown"

    return {
        "tensorflow": tf_ver,
        "numpy": np_ver,
        "librosa": librosa_ver,
    }
