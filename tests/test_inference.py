import sys
from pathlib import Path

# Make src importable when running `pytest` from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from caching import get_artifacts_and_model  # noqa: E402
from audio_io import load_audio_16k_mono  # noqa: E402
from inference import run_detection  # noqa: E402


def _find_sample() -> Path | None:
    samples = PROJECT_ROOT / "samples"
    if not samples.exists():
        return None
    for p in sorted(samples.iterdir()):
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}:
            return p
    return None


@pytest.mark.skipif(_find_sample() is None, reason="No sample audio file found in samples/")
def test_basic_inference_on_sample():
    cfg, paths, model = get_artifacts_and_model()

    sample_path = _find_sample()
    wav, sr = load_audio_16k_mono(
        str(sample_path), target_sr=int(cfg["samplerate"]))

    result = run_detection(wav=wav, cfg=cfg, model=model)

    # Basic shape/type checks
    assert isinstance(result, dict)
    assert "count" in result and isinstance(result["count"], int)
    assert "events" in result and isinstance(result["events"], list)
    assert "probs" in result and isinstance(result["probs"], list)
    assert len(result["probs"]) > 0
    assert np.all((np.array(result["probs"]) >= 0.0) & (
        np.array(result["probs"]) <= 1.0))
