import sys
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Allow "src" to be importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from caching import get_artifacts_and_model, read_bytes_like  # noqa: E402
from audio_io import load_audio_16k_mono  # noqa: E402
from inference import run_detection  # noqa: E402
from config import summarize_environment  # noqa: E402
from visualize import plot_waveform_with_events, plot_spectrogram_with_events  # noqa: E402


st.set_page_config(page_title="Capuchinbird Call Detector", layout="wide")
st.title("Capuchinbird Call Detector")

# --- Load config & model once ---
with st.spinner("Loading artifacts and model..."):
    try:
        cfg, paths, model = get_artifacts_and_model()
        env = summarize_environment()
        load_ok = True
        error_msg = ""
    except Exception as e:
        cfg, paths, model, env = {}, {}, None, {}
        load_ok = False
        error_msg = str(e)

# --- Sidebar controls ---
st.sidebar.header("Detection Parameters")
th = st.sidebar.slider("Threshold", 0.1, 0.99, float(
    cfg.get("threshold", 0.68)), 0.01)
k_consec = st.sidebar.number_input(
    "k consecutive windows", 1, 10, int(cfg.get("k_consecutive", 2)))
min_gap = st.sidebar.slider("Min gap merge (s)", 0.1, 2.0, float(
    cfg.get("min_gap_seconds", 0.40)), 0.1)

# --- Status / Instructions ---
col1, col2 = st.columns([1, 2], gap="large")
with col1:
    st.subheader("Status")
    if load_ok:
        st.success("Artifacts & model loaded")
        st.write(f"Model path: `{paths.get('keras_model_path')}`")
    else:
        st.error(f"Config/model load failed: {error_msg}")
    st.write("Environment:")
    if env:
        st.json(env)

with col2:
    st.subheader("Instructions")
    st.markdown(
        """
        1) Upload a `.wav`, `.mp3`, `.flac`, or `.m4a` file **or** choose a sample.  
        2) Click **Detect Capuchinbird Calls** to run inference.  
        3) Adjust threshold / k / min-gap from the sidebar, then detect again.
        """
    )

st.divider()

# --- File uploader / sample picker ---
col_upload, col_sample = st.columns(2)
with col_upload:
    uploaded = st.file_uploader("Upload an audio file", type=[
                                "wav", "mp3", "flac", "m4a"])
with col_sample:
    sample_path = None
    samples_dir = PROJECT_ROOT / "samples"
    if samples_dir.exists():
        sample_files = sorted([f.name for f in samples_dir.iterdir(
        ) if f.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]])
        choice = st.selectbox("Or choose a sample", ["(none)"] + sample_files)
        if choice != "(none)":
            sample_path = samples_dir / choice

# --- Ready-to-run section ---
wav, sr, fname = None, None, None
raw_bytes = None

if uploaded or sample_path:
    if uploaded:
        raw_bytes = read_bytes_like(uploaded)
        wav, sr = load_audio_16k_mono(
            raw_bytes, target_sr=int(cfg["samplerate"]))
        fname = uploaded.name
    else:
        with open(sample_path, "rb") as f:
            raw_bytes = f.read()
        wav, sr = load_audio_16k_mono(
            raw_bytes, target_sr=int(cfg["samplerate"]))
        fname = sample_path.name

    st.info(f"File ready: **{fname}** (duration ≈ {len(wav)/sr:.1f} s)")
    # Streamlit will sniff format
    st.audio(io.BytesIO(raw_bytes), format="audio/mp3")

    run_button = st.button("🔍 Detect Capuchinbird Calls")

    if run_button:
        with st.spinner("Running detection..."):
            result = run_detection(
                wav=wav,
                cfg=cfg,
                model=model,
                threshold=th,
                k_consecutive=k_consec,
                min_gap_seconds=min_gap,
            )

        st.subheader(f"Results for: {fname}")
        st.write(f"Detected **{result['count']}** Capuchinbird call(s)")

        # Plots
        wf_fig = plot_waveform_with_events(
            wav, sr, result["events"], title="Waveform with detections")
        st.pyplot(wf_fig)

        spec_fig = plot_spectrogram_with_events(
            wav,
            sr,
            cfg["stft_frame_length"],
            cfg["stft_frame_step"],
            result["events"],
            title="Spectrogram with detections",
        )
        st.pyplot(spec_fig)

        # Probability curve
        probs = np.asarray(result["probs"], dtype=float)
        hop_sec = float(result["hop_sec"])
        times = np.arange(len(probs)) * hop_sec

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(times, probs, linewidth=1.0)
        ax.axhline(th, color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Probability")
        ax.set_title("Per-window probability (model output)")
        ax.grid(True, alpha=0.2)
        for stt, enn in result["events"]:
            ax.axvspan(stt, enn, color="tab:orange", alpha=0.15, lw=0)
        st.pyplot(fig)

        # Events table + CSV download
        if result["events"]:
            df = pd.DataFrame(result["events"], columns=["start_s", "end_s"])
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download events CSV", data=csv_bytes,
                               file_name=f"{Path(fname).stem}_events.csv", mime="text/csv")

        with st.expander("Parameters used"):
            st.json(result["params"])
