# Capuchinbird Call Detector 🎶🦜

A deep learning–powered tool for detecting **Capuchinbird calls** in audio recordings.  
Built with TensorFlow, Streamlit, and signal processing techniques, this project lets you upload or select audio samples, run real-time inference, and visualize detected bird calls with waveform plots, spectrograms, and probability curves.

---

## 📸 Screenshots

### App Overview & Controls
<img src="screenshots/screenshot1.png" width="800">

### Detection Results & Visualizations
<img src="screenshots/screenshot2.png" width="800">

### Spectrograms & Probability Curve
<img src="screenshots/screenshot3.png" width="800">

### Parameters & CSV Export
<img src="screenshots/screenshot4.png" width="800">

---

## ✨ Features
- Upload `.wav`, `.mp3`, `.flac`, or `.m4a` files, or choose from built-in sample recordings.
- Adjustable **detection parameters**: threshold, consecutive windows, and min-gap merging.
- Rich **visualizations**:
  - Waveform with highlighted detections
  - Spectrogram with detected spans
  - Per-window probability curves
- Download detected events as a **CSV** file.
- Built with **TensorFlow on Apple Silicon** (MPS) for fast local inference.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/capuchinbird-call-detector.git
cd capuchinbird-call-detector
````

### 2. Set up environment

We recommend using conda or venv:

```bash
conda create -n birdcall python=3.10
conda activate birdcall
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For development & testing:

```bash
pip install -r requirements-dev.txt
```

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 📂 Project Structure

```
birdcall-detection/
│
├─ app/                  # Streamlit app (UI)
│   └─ streamlit_app.py
├─ src/                  # Core modules (audio, features, inference, viz)
├─ artifacts/            # Saved model & configs
├─ samples/              # Demo audio files
├─ tests/                # Unit tests (pytest)
├─ requirements.txt      # Runtime dependencies
├─ requirements-dev.txt  # Dev/test dependencies
└─ README.md
```

---

## 🧪 Testing

Run sanity tests with:

```bash
pytest -q
```

---

## 📜 License

MIT License © 2025 Your Name

---

## 🌟 Acknowledgements

* TensorFlow team for Apple Silicon support
* Streamlit for rapid UI prototyping
* Inspiration from real-world ecoacoustics & birdcall monitoring

---

