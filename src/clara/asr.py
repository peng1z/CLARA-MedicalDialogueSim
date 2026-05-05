"""ASR module: converts audio (or direct text) into transcript + paralinguistic features.

transcribe()       — text-override path (mock ASR, real acoustic features not available)
transcribe_audio() — WAV file path using VOSK for ASR + librosa for acoustic features

Acoustic feature extraction is ported from VietMed.ipynb (ClaraASRModule repo):
  - avg_pitch    : mean fundamental frequency via librosa.pyin (Hz)
  - speaking_rate: tempo via librosa.beat.beat_track (BPM)
  - pause_duration: total pause time from low-energy RMS frames (seconds)
  - jitter       : pitch instability (mean |diff(f0)| / mean(f0))
"""
from typing import Dict, Optional
from pathlib import Path

try:
    import numpy as np
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Acoustic feature extraction (ported from VietMed.ipynb)
# ---------------------------------------------------------------------------

def _compute_jitter(f0) -> Optional[float]:
    """Pitch instability: mean absolute pitch difference divided by mean pitch.

    Ported directly from VietMed.ipynb compute_jitter().
    Returns None if fewer than 2 voiced frames exist.
    """
    import numpy as np
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 2:
        return None
    diffs = np.abs(np.diff(f0))
    return float(np.mean(diffs) / np.mean(f0))


def _extract_audio_features(file_path: str) -> Dict:
    """Extract prosodic/paralinguistic features from an audio file using librosa.

    Ported directly from VietMed.ipynb extract_audio_features().

    Returns:
        dict with keys:
          avg_pitch      (float, Hz)       — mean fundamental frequency
          speaking_rate  (float, BPM)      — estimated tempo
          pause_duration (float, seconds)  — total low-energy (pause) time
          jitter         (float or None)   — pitch instability ratio
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(file_path, sr=None)

    # Pitch (fundamental frequency)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300)
    avg_pitch = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0.0

    # Speaking rate (tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    speaking_rate = float(tempo)

    # Pause duration: frames where RMS energy < 10th percentile
    energy = librosa.feature.rms(y=y)[0]
    low_energy_frames = np.nonzero(energy < np.percentile(energy, 10))[0]
    pauses = librosa.frames_to_time(low_energy_frames, sr=sr)
    total_pause_time = float(len(pauses) * (len(y) / sr / len(energy)))

    # Jitter
    jitter = _compute_jitter(f0)

    return {
        "avg_pitch": avg_pitch,
        "speaking_rate": speaking_rate,
        "pause_duration": total_pause_time,
        "jitter": jitter,
    }


def _acoustic_to_paralinguistic(features: Dict) -> Dict:
    """Map numeric librosa features to the categorical paralinguistic labels
    used downstream by feedback.py and nlp.py.

    Thresholds are derived from VietMed.ipynb dataset distributions:
      - pitch:   median ~130 Hz, low < 100 Hz
      - tempo:   slow < 80 BPM, fast > 140 BPM
      - pauses:  long pause > 2.0 s total low-energy time
      - jitter:  high instability > 0.05
    """
    avg_pitch = features.get("avg_pitch", 0.0)
    speaking_rate = features.get("speaking_rate", 0.0)
    pause_duration = features.get("pause_duration", 0.0)
    jitter = features.get("jitter") or 0.0

    pitch_label = "low" if avg_pitch < 100 else ("high" if avg_pitch > 200 else "normal")
    pacing_label = "slow" if speaking_rate < 80 else ("fast" if speaking_rate > 140 else "normal")
    emphasis_label = "high" if jitter > 0.05 or pause_duration > 2.0 else "normal"

    return {
        "emphasis": emphasis_label,
        "pacing": pacing_label,
        "pitch": pitch_label,
        # raw numeric values for downstream modules that want them
        "avg_pitch_hz": avg_pitch,
        "speaking_rate_bpm": speaking_rate,
        "pause_duration_sec": pause_duration,
        "jitter": jitter,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(text_override: str = None) -> Dict:
    """Return a transcription dict from a text string (no audio file).

    Paralinguistic features are heuristic-only since there is no audio to
    analyse. Use transcribe_audio() when a WAV file is available.
    """
    text = text_override or "(no input)"

    emphasis = "high" if "!" in text or text.isupper() else "normal"
    pacing = "slow" if "..." in text or text.count(",") > 2 else "normal"
    pitch = "low" if any(w in text.lower() for w in ["sigh", "tired"]) else "normal"

    return {
        "text": text.strip(),
        "paralinguistic": {
            "emphasis": emphasis,
            "pacing": pacing,
            "pitch": pitch,
            "avg_pitch_hz": None,
            "speaking_rate_bpm": None,
            "pause_duration_sec": None,
            "jitter": None,
        },
    }


def transcribe_audio(wav_path: str) -> Dict:
    """Transcribe a WAV file using VOSK and extract acoustic features via librosa.

    Requirements:
      - pip install vosk soundfile librosa numpy
      - A VOSK model downloaded (set VOSK_MODEL_PATH env var or place under
        data/vosk-model/ in the repo root)

    Acoustic features (from VietMed.ipynb pipeline):
      avg_pitch_hz, speaking_rate_bpm, pause_duration_sec, jitter

    Raises RuntimeError if VOSK or its model is unavailable.
    """
    # --- ASR via VOSK ---
    try:
        import os
        import json
        import soundfile as sf
        import numpy as np
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        raise RuntimeError(
            "VOSK or soundfile not installed. Run: pip install vosk soundfile"
        ) from e

    model_path = os.environ.get("VOSK_MODEL_PATH")
    if not model_path:
        model_path = Path(__file__).resolve().parents[2] / "data" / "vosk-model"

    if not Path(model_path).exists():
        raise RuntimeError(
            f"VOSK model not found at {model_path}. "
            "Download a model and set VOSK_MODEL_PATH."
        )

    data, samplerate = sf.read(wav_path)
    if samplerate != 16000:
        raise RuntimeError(
            "VOSK requires 16000 Hz mono WAV input. Please resample your audio to 16k mono."
        )

    model = Model(str(model_path))
    rec = KaldiRecognizer(model, samplerate)

    if len(data.shape) > 1:
        data = data[:, 0]

    pcm_data = (data * 32767).astype(np.int16).tobytes()
    if rec.AcceptWaveform(pcm_data):
        text = json.loads(rec.Result()).get("text", "")
    else:
        text = json.loads(rec.FinalResult()).get("text", "")

    # --- Acoustic features via librosa (ported from VietMed.ipynb) ---
    if _LIBROSA_AVAILABLE:
        try:
            features = _extract_audio_features(wav_path)
            paralinguistic = _acoustic_to_paralinguistic(features)
        except Exception:
            paralinguistic = {
                "emphasis": "normal", "pacing": "normal", "pitch": "normal",
                "avg_pitch_hz": None, "speaking_rate_bpm": None,
                "pause_duration_sec": None, "jitter": None,
            }
    else:
        paralinguistic = {
            "emphasis": "normal", "pacing": "normal", "pitch": "normal",
            "avg_pitch_hz": None, "speaking_rate_bpm": None,
            "pause_duration_sec": None, "jitter": None,
        }

    return {"text": text, "paralinguistic": paralinguistic}
