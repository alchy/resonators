"""
gui/routers/audio.py
─────────────────────
Spectrum analysis endpoint for generated WAV files.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException

from gui.routers.sessions import session_dir

router = APIRouter()

N_FFT = 4096
N_POINTS = 512  # spectrum points returned to frontend


@router.get("/{name}/spectrum/{filename}")
def get_spectrum(name: str, filename: str):
    """Compute magnitude spectrum of a generated WAV and return as JSON."""
    wav_path = session_dir(name) / "generated" / filename
    if not wav_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    # Use first 2 seconds for spectrum (attack + sustain)
    n_use = min(len(mono), 2 * sr)
    segment = mono[:n_use]

    # Welch-like average: 4 overlapping windows
    hop = len(segment) // 5
    window = np.hanning(N_FFT)
    mags = []
    for start in range(0, len(segment) - N_FFT + 1, max(hop, 1)):
        frame = segment[start:start + N_FFT] * window
        mag = np.abs(np.fft.rfft(frame, n=N_FFT))
        mags.append(mag)

    if not mags:
        # Too short — just FFT the whole thing zero-padded
        frame = np.zeros(N_FFT, dtype=np.float32)
        frame[:len(segment)] = segment
        frame *= np.hanning(N_FFT)
        mags = [np.abs(np.fft.rfft(frame, n=N_FFT))]

    avg_mag = np.mean(mags, axis=0)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)

    # Log-spaced resampling 20 Hz–20 kHz
    f_min, f_max = 20.0, min(20000.0, sr / 2 - 1)
    log_freqs = np.logspace(np.log10(f_min), np.log10(f_max), N_POINTS)
    log_mags = np.interp(log_freqs, freqs, avg_mag)

    # Convert to dB, floor at -80
    db = 20.0 * np.log10(np.maximum(log_mags, 1e-4))
    db = np.maximum(db, -80.0)

    return {
        "freqs": log_freqs.tolist(),
        "magnitudes_db": db.tolist(),
        "sr": int(sr),
        "duration_s": round(len(audio) / sr, 2),
    }
