"""
analysis/compute_spectral_eq.py
───────────────────────────────
Compute per-note spectral EQ correction H(f) = LTASE_orig / LTASE_synth and
store it in params.json as `spectral_eq: {freqs_hz, gains_db}`.

This acts as a "resonant cavity" / soundboard shaping layer: it captures the
spectral envelope difference between the original sample and the physics
synthesizer, encoding the instrument body's frequency response.

Usage:
    python analysis/compute_spectral_eq.py \
        --params analysis/params.json \
        --bank C:/SoundBanks/IthacaPlayer/ks-grand \
        --workers 4
"""

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow import of analysis.physics_synth from project root
sys.path.insert(0, '.')
from analysis.physics_synth import synthesize_note  # noqa: E402

# ── STFT / LTASE parameters ───────────────────────────────────────────────────

N_FFT = 8192
HOP   = 2048
SR    = 44100

# EQ frequency grid
N_EQ_POINTS = 64
EQ_F_MIN    = 20.0
EQ_F_MAX    = 20000.0

# 1/6-octave smoothing: half-width in octaves
SMOOTH_OCT = 1.0 / 12.0  # 1/12 oct each side = 1/6 oct window


# ── Global params dict (for multiprocessing) ─────────────────────────────────

_G_PARAMS: dict = {}
_G_BANK_DIR: str = ''


def _init_worker(params_dict: dict, bank_dir: str) -> None:
    """Initializer for Pool workers: loads shared data into globals."""
    global _G_PARAMS, _G_BANK_DIR
    _G_PARAMS   = params_dict
    _G_BANK_DIR = bank_dir


# ── Core per-sample function (top-level for pickling) ────────────────────────

def process_sample(key: str):
    """
    Compute spectral_eq for one sample key (e.g. 'm060_vel3').
    Returns (key, spectral_eq_dict) or (key, None) on error.
    """
    try:
        sample = _G_PARAMS['samples'][key]
        midi   = sample['midi']
        vel    = sample['vel']

        # ── Load original WAV ──────────────────────────────────────────────
        wav_name = f'm{midi:03d}-vel{vel}-f44.wav'
        wav_path = Path(_G_BANK_DIR) / wav_name
        if not wav_path.exists():
            print(f"  SKIP {key}: WAV not found at {wav_path}")
            return key, None

        orig_stereo, sr_orig = sf.read(str(wav_path), dtype='float32', always_2d=True)
        # Convert to mono
        orig_mono = orig_stereo.mean(axis=1).astype(np.float64)

        # ── Synthesize (no EQ, eq_strength=0 is the default) ──────────────
        synth_stereo = synthesize_note(
            sample,
            duration=None,          # uses duration_s from params
            sr=SR,
            soundboard_strength=0.0,
            beat_scale=1.0,
            pan_spread=0.55,
        )  # returns (N, 2) float32
        synth_mono = synth_stereo.mean(axis=1).astype(np.float64)

        # Align lengths to the shorter of the two
        n = min(len(orig_mono), len(synth_mono))
        orig_mono  = orig_mono[:n]
        synth_mono = synth_mono[:n]

        # ── LTASE via STFT ─────────────────────────────────────────────────
        ltase_orig = _compute_ltase(orig_mono)
        ltase_synth = _compute_ltase(synth_mono)

        # Frequency axis for N_FFT/2+1 bins
        freqs_fft = np.linspace(0.0, SR / 2.0, N_FFT // 2 + 1)

        # ── Ratio with regularization ──────────────────────────────────────
        eps = max(ltase_synth.max(), ltase_orig.max()) * 1e-3
        H   = (ltase_orig + eps) / (ltase_synth + eps)

        # ── 1/6-octave Gaussian smoothing ─────────────────────────────────
        H_smooth = _smooth_octave(H, freqs_fft, SMOOTH_OCT)

        # ── Convert to dB and normalize (mean above 100 Hz = 0 dB) ────────
        H_db   = 20.0 * np.log10(np.maximum(H_smooth, 1e-10))
        mask   = freqs_fft > 100.0
        if mask.any():
            H_db -= H_db[mask].mean()
        else:
            H_db -= H_db.mean()

        # ── Resample to 64 log-spaced points ──────────────────────────────
        eq_freqs = np.logspace(
            np.log10(EQ_F_MIN), np.log10(EQ_F_MAX), N_EQ_POINTS
        )
        # np.interp requires sorted xp; freqs_fft is already ascending
        eq_gains = np.interp(eq_freqs, freqs_fft, H_db)

        max_gain  = eq_gains.max()
        range_db  = eq_gains.max() - eq_gains.min()
        print(f"  {key} ... EQ peak={max_gain:.1f}dB range={range_db:.1f}dB")

        return key, {
            'freqs_hz': eq_freqs.tolist(),
            'gains_db': eq_gains.tolist(),
        }

    except Exception:
        print(f"  ERROR {key}:\n{traceback.format_exc()}")
        return key, None


# ── STFT helpers ──────────────────────────────────────────────────────────────

def _compute_ltase(audio: np.ndarray) -> np.ndarray:
    """
    Long-Time Average Spectral Envelope via STFT.
    Returns 1-D array of shape (N_FFT//2 + 1,): mean magnitude over time.
    """
    window = np.hanning(N_FFT)
    n_bins = N_FFT // 2 + 1
    frames = []
    for start in range(0, len(audio) - N_FFT + 1, HOP):
        frame = audio[start: start + N_FFT] * window
        spec  = np.fft.rfft(frame, n=N_FFT)
        frames.append(np.abs(spec))
    if not frames:
        return np.ones(n_bins)
    Z = np.stack(frames, axis=-1)        # (n_bins, n_frames)
    return Z.mean(axis=-1)               # (n_bins,)


def _smooth_octave(H: np.ndarray, freqs: np.ndarray, half_width_oct: float) -> np.ndarray:
    """
    Per-bin 1/6-octave smoothing: for each bin i, average all bins j where
    freqs[j] is within [freqs[i] / 2^half_width_oct, freqs[i] * 2^half_width_oct].
    DC bin (freq=0) is left unchanged.
    """
    factor = 2.0 ** half_width_oct
    H_smooth = np.empty_like(H)
    for i, f in enumerate(freqs):
        if f <= 0.0:
            H_smooth[i] = H[i]
            continue
        f_lo = f / factor
        f_hi = f * factor
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        H_smooth[i] = H[mask].mean() if mask.any() else H[i]
    return H_smooth


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute per-note spectral EQ and store in params.json'
    )
    parser.add_argument('--params',  default='analysis/params.json',
                        help='Path to params.json')
    parser.add_argument('--bank',    default='C:/SoundBanks/IthacaPlayer/ks-grand',
                        help='Directory with original WAV files')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    params_path = Path(args.params)
    bank_dir    = args.bank

    print(f"Loading {params_path} ...")
    with open(params_path, 'r') as f:
        data = json.load(f)

    keys = list(data['samples'].keys())
    print(f"Processing {len(keys)} samples with {args.workers} workers ...")

    results: dict = {}

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(data, bank_dir),
    ) as pool:
        for key, eq in pool.imap_unordered(process_sample, keys):
            if eq is not None:
                results[key] = eq

    # Write back to params.json
    n_ok = 0
    for key, eq in results.items():
        data['samples'][key]['spectral_eq'] = eq
        n_ok += 1

    print(f"\nSuccessfully computed EQ for {n_ok}/{len(keys)} samples.")
    print(f"Saving {params_path} ...")
    with open(params_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Done.")


if __name__ == '__main__':
    main()
