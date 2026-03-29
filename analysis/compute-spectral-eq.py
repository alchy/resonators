"""
analysis/compute-spectral-eq.py
───────────────────────────────
Phase 2 (pipeline step 2): compute per-note LTASE spectral EQ correction.

Method (LTASE — Long-Term Average Spectrum Envelope):
  H(f) = LTASE_orig(f) / LTASE_synth(f)
  Stored as spectral_eq: {freqs_hz, gains_db} in each sample entry of params JSON.
  Captures instrument body resonance: encodes frequency response difference
  between original recording and physics synthesizer output.

Modifies params JSON in-place (adds/replaces spectral_eq field per note).
Log:  runtime-logs/spectral-eq-log.txt  (auto-created, tee of stdout)

Usage:
    python -u analysis/compute-spectral-eq.py \\
        --params  analysis/params-ks-grand.json \\
        --bank    C:/SoundBanks/IthacaPlayer/ks-grand \\
        --workers 4

Arguments:
  --params   Params JSON to update in-place (output of extract-params.py)
  --bank     WAV sample bank directory (same as used in extract step)
  --workers  Parallel worker processes (default: 4)
"""

import argparse
import json
import math
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path


# ── Runtime logging (tee stdout → runtime-logs/spectral-eq-log.txt) ──────────

def _setup_log() -> None:
    log_dir = Path("runtime-logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "spectral-eq-log.txt"

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()

    sys.stdout = _Tee(sys.__stdout__, open(log_path, "w", encoding="utf-8", buffering=1))

import numpy as np
import soundfile as sf

# Allow import of analysis.physics_synth from project root
sys.path.insert(0, '.')
from analysis.physics_synth import synthesize_note  # noqa: E402

# ── STFT / LTASE parameters ───────────────────────────────────────────────────

SR    = 44100

# Adaptive N_FFT: target 20 bins per harmonic (same as extract_params.py).
# Clamped [2^13=8192, 2^15=32768] — ensures enough frequency resolution for
# bass notes while keeping computation tractable and hop reasonable.
NFFT_BINS_TARGET = 20
NFFT_EXP_MIN     = 13   # 8192  — floor (high notes, keeps behaviour unchanged)
NFFT_EXP_MAX     = 15   # 32768 — ceiling (A0: 1.35 Hz/bin, 2.4 bins per 1/6-oct)

# EQ frequency grid
N_EQ_POINTS = 64
EQ_F_MIN    = 20.0
EQ_F_MAX    = 20000.0

# 1/6-octave smoothing: half-width in octaves
SMOOTH_OCT = 1.0 / 12.0  # 1/12 oct each side = 1/6 oct window


def _adaptive_nfft(midi: int) -> tuple[int, int]:
    """Return (N_FFT, HOP) for the given MIDI note using adaptive window sizing."""
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    raw = int(NFFT_BINS_TARGET * SR / f0)
    exp = max(NFFT_EXP_MIN, min(NFFT_EXP_MAX, round(math.log2(raw))))
    nfft = 1 << exp
    return nfft, nfft // 4


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
            return key, {'_log': f"SKIP: WAV not found at {wav_path}", '_skip': True}

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
        orig_mono        = orig_mono[:n]
        synth_mono       = synth_mono[:n]
        orig_stereo_trim = orig_stereo[:n].astype(np.float64)
        synth_stereo_trim= synth_stereo[:n].astype(np.float64)

        # ── Stereo width factor: rms(Side)/rms(Mid) ratio, skip first 100 ms ──
        # Skip attack transient; measure steady-state stereo spread.
        skip = int(0.10 * SR)
        orig_M = (orig_stereo_trim[skip:, 0] + orig_stereo_trim[skip:, 1]) / 2
        orig_S = (orig_stereo_trim[skip:, 0] - orig_stereo_trim[skip:, 1]) / 2
        syn_M  = (synth_stereo_trim[skip:, 0] + synth_stereo_trim[skip:, 1]) / 2
        syn_S  = (synth_stereo_trim[skip:, 0] - synth_stereo_trim[skip:, 1]) / 2
        rms = lambda x: float(np.sqrt(np.mean(x**2)) + 1e-12)
        SM_orig  = rms(orig_S) / rms(orig_M)
        SM_synth = rms(syn_S) / rms(syn_M)
        width_factor = float(np.clip(SM_orig / SM_synth, 0.2, 8.0))

        # ── Adaptive N_FFT for this MIDI note ─────────────────────────────
        n_fft, hop = _adaptive_nfft(midi)

        # ── LTASE via STFT ─────────────────────────────────────────────────
        ltase_orig  = _compute_ltase(orig_mono,  n_fft, hop)
        ltase_synth = _compute_ltase(synth_mono, n_fft, hop)

        # Frequency axis for n_fft/2+1 bins
        freqs_fft = np.linspace(0.0, SR / 2.0, n_fft // 2 + 1)

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
        log_msg   = f"N_FFT={n_fft}  EQ peak={max_gain:.1f}dB range={range_db:.1f}dB  width={width_factor:.3f}"

        return key, {
            'freqs_hz': eq_freqs.tolist(),
            'gains_db': eq_gains.tolist(),
            'stereo_width_factor': width_factor,
            '_log': log_msg,
        }

    except Exception:
        return key, {'_log': traceback.format_exc(), '_error': True}


# ── STFT helpers ──────────────────────────────────────────────────────────────

def _compute_ltase(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """
    Long-Time Average Spectral Envelope via STFT.
    Returns 1-D array of shape (n_fft//2 + 1,): mean magnitude over time.
    """
    window = np.hanning(n_fft)
    n_bins = n_fft // 2 + 1
    frames = []
    for start in range(0, len(audio) - n_fft + 1, hop):
        frame = audio[start: start + n_fft] * window
        spec  = np.fft.rfft(frame, n=n_fft)
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
    _setup_log()
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
    total = len(keys)
    done = 0

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(data, bank_dir),
    ) as pool:
        for key, eq in pool.imap_unordered(process_sample, keys):
            done += 1
            log_msg = eq.pop('_log', '') if eq else ''
            is_skip = eq.pop('_skip', False) if eq else False
            is_err  = eq.pop('_error', False) if eq else False
            if is_err:
                print(f"  {done}/{total}: {key} ... ERROR:\n{log_msg}")
            elif is_skip or eq is None:
                print(f"  {done}/{total}: {key} ... {log_msg}")
            else:
                results[key] = eq
                print(f"  {done}/{total}: {key} ... {log_msg}")

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
