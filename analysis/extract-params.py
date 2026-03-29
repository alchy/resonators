"""
analysis/extract-params.py
──────────────────────────
Phase 0: Analytical extraction of physical parameters from a WAV sample bank.

Extracts per-note/velocity:
  - Inharmonicity coefficient B  (f_k = k·f0·√(1+B·k²))
  - Per-partial bi-exponential decay τ1_k, τ2_k
  - Per-partial beating frequency Δf_k and depth m_k
  - Noise model: attack burst τ, sustained floor RMS, spectral centroid
  - Initial amplitude A0 per partial

Output:  analysis/params-{bank}.json   (e.g. params-ks-grand.json)
Log:     runtime-logs/extract-params-log.txt  (auto-created, tee of stdout)

Usage:
    python -u analysis/extract-params.py \\
        --bank    C:/SoundBanks/IthacaPlayer/ks-grand \\
        --out     analysis/params-ks-grand.json \\
        --workers 4 \\
        [--verbose] [--plot] [--midi 60] [--vel 3]

Arguments:
  --bank      WAV sample bank directory
  --out       Output JSON path (default: analysis/params.json)
  --workers   Parallel worker processes (default: cpu_count)
  --verbose   Detailed per-file output
  --plot      Show diagnostic plots for individual notes
  --midi      Process only this MIDI note (debug)
  --vel       Process only this velocity layer (debug)
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count


# ── Runtime logging (tee stdout → runtime-logs/extract-params-log.txt) ───────

def _setup_log() -> None:
    log_dir = Path("runtime-logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "extract-params-log.txt"

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()

    sys.stdout = _Tee(sys.__stdout__, open(log_path, "w", encoding="utf-8", buffering=1))

import numpy as np
import soundfile as sf
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import welch

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── MIDI note → Hz (equal temperament) ──────────────────────────────────────

def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


# ── Audio loading ────────────────────────────────────────────────────────────

def load_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    return audio, sr


# ── High-resolution spectrum ─────────────────────────────────────────────────

def compute_spectrum(audio: np.ndarray, sr: int,
                     t_start: float = 0.1, t_end: float = 2.0,
                     zero_pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT magnitude spectrum with zero-padding on a steady-state window.
    Uses full available audio up to t_end.
    """
    i0 = int(t_start * sr)
    i1 = min(int(t_end * sr), len(audio))
    if i1 <= i0 + 512:
        i0 = 0
        i1 = len(audio)
    segment = audio[i0:i1]
    segment = segment * np.hanning(len(segment))
    n_fft = len(segment) * zero_pad
    spec = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freqs, spec


# ── Peak detection ───────────────────────────────────────────────────────────

def find_peaks_near(freqs: np.ndarray, spec: np.ndarray,
                    f_center: float, width_frac: float = 0.025) -> tuple[float, float]:
    """Find the dominant spectral peak near f_center. Returns (f_peak, amp)."""
    f_lo = f_center * (1 - width_frac)
    f_hi = f_center * (1 + width_frac)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return f_center, 0.0
    local_spec = spec[mask]
    local_freqs = freqs[mask]
    idx = local_spec.argmax()
    return float(local_freqs[idx]), float(local_spec[idx])


def detect_harmonic_peaks(freqs: np.ndarray, spec: np.ndarray,
                          f0_nominal: float, sr: int,
                          n_max: int = 90) -> tuple[list[dict], float, float]:
    """
    Detect harmonic peaks and fit f0 + B.

    Strategy:
    1. Estimate f0_true from k=1 peak (allows for tuning offset).
    2. Iteratively refine peak search using current f0+B estimate.
    3. Fit B using k>=2 peaks with f0 fixed.

    Returns (peaks_list, B, f0_true).
    """
    nyquist = sr / 2.0

    # Step 1: estimate f0 from k=1 (allow ±3% tuning offset)
    f0_k1, amp_k1 = find_peaks_near(freqs, spec, f0_nominal, width_frac=0.03)
    if amp_k1 < 1e-12:
        f0_k1 = f0_nominal
    f0_est = f0_k1

    # Step 2: rough B estimate from k=5 and k=10 if signal is there
    B_est = 0.0
    for k_probe in [5, 6, 7, 8]:
        f_exp = k_probe * f0_est
        if f_exp > nyquist * 0.9:
            break
        f_peak, amp = find_peaks_near(freqs, spec, f_exp, width_frac=0.04)
        if amp < 1e-12:
            continue
        # f_peak = k * f0 * sqrt(1 + B * k^2)
        ratio = f_peak / (k_probe * f0_est)
        if ratio > 1.0:
            B_est_local = (ratio ** 2 - 1.0) / (k_probe ** 2)
            B_est = max(B_est, B_est_local)

    # Step 3: collect all peaks using current f0+B estimate
    peaks = []
    for k in range(1, n_max + 1):
        f_inharmonic = k * f0_est * math.sqrt(1.0 + B_est * k * k)
        if f_inharmonic > nyquist * 0.97:
            break
        # Tighter search window for high k (partials closer together)
        width = max(0.008, 0.025 / math.sqrt(k))
        f_peak, amp = find_peaks_near(freqs, spec, f_inharmonic, width_frac=width)
        if amp < 1e-12:
            continue
        peaks.append({'k': k, 'f_measured': f_peak, 'amp': amp})

    if len(peaks) < 3:
        return peaks, B_est, f0_est

    # Step 4: fit B precisely with f0 fixed (least squares on log-freq)
    # f_k = k * f0 * sqrt(1 + B * k^2)  →  log(f_k/k) = log(f0) + 0.5*log(1+B*k^2)
    B, f0_fit = _fit_B_f0(peaks, f0_est)

    return peaks, B, f0_fit


def _fit_B_f0(peaks: list[dict], f0_nominal: float) -> tuple[float, float]:
    """Fit B and f0 from observed harmonic peaks using weighted least squares."""
    ks = np.array([p['k'] for p in peaks], dtype=float)
    fs = np.array([p['f_measured'] for p in peaks], dtype=float)
    ws = np.array([p['amp'] for p in peaks], dtype=float)
    ws = ws / (ws.sum() + 1e-12)

    def model(k, f0, B):
        B = max(B, 0.0)
        return k * f0 * np.sqrt(1.0 + B * k ** 2)

    try:
        p0 = [f0_nominal, 1e-4]
        bounds = ([f0_nominal * 0.97, 0.0], [f0_nominal * 1.03, 5e-3])
        popt, _ = curve_fit(model, ks, fs, p0=p0, bounds=bounds,
                            sigma=1.0 / (ws + 1e-9), absolute_sigma=False,
                            maxfev=8000)
        return float(max(popt[1], 0.0)), float(popt[0])
    except Exception:
        # Fallback: fix f0, just fit B
        try:
            def model_B(k, B):
                B = max(B, 0.0)
                return k * f0_nominal * np.sqrt(1.0 + B * k ** 2)
            popt, _ = curve_fit(model_B, ks, fs, p0=[1e-4],
                                bounds=([0.0], [5e-3]), maxfev=4000)
            return float(max(popt[0], 0.0)), f0_nominal
        except Exception:
            return 0.0, f0_nominal


def inharmonic_freq(k: int, f0: float, B: float) -> float:
    return k * f0 * math.sqrt(1.0 + B * k * k)


# ── Per-partial amplitude envelope ───────────────────────────────────────────

def compute_stft(audio: np.ndarray, sr: int,
                 hop: int = 1024, frame: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full STFT magnitude matrix once for the entire audio.
    Returns (times, freqs, mag_matrix)  where mag_matrix is [n_frames × n_bins].
    Caller reuses this for all partials (avoids repeated FFT computation).
    """
    window = np.hanning(frame)
    n_frames = (len(audio) - frame) // hop
    if n_frames < 8:
        return np.array([]), np.array([]), np.zeros((0, 0))

    mag = np.zeros((n_frames, frame // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        seg = audio[i * hop: i * hop + frame] * window
        mag[i] = np.abs(np.fft.rfft(seg))

    times = np.array([(i * hop + frame // 2) / sr for i in range(n_frames)])
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    return times, freqs, mag


def partial_envelope_from_stft(times: np.ndarray, freqs: np.ndarray, mag: np.ndarray,
                                f_center: float, search_bins: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract amplitude envelope of one partial from a precomputed STFT.
    Returns (times, amplitudes).
    """
    if mag.shape[0] < 8:
        return np.array([]), np.array([])

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    target_bin = int(round(f_center / freq_res))
    lo = max(0, target_bin - search_bins)
    hi = min(mag.shape[1] - 1, target_bin + search_bins)

    amps = mag[:, lo: hi + 1].max(axis=1).astype(np.float64)

    # Gate out frames below 0.1% of peak (silence)
    threshold = amps.max() * 0.001
    if amps.max() < 1e-12:
        return np.array([]), np.array([])

    return times, amps


def find_peak_frame(amps: np.ndarray) -> int:
    """Return index of amplitude peak (attack end)."""
    if len(amps) == 0:
        return 0
    # Smooth a bit first to avoid noise spike
    kernel = np.ones(3) / 3
    smoothed = np.convolve(amps, kernel, mode='same')
    return int(smoothed.argmax())


# ── Decay fitting ─────────────────────────────────────────────────────────────

def fit_decay(times: np.ndarray, amps: np.ndarray, i_peak: int) -> dict:
    """
    Fit bi-exponential decay: A(t) = A0 * [a1*exp(-t/τ1) + (1-a1)*exp(-t/τ2)]
    where τ1 < τ2 (fast component from hammer/early decay, slow from string).

    Also tries single exponential. Returns best fit.
    """
    default = {'tau1': None, 'tau2': None, 'a1': 1.0, 'A0': float(amps[i_peak]) if len(amps) > i_peak else 0.0, 'mono': True}

    if i_peak >= len(times):
        return default

    t = times[i_peak:] - times[i_peak]
    a = amps[i_peak:]
    if len(t) < 12 or a[0] < 1e-12:
        return default

    A0 = float(a[0])
    a_norm = a / A0

    # Clip extreme values
    a_norm = np.clip(a_norm, 1e-8, 1.2)

    result = dict(default)
    result['A0'] = A0

    # Single exponential
    try:
        popt, _ = curve_fit(lambda t, tau: np.exp(-t / tau),
                            t, a_norm, p0=[3.0],
                            bounds=([0.05], [120.0]), maxfev=3000)
        result['tau1'] = float(popt[0])
        result['mono'] = True
    except Exception:
        result['tau1'] = float(t[-1] / 3.0) if t[-1] > 0 else 3.0

    # Bi-exponential (only meaningful if note is long enough and SNR is ok)
    # tau2 max scales with signal duration to avoid fitting noise floor
    tau2_max = min(60.0, t[-1] * 0.9)
    if t[-1] > 1.0 and tau2_max > 0.15 and A0 > 1e-6:
        try:
            def bi_exp(t, a1, tau1, tau2):
                a1 = np.clip(a1, 0.01, 0.99)
                tau1 = max(tau1, 0.01)
                tau2 = max(tau2, tau1 + 0.05)
                return a1 * np.exp(-t / tau1) + (1 - a1) * np.exp(-t / tau2)

            tau1_init = min(result['tau1'] * 0.2, 2.0)
            tau2_init = result['tau1']
            popt2, _ = curve_fit(bi_exp, t, a_norm,
                                 p0=[0.2, tau1_init, tau2_init],
                                 bounds=([0.01, 0.02, 0.1], [0.99, 5.0, tau2_max]),
                                 maxfev=8000)
            a1, tau1, tau2 = popt2
            # Accept bi-exp only if:
            # - components are clearly separated (ratio > 3x)
            # - both components have meaningful weight
            # - tau2 < 90% of max bound (not hitting bound)
            if (tau2 / tau1 > 3.0 and 0.05 < a1 < 0.95
                    and tau2 < tau2_max * 0.9):
                result['tau1'] = float(tau1)
                result['tau2'] = float(tau2)
                result['a1'] = float(a1)
                result['mono'] = False
        except Exception:
            pass

    return result


# ── Beating detection ─────────────────────────────────────────────────────────

def detect_beating(times: np.ndarray, amps: np.ndarray, i_peak: int) -> dict:
    """
    Detect amplitude modulation (beating) from string coupling.
    Returns {'beat_hz': float, 'beat_depth': float}.

    Method: fit exponential trend, analyze residual oscillation.
    """
    result = {'beat_hz': 0.0, 'beat_depth': 0.0}

    t = times[i_peak:]
    a = amps[i_peak:]
    if len(a) < 48:
        return result

    t_rel = t - t[0]
    a_log = np.log(a + 1e-15)

    # Remove linear trend in log domain (exponential decay)
    if len(t_rel) >= 4:
        coeffs = np.polyfit(t_rel, a_log, 1)
        trend = np.polyval(coeffs, t_rel)
        residual = a_log - trend
    else:
        return result

    # Mean-center
    residual -= residual.mean()

    # Estimate envelope sample rate
    dt = float(np.mean(np.diff(t_rel))) if len(t_rel) > 1 else 0.02
    sr_env = 1.0 / dt  # Hz

    # FFT of residual
    n = len(residual)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(residual * win))
    freqs_env = np.fft.rfftfreq(n, dt)

    # Search 0.1–10 Hz (beating range for piano)
    mask = (freqs_env >= 0.1) & (freqs_env <= 10.0)
    if not mask.any():
        return result

    local_spec = spec[mask]
    local_freqs = freqs_env[mask]

    # Find dominant beating frequency
    idx_max = local_spec.argmax()
    beat_hz = float(local_freqs[idx_max])

    # SNR: peak vs median of local spectrum
    noise_level = np.median(local_spec) + 1e-15
    snr = local_spec[idx_max] / noise_level

    # Depth: peak amplitude in physical domain (not log)
    # residual is log-amplitude, so beat_depth ≈ 2 * amplitude of oscillation
    beat_amplitude = local_spec[idx_max] * 2 / n
    beat_depth = float(np.clip(2 * beat_amplitude, 0.0, 1.0))  # approx modulation depth

    # Only report if SNR > 3 and beat_depth > 2%
    if snr > 3.0 and beat_depth > 0.02:
        result['beat_hz'] = beat_hz
        result['beat_depth'] = beat_depth

    return result


# ── Noise analysis ─────────────────────────────────────────────────────────────

def analyze_noise(audio: np.ndarray, sr: int,
                  peaks: list[dict], f0: float, B: float) -> dict:
    """
    Estimate noise model by analyzing early attack residual after harmonic subtraction.
    Returns {'attack_tau_s', 'floor_rms', 'centroid_hz', 'spectral_slope_db_oct'}.
    """
    result = {'attack_tau_s': 0.05, 'floor_rms': 0.001,
              'centroid_hz': 2000.0, 'spectral_slope_db_oct': -3.0}

    if len(audio) < sr * 0.1:
        return result

    # Attack window: first 200ms
    attack = audio[:int(0.2 * sr)].copy()

    # Subtract top harmonics from attack
    n = len(attack)
    t_vec = np.arange(n) / sr

    # Build harmonic signal from peak parameters
    harmonic = np.zeros(n, dtype=np.float32)
    A_scale = max(np.abs(attack).max(), 1e-10)
    for p in peaks[:30]:  # top 30 partials
        fk = inharmonic_freq(p['k'], f0, B)
        if fk < sr / 2 * 0.95:
            # Amplitude from spectrum is not calibrated to waveform; use scaled version
            harmonic += np.cos(2 * np.pi * fk * t_vec).astype(np.float32)

    # Normalize harmonic and subtract
    h_scale = np.std(harmonic) + 1e-10
    harmonic = harmonic / h_scale * np.std(attack) * 0.8
    residual = attack - harmonic

    # Attack noise decay (RMS envelope of residual)
    hop = 256
    frame = 1024
    n_frames = max(1, (len(residual) - frame) // hop)
    rms_env = np.array([
        math.sqrt(max(np.mean(residual[i * hop: i * hop + frame] ** 2), 1e-30))
        for i in range(n_frames)
    ])
    t_rms = np.array([(i * hop + frame // 2) / sr for i in range(n_frames)])

    if len(rms_env) >= 6:
        i_peak = rms_env.argmax()
        t_dec = t_rms[i_peak:] - t_rms[i_peak]
        a_dec = rms_env[i_peak:]
        if len(t_dec) >= 4 and a_dec[0] > 1e-10:
            try:
                popt, _ = curve_fit(
                    lambda t, tau: a_dec[0] * np.exp(-t / tau),
                    t_dec, a_dec, p0=[0.05],
                    bounds=([0.003], [1.0]), maxfev=2000
                )
                result['attack_tau_s'] = float(popt[0])
            except Exception:
                pass
        result['floor_rms'] = float(rms_env[-1])

    # Spectral centroid and slope of noise (from later in the note to avoid hammer)
    i_late = min(int(0.1 * sr), len(audio) - frame)
    if i_late >= 0:
        noise_segment = audio[i_late: i_late + frame]
        try:
            f_w, psd = welch(noise_segment, fs=sr, nperseg=frame // 2)
            if psd.sum() > 0:
                # Centroid
                centroid = float(np.sum(f_w * psd) / psd.sum())
                result['centroid_hz'] = centroid

                # Spectral slope (linear regression on log-log scale, 200Hz–8kHz)
                mask = (f_w >= 200) & (f_w <= 8000)
                if mask.sum() >= 4:
                    log_f = np.log2(f_w[mask] + 1)
                    log_p = np.log10(psd[mask] + 1e-30)
                    coeffs = np.polyfit(log_f, log_p, 1)
                    # slope in dB/octave ≈ 10 * polyfit_slope
                    result['spectral_slope_db_oct'] = float(coeffs[0] * 10)
        except Exception:
            pass

    return result


# ── Per-file analysis ─────────────────────────────────────────────────────────

def analyze_file(path: str, midi: int, vel: int, verbose: bool = False) -> dict:
    """Full physical parameter extraction for one sample file."""
    audio, sr = load_mono(path)
    duration = len(audio) / sr
    f0_nominal = midi_to_hz(midi)

    result = {
        'midi': midi, 'vel': vel,
        'f0_nominal_hz': float(f0_nominal),
        'sr': sr,
        'duration_s': float(duration),
        'B': 0.0,
        'f0_fitted_hz': float(f0_nominal),
        'n_partials': 0,
        'partials': [],
        'noise': {},
    }

    if duration < 0.1:
        return result

    # Spectrum: use 0.1s to 3s (steady state, long enough for frequency resolution)
    t_spec_start = min(0.1, duration * 0.05)
    t_spec_end = min(t_spec_start + 4.0, duration * 0.95)

    freqs, spec = compute_spectrum(audio, sr, t_spec_start, t_spec_end, zero_pad=8)

    # Detect peaks and fit B
    peaks, B, f0_fit = detect_harmonic_peaks(freqs, spec, f0_nominal, sr)

    result['B'] = float(B)
    result['f0_fitted_hz'] = float(f0_fit)
    result['n_partials'] = len(peaks)

    if verbose:
        print(f"    f0_nominal={f0_nominal:.2f}Hz  f0_fit={f0_fit:.3f}Hz  "
              f"B={B:.6f}  peaks={len(peaks)}")

    if not peaks:
        return result

    # Frequency-adaptive STFT: frame scales to ~20 bins per harmonic spacing (f0).
    # Bass notes need larger frame for frequency resolution (prevent k=1/k=2 bleed).
    # Treble notes benefit from smaller frame (better time resolution for fast decays).
    # Frame is rounded to next power of 2, clamped to [2048, 32768].
    TARGET_BINS_PER_HARMONIC = 20
    raw_frame = int(TARGET_BINS_PER_HARMONIC * sr / f0_nominal)
    frame_exp = max(11, min(15, round(math.log2(raw_frame))))  # 2048..32768
    frame_env = 1 << frame_exp
    hop_env   = frame_env // 4
    stft_times, stft_freqs, stft_mag = compute_stft(audio, sr, hop=hop_env, frame=frame_env)

    partials_out = []
    for p in peaks:
        k = p['k']
        fk = inharmonic_freq(k, f0_fit, B)

        if stft_mag.shape[0] >= 8:
            t_env, a_env = partial_envelope_from_stft(stft_times, stft_freqs, stft_mag, fk)
        else:
            t_env, a_env = np.array([]), np.array([])

        if len(t_env) < 12:
            partials_out.append({
                'k': k, 'f_hz': float(fk),
                'A0': float(p['amp']),
                'tau1': None, 'tau2': None, 'a1': 1.0, 'mono': True,
                'beat_hz': 0.0, 'beat_depth': 0.0,
            })
            continue

        i_peak = find_peak_frame(a_env)
        decay = fit_decay(t_env, a_env, i_peak)
        beat = detect_beating(t_env, a_env, i_peak)

        partials_out.append({
            'k': k,
            'f_hz': float(fk),
            'A0': float(decay['A0']),
            'tau1': decay['tau1'],
            'tau2': decay['tau2'],
            'a1': float(decay['a1']),
            'mono': decay['mono'],
            'beat_hz': beat['beat_hz'],
            'beat_depth': beat['beat_depth'],
        })

    result['partials'] = partials_out
    result['noise'] = analyze_noise(audio, sr, peaks, f0_fit, B)

    return result


# ── Bank-level analysis ───────────────────────────────────────────────────────

def _analyze_file_worker(args):
    """Worker function for multiprocessing."""
    path, midi, vel = args
    try:
        data = analyze_file(path, midi, vel, verbose=False)
        return (f"m{midi:03d}_vel{vel}", data, None)
    except Exception as e:
        return (f"m{midi:03d}_vel{vel}", None, str(e))


def analyze_bank(bank_dir: str, out_path: str,
                 verbose: bool = False,
                 midi_filter: int = None,
                 vel_filter: int = None,
                 n_workers: int = None):
    bank_dir = Path(bank_dir)
    wav_files = sorted(bank_dir.glob("m*-vel*-f*.wav"))

    if not wav_files:
        print(f"No WAV files found in {bank_dir}")
        return None

    # Build work list
    work = []
    for wav_path in wav_files:
        name = wav_path.stem
        parts = name.split('-')
        try:
            midi = int(parts[0][1:])
            vel  = int(parts[1][3:])
        except (ValueError, IndexError):
            continue
        if midi_filter is not None and midi != midi_filter:
            continue
        if vel_filter is not None and vel != vel_filter:
            continue
        work.append((str(wav_path), midi, vel))

    n_workers = n_workers or max(1, cpu_count() - 1)
    print(f"Found {len(wav_files)} samples, processing {len(work)} files "
          f"with {n_workers} workers. Output: {out_path}")

    results = {}
    errors = []
    total = len(work)
    done = 0

    if n_workers == 1 or len(work) == 1:
        # Single-process (easier debugging)
        for path, midi, vel in work:
            key, data, err = _analyze_file_worker((path, midi, vel))
            done += 1
            name = Path(path).name
            if err:
                print(f"  {done}/{total}: {name} ... ERROR: {err}")
                errors.append((name, err))
            else:
                results[key] = data
                print(f"  {done}/{total}: {name} ... B={data['B']:.5f}  partials={data['n_partials']}  dur={data['duration_s']:.1f}s")
    else:
        with Pool(n_workers) as pool:
            for key, data, err in pool.imap_unordered(_analyze_file_worker, work):
                done += 1
                if err:
                    print(f"  {done}/{total}: {key} ... ERROR: {err}")
                    errors.append((key, err))
                else:
                    results[key] = data
                    print(f"  {done}/{total}: {key} ... B={data['B']:.5f}  p={data['n_partials']}  dur={data['duration_s']:.1f}s")

    summary = _compute_summary(results)

    output = {
        'bank_dir': str(bank_dir),
        'n_samples': len(results),
        'summary': summary,
        'samples': results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output = _sanitize_for_json(output)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, allow_nan=False)

    print(f"\nDone. {len(results)} samples, {len(errors)} errors -> {out_path}")
    if errors:
        for name, e in errors:
            print(f"  ERROR {name}: {e}")

    return output


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _compute_summary(results: dict) -> dict:
    """Keyboard-wide statistics and B(midi) regression."""
    # B per MIDI note (average over velocities)
    B_by_midi = {}
    f0_by_midi = {}
    for key, data in results.items():
        m = data['midi']
        if data['B'] > 0:
            B_by_midi.setdefault(m, []).append(data['B'])
        f0_by_midi.setdefault(m, []).append(data['f0_fitted_hz'])

    B_mean = {m: float(np.mean(vs)) for m, vs in B_by_midi.items()}
    f0_mean = {m: float(np.mean(vs)) for m, vs in f0_by_midi.items()}

    # Log-linear B vs MIDI fit: log(B) = a*midi + b
    B_fit = {'slope': 0.0, 'intercept': math.log(1e-4)}
    if len(B_mean) >= 4:
        midis = np.array(sorted(B_mean.keys()))
        Bs = np.array([B_mean[m] for m in midis])
        valid = Bs > 0
        if valid.sum() >= 4:
            try:
                coeffs = np.polyfit(midis[valid], np.log(Bs[valid]), 1)
                B_fit = {'slope': float(coeffs[0]), 'intercept': float(coeffs[1])}
            except Exception:
                pass

    # Tuning offsets (cents relative to equal temperament)
    tuning_cents = {}
    for m, f0_vals in f0_by_midi.items():
        f0_et = midi_to_hz(m)
        f0_measured = float(np.mean(f0_vals))
        cents = 1200 * math.log2(f0_measured / f0_et) if f0_measured > 0 and f0_et > 0 else 0.0
        tuning_cents[m] = round(cents, 2)

    return {
        'n_midi_notes': len(set(d['midi'] for d in results.values())),
        'n_velocities': len(set(d['vel'] for d in results.values())),
        'B_by_midi': {str(k): v for k, v in B_mean.items()},
        'B_log_linear_fit': B_fit,
        'f0_fitted_hz_by_midi': {str(k): v for k, v in f0_mean.items()},
        'tuning_cents_by_midi': {str(k): v for k, v in tuning_cents.items()},
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_sample(params_path: str, midi: int, vel: int):
    import matplotlib.pyplot as plt

    with open(params_path) as f:
        data = json.load(f)

    key = f"m{midi:03d}_vel{vel}"
    if key not in data['samples']:
        print(f"Key {key} not found")
        return

    sample = data['samples'][key]
    ps = sample['partials']
    if not ps:
        print("No partials")
        return

    ks        = [p['k'] for p in ps]
    A0s       = [p['A0'] for p in ps]
    tau1s     = [p['tau1'] or 0 for p in ps]
    tau2s     = [p['tau2'] or 0 for p in ps]
    beat_hzs  = [p['beat_hz'] for p in ps]
    beat_deps = [p['beat_depth'] for p in ps]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = (f"MIDI={midi} vel={vel}  "
             f"B={sample['B']:.5f}  f0={sample['f0_fitted_hz']:.3f}Hz  "
             f"dur={sample['duration_s']:.1f}s")
    fig.suptitle(title, fontsize=11)

    axes[0, 0].bar(ks, 20 * np.log10(np.array(A0s) + 1e-12))
    axes[0, 0].set(xlabel='Partial k', ylabel='Amplitude (dB)', title='Partial amplitudes')

    axes[0, 1].semilogy(ks, [max(t, 1e-3) for t in tau1s], 'o-', label='τ1')
    axes[0, 1].semilogy(ks, [max(t, 1e-3) for t in tau2s], 's--', label='τ2 (bi-exp)')
    axes[0, 1].set(xlabel='Partial k', ylabel='τ (s)', title='Decay times per partial')
    axes[0, 1].legend()

    axes[1, 0].stem(ks, beat_hzs)
    axes[1, 0].set(xlabel='Partial k', ylabel='Beat freq (Hz)', title='Beating frequency')

    axes[1, 1].stem(ks, beat_deps)
    axes[1, 1].set(xlabel='Partial k', ylabel='Beat depth', title='Beating depth')

    plt.tight_layout()
    out = f'analysis/plot_m{midi:03d}_vel{vel}.png'
    Path('analysis').mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


def plot_keyboard_B(params_path: str):
    """Plot B coefficient across the keyboard."""
    import matplotlib.pyplot as plt

    with open(params_path) as f:
        data = json.load(f)

    summary = data['summary']
    B_by_midi = {int(k): v for k, v in summary['B_by_midi'].items()}
    midis = sorted(B_by_midi.keys())
    Bs = [B_by_midi[m] for m in midis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(midis, Bs, 'o-')
    # Overlay fit
    fit = summary['B_log_linear_fit']
    B_pred = [math.exp(fit['slope'] * m + fit['intercept']) for m in midis]
    axes[0].semilogy(midis, B_pred, 'r--', label='log-linear fit')
    axes[0].set(xlabel='MIDI note', ylabel='B (inharmonicity)', title='B across keyboard')
    axes[0].legend()

    # Tuning
    tuning = {int(k): v for k, v in summary['tuning_cents_by_midi'].items()}
    t_midis = sorted(tuning.keys())
    t_cents = [tuning[m] for m in t_midis]
    axes[1].plot(t_midis, t_cents, 'o-')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set(xlabel='MIDI note', ylabel='Cents from equal temperament',
                title='Tuning deviation across keyboard')

    plt.tight_layout()
    out = 'analysis/plot_keyboard_B.png'
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    _setup_log()
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='C:/SoundBanks/IthacaPlayer/ks-grand')
    parser.add_argument('--out', default='IthacaCoreResonator/soundbanks/salamander.json')
    parser.add_argument('--plot', action='store_true', help='Plot one sample')
    parser.add_argument('--plot-keyboard', action='store_true', help='Plot B across keyboard')
    parser.add_argument('--midi', type=int, default=None)
    parser.add_argument('--vel', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count-1, 1=serial)')
    args = parser.parse_args()

    if args.plot:
        plot_sample(args.out, args.midi or 60, args.vel or 3)
    elif args.plot_keyboard:
        plot_keyboard_B(args.out)
    else:
        analyze_bank(args.bank, args.out,
                     verbose=args.verbose,
                     midi_filter=args.midi,
                     vel_filter=args.vel,
                     n_workers=args.workers)


if __name__ == '__main__':
    main()
