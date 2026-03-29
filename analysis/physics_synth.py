"""
analysis/physics_synth.py
─────────────────────────
Phase 1+: Pure physical synthesizer with improved beating and soundboard.

Physical model:
  Each harmonic k has N_strings independent oscillators (2 or 3 based on MIDI range):
    osc_i(t) = cos(2pi*f_k_i*t + phi_i)
    where f_k_i = f_k + delta_i  (delta_i distributed around 0, sum=0)

  Beating emerges naturally from the sum of independent oscillators.
  This gives full-depth amplitude modulation (goes from 0 to N_strings * A0)
  rather than the limited (1 + depth * cos) approximation.

  beat_scale: multiplier on extracted beat_hz — increase to make beating more
  audible (default 1.0, try 1.5-2.5 for more prominent string interference).

  Soundboard: PARKED — parametric convolution [0..1] disabled by default.
    The synthetic modal IR narrows frequency response (band-pass distortion) and
    creates amplitude modulation ("croaking") rather than adding body/warmth.
    Needs redesign from measured IR or different approach.
    0.0 = bypass (default, hardware may have real soundboard)
    1.0 = full virtual soundboard (when functional)

Usage (via CLI wrapper):
    python analysis/physics-synth.py --params analysis/params-ks-grand.json \
                                     --midi 60 --vel 3 --duration 6 \
                                     [--soundboard 0.0] [--beat-scale 1.5] \
                                     [--compare] [--preview]

Note: import as module via  from analysis.physics_synth import synthesize_note
      CLI entry point: analysis/physics-synth.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter


def load_synth_config(path: str = 'analysis/synth_config.json') -> dict:
    """Load synthesis config. Returns empty dict if file not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def synth_config_to_kwargs(config: dict) -> dict:
    """Extract synthesize_note() kwargs from synth_config.json structure."""
    kwargs = {}
    r = config.get('render', {})
    t = config.get('timbre', {})
    s = config.get('stereo', {})
    for key, section in [
        ('sr', r), ('duration', r), ('fade_out', r), ('target_rms', r),
        ('onset_ms', r),
        ('harmonic_brightness', t), ('beat_scale', t),
        ('eq_strength', t), ('soundboard_strength', t), ('noise_level', t),
        ('pan_spread', s), ('stereo_boost', s), ('stereo_decorr', s),
    ]:
        if key in section and not key.startswith('_'):
            kwargs[key] = section[key]
    return kwargs


# ── String count per MIDI note ────────────────────────────────────────────────

def n_strings_for_midi(midi: int) -> int:
    """
    Number of strings per note. Approximate typical grand piano stringing.
    Bass: 1 string (monochord) for lowest ~3 notes
    Low-mid: 2 strings (bichord)
    Upper: 3 strings (trichord)
    """
    if midi <= 27:      # A0-Eb1: 1 string (wound bass)
        return 1
    elif midi <= 48:    # E1-C3: 2 strings (wound)
        return 2
    else:               # C#3 upwards: 3 strings
        return 3


# ── Soundboard IR synthesis ───────────────────────────────────────────────────

def make_soundboard_ir(sr: int, duration: float = 0.3,
                       n_modes: int = 40, seed: int = 42) -> np.ndarray:
    """
    PARKED: Soundboard IR synthesis needs redesign.

    Current synthetic modal IR is too aggressive -- it narrows frequency response
    (band-pass distortion), modulates amplitude ("croaking" effect), and reduces
    rather than expands perceived body. Real soundboard behavior: adds diffuse
    reverb tail, slightly brightens transient, does NOT narrow band.

    This function is kept for reference. Use soundboard_strength=0.0 (default)
    until a better model is implemented.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr
    ir = np.zeros(n, dtype=np.float64)

    freqs = np.concatenate([
        rng.uniform(50, 600, n_modes // 2),
        rng.uniform(600, 3000, n_modes // 4),
        rng.uniform(3000, 5000, n_modes // 4),
    ])
    freqs = freqs[:n_modes]

    T60 = np.clip(60 / (np.pi * freqs) * 80, 0.02, 0.5)
    amps = 1.0 / (freqs ** 0.5)
    amps = amps / amps.sum()
    phases = rng.uniform(0, 2 * np.pi, n_modes)

    for f, T, A, phi in zip(freqs, T60, amps, phases):
        tau = T / 2.303  # T60 -> 1/e time
        ir += A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi)

    peak = np.abs(ir).max()
    if peak > 1e-10:
        ir = ir / peak * 0.5

    return ir.astype(np.float32)


_SOUNDBOARD_IR_CACHE: dict[int, np.ndarray] = {}


def get_soundboard_ir(sr: int) -> np.ndarray:
    if sr not in _SOUNDBOARD_IR_CACHE:
        _SOUNDBOARD_IR_CACHE[sr] = make_soundboard_ir(sr)
    return _SOUNDBOARD_IR_CACHE[sr]


def apply_soundboard(audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """
    Apply parametric soundboard convolution. PARKED: strength=0.0 by default.
    """
    if strength < 0.005:
        return audio

    ir = get_soundboard_ir(sr)
    wet = fftconvolve(audio, ir, mode='same').astype(np.float64)
    return audio + strength * wet


# ── Panning helper ───────────────────────────────────────────────────────────

def _pan_gains(angle: float) -> tuple:
    """Constant-power stereo pan. angle=pi/4 is center (equal L and R)."""
    return math.cos(angle), math.sin(angle)


# ── Core synthesis ────────────────────────────────────────────────────────────

def apply_spectral_eq(audio: np.ndarray, eq_data: dict,
                      sr: int, strength: float = 1.0,
                      freq_min: float = 400.0) -> np.ndarray:
    """Apply per-note spectral EQ derived from original sample comparison.

    H(f) = LTASE_original / LTASE_synth captures the soundboard spectral
    shaping (body resonance boost, high-frequency rolloff) without IR
    convolution artifacts. Applied identically to L and R channels --
    spectral coloring is mono, stereo is preserved.

    Args:
        audio:    (N, 2) stereo float32
        eq_data:  dict with keys freqs_hz and gains_db (64 log-spaced points)
        sr:       sample rate
        strength: [0,1] blend (0=bypass, 1=full EQ)
        freq_min: EQ is flat (0 dB) below this frequency. Avoids room
                  acoustics contamination from LTASE at low frequencies
                  (default 400 Hz — below this the EQ distorts the fundamental).

    Returns: (N, 2) float32
    """
    if strength < 0.005 or not eq_data:
        return audio

    freqs_stored = np.array(eq_data.get("freqs_hz", []), dtype=np.float64)
    gains_db     = np.array(eq_data.get("gains_db", []), dtype=np.float64)
    if len(freqs_stored) == 0:
        return audio

    # Zero out EQ gains below freq_min (transition over one octave)
    # This prevents the EQ from cutting the fundamental/low harmonics
    # where the LTASE ratio is contaminated by room acoustics.
    if freq_min > 0:
        fade_low = freq_min / 2.0  # start of transition (-1 oct)
        for i, f in enumerate(freqs_stored):
            if f < fade_low:
                gains_db[i] = 0.0
            elif f < freq_min:
                t = (f - fade_low) / (freq_min - fade_low)  # 0..1
                gains_db[i] = gains_db[i] * t

    n = len(audio)
    # FFT on next power of 2 for efficiency
    n_fft = 1 << (n - 1).bit_length()
    freqs_fft = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Interpolate stored EQ to FFT grid (linear in log-freq space)
    gains_linear = 10.0 ** (gains_db / 20.0)
    H = np.interp(freqs_fft, freqs_stored, gains_linear,
                  left=gains_linear[0], right=gains_linear[-1])

    # Blend with flat (1.0) according to strength
    H_blend = 1.0 + strength * (H - 1.0)

    # Apply to each channel independently (same H -- spectral shape is mono)
    result = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        X = np.fft.rfft(audio[:, ch].astype(np.float64), n=n_fft)
        y = np.fft.irfft(X * H_blend, n=n_fft)
        result[:, ch] = y[:n].astype(np.float32)

    return result


def apply_stereo_width(audio: np.ndarray, width_factor: float, stereo_boost: float = 1.0) -> np.ndarray:
    """M/S stereo width scaling.

    width_factor: derived from original sample (rms(S_orig)/rms(M_orig)) / (rms(S_synth)/rms(M_synth))
    stereo_boost: additional multiplicative boost on top of width_factor (1.0 = no extra boost)
    Effective Side gain = width_factor * stereo_boost
    Mid channel is unchanged.
    """
    effective = float(np.clip(width_factor * stereo_boost, 0.0, 6.0))
    if abs(effective - 1.0) < 0.01:
        return audio
    L = audio[:, 0].astype(np.float64)
    R = audio[:, 1].astype(np.float64)
    M = (L + R) / 2.0
    S = (L - R) / 2.0
    S_scaled = S * effective
    result = np.empty_like(audio)
    result[:, 0] = (M + S_scaled).astype(np.float32)
    result[:, 1] = (M - S_scaled).astype(np.float32)
    return result


def _string_angles(midi: int, n_str: int, pan_spread: float) -> list:
    """Pan angle per string. Global: bass=left, treble=right (+/-0.20 rad)."""
    center = math.pi / 4 + (midi - 64.5) / 87.0 * 0.20
    if n_str == 1:
        return [center]
    half = pan_spread / 2
    if n_str == 2:
        return [center - half, center + half]
    return [center - half, center, center + half]


def synthesize_note(params: dict,
                    duration: float = None,
                    sr: int = 44100,
                    soundboard_strength: float = 0.0,
                    beat_scale: float = 1.0,
                    pan_spread: float = 0.55,
                    eq_strength: float = 1.0,
                    eq_freq_min: float = 400.0,
                    stereo_boost: float = 1.0,
                    harmonic_brightness: float = 0.0,
                    fade_out: float = 0.5,
                    target_rms: float = 0.06,
                    noise_level: float = 1.0,
                    stereo_decorr: float = 1.0,
                    onset_ms: float = 3.0,
                    rng_seed: int = None) -> np.ndarray:
    """Synthesize a piano note in stereo (N,2) via per-string panning.

    Each string has a distinct pan angle; beating creates different L/R
    interference patterns giving stereo width + different L/R envelope shapes,
    matching the character of two-microphone piano recordings.

    Returns: (N, 2) float32 stereo array.
    """
    rng = np.random.default_rng(rng_seed)
    if duration is None:
        duration = min(params.get('duration_s', 4.0), 8.0)
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr

    L = np.zeros(n, dtype=np.float64)
    R = np.zeros(n, dtype=np.float64)

    partials = params.get('partials', [])
    if not partials:
        return np.zeros((n, 2), dtype=np.float32)

    A0_ref = next((p['A0'] for p in partials if p['A0'] and p['A0'] > 1e-10), 1.0)
    midi   = params.get('midi', 60)
    n_str  = n_strings_for_midi(midi)
    angles = _string_angles(midi, n_str, pan_spread)

    for p in partials:
        f = p['f_hz']
        A = p['A0']
        if A is None or A < 1e-10 or f > sr * 0.495:
            continue
        k = p.get('k', 1) or 1
        # Harmonic brightness: boost amplitude of upper partials.
        # harmonic_brightness=0: no change (default).
        # harmonic_brightness=1: doubles amplitude at k=5, triples at k=9, etc.
        # Models the fact that the original samples have stronger attack transient
        # in upper harmonics than our extraction captures.
        if harmonic_brightness != 0.0 and k > 1:
            bright_gain = 1.0 + harmonic_brightness * math.log2(k)
        else:
            bright_gain = 1.0
        amp  = (A / A0_ref) * bright_gain
        tau1 = p.get('tau1') or 3.0
        tau2 = p.get('tau2')
        a1   = p.get('a1') or 1.0
        if tau2 is not None and not p.get('mono', True):
            env = a1 * np.exp(-t / tau1) + (1 - a1) * np.exp(-t / tau2)
        else:
            env = np.exp(-t / tau1)
        beat = (p.get('beat_hz', 0.0) or 0.0) * beat_scale

        if n_str == 1:
            # Monochord: always center, no beat
            phi = rng.uniform(0, 2 * math.pi)
            s = amp * env * np.cos(2 * math.pi * f * t + phi)
            gl, gr = _pan_gains(angles[0])
            L += s * gl;  R += s * gr
        elif beat < 0.05 and n_str > 1:
            # Multi-string but effectively no beating: use per-string panning
            # without frequency spread (same frequency, different phase + pan)
            phis = rng.uniform(0, 2 * math.pi, n_str)
            oscs = [np.cos(2 * math.pi * f * t + phis[i]) for i in range(n_str)]
            for i in range(n_str):
                gl, gr = _pan_gains(angles[i])
                s = amp * env * oscs[i] / n_str
                L += s * gl;  R += s * gr
        elif n_str == 2:
            pa, pb = rng.uniform(0, 2 * math.pi, 2)
            sa = amp * env * np.cos(2 * math.pi * (f + beat / 2) * t + pa)
            sb = amp * env * np.cos(2 * math.pi * (f - beat / 2) * t + pb)
            gla, gra = _pan_gains(angles[0]);  glb, grb = _pan_gains(angles[1])
            L += (sa * gla + sb * glb) * 0.5
            R += (sa * gra + sb * grb) * 0.5
        else:
            pa, pb, pc = rng.uniform(0, 2 * math.pi, 3)
            d = beat / 2
            sa = amp * env * np.cos(2 * math.pi * (f - d) * t + pa)
            sb = amp * env * np.cos(2 * math.pi * f * t + pb)
            sc = amp * env * np.cos(2 * math.pi * (f + d) * t + pc)
            gla, gra = _pan_gains(angles[0])
            glb, grb = _pan_gains(angles[1])
            glc, grc = _pan_gains(angles[2])
            L += (sa * gla + sb * glb + sc * glc) / 3.0
            R += (sa * gra + sb * grb + sc * grc) / 3.0

    # Attack noise: independent L/R, pure transient (no persistent floor)
    # tau_noise capped at k=1 string decay -- hammer always decays faster than string
    noise_p  = params.get('noise', {})
    taun_raw = noise_p.get('attack_tau_s', 0.05) or 0.05
    cent     = noise_p.get('centroid_hz', 3000.0) or 3000.0
    A_noise  = (noise_p.get('A_noise', 0.06) or 0.06) * noise_level
    # Find tau1 of k=1 partial to cap noise decay
    tau1_k1 = next((p.get('tau1', 3.0) for p in partials
                    if p.get('k') == 1 and p.get('A0') and p['A0'] > 1e-10), 3.0) or 3.0
    taun = min(taun_raw, tau1_k1)  # noise never outlasts the string
    alp  = 1.0 - math.exp(-2 * math.pi * min(cent, sr * 0.45) / sr)
    nenv = np.exp(-t / max(taun, 0.001))
    for buf in (L, R):
        raw = rng.standard_normal(n)
        sh = np.zeros(n);  y = 0.0
        for i in range(n):
            y = alp * raw[i] + (1 - alp) * y;  sh[i] = y
        buf += A_noise * sh * nenv

    if soundboard_strength > 0.005:
        ir = get_soundboard_ir(sr)
        L = L + soundboard_strength * fftconvolve(L, ir, mode='same')
        R = R + soundboard_strength * fftconvolve(R, ir, mode='same')

    if fade_out > 0 and duration > fade_out:
        nf = int(fade_out * sr);  fade = np.linspace(1.0, 0.0, nf)
        L[-nf:] *= fade;  R[-nf:] *= fade

    # Frequency-dependent stereo decorrelation (Schroeder all-pass pair)
    # Replicates the natural L/R differences from mic geometry and soundboard coupling.
    # Decorrelation strength scales with MIDI (treble needs more width).
    decor_strength = min(1.0, (midi - 40) / 60.0) * 0.45 * stereo_decorr
    if decor_strength > 0.01:
        # Simple first-order all-pass: y[n] = -g*x[n] + x[n-1] + g*y[n-1]
        # Applied with different coefficients to L and R -> decorrelates them
        g_L = 0.35 + decor_strength * 0.25  # ~0.35-0.46
        g_R = -(0.35 + decor_strength * 0.20)  # opposite sign = phase flip at Nyquist
        # All-pass for L: b=[−g, 1], a=[1, g]
        L_ap = lfilter([-g_L, 1.0], [1.0, g_L], L)
        R_ap = lfilter([-g_R, 1.0], [1.0, g_R], R)
        L = L * (1 - decor_strength) + L_ap * decor_strength
        R = R * (1 - decor_strength) + R_ap * decor_strength

    # RMS normalize -- avoids overboosting from peak normalization
    rms = math.sqrt((np.mean(L ** 2) + np.mean(R ** 2)) / 2)
    if rms > 1e-10:
        scale = min(target_rms / rms, 0.95 / max(np.abs(L).max(), np.abs(R).max()))
        L *= scale;  R *= scale

    stereo = np.stack([L, R], axis=1).astype(np.float32)

    # Spectral EQ -- "resonant cavity" layer: corrects spectral shape to
    # match original sample (body resonance, high-frequency rolloff).
    # Derived per-note from LTASE_original / LTASE_synth in compute_spectral_eq.py
    eq_data = params.get("spectral_eq")
    if eq_data and eq_strength > 0.005:
        stereo = apply_spectral_eq(stereo, eq_data, sr, strength=eq_strength,
                                   freq_min=eq_freq_min)
        # Re-normalize after EQ (EQ may shift level despite 0 dB mean target)
        rms_post = math.sqrt(np.mean(stereo ** 2))
        if rms_post > 1e-10:
            scale = min(target_rms / rms_post, 0.95 / np.abs(stereo).max())
            stereo = stereo * scale

    # Stereo width from sample-derived factor + parametric boost
    # width_factor stored per-note in spectral_eq by compute_spectral_eq.py
    width_factor = 1.0
    if eq_data:
        width_factor = float(eq_data.get('stereo_width_factor', 1.0) or 1.0)
    if abs(width_factor * stereo_boost - 1.0) > 0.01:
        stereo = apply_stereo_width(stereo, width_factor, stereo_boost)
        # Re-normalize after width scaling (louder side increases RMS)
        rms_w = math.sqrt(np.mean(stereo ** 2))
        if rms_w > 1e-10:
            scale = min(target_rms / rms_w, 0.95 / np.abs(stereo).max())
            stereo = stereo * scale

    # Short onset ramp: oscillators start at cos(phi) which is generally non-zero.
    # Linear ramp from 0 eliminates the click without affecting perceived attack.
    n_onset = min(int(onset_ms * 0.001 * sr), n // 10)
    if n_onset > 1:
        ramp = np.linspace(0.0, 1.0, n_onset, dtype=np.float32)
        stereo[:n_onset] *= ramp[:, np.newaxis]

    return stereo


# ── Convenience functions ─────────────────────────────────────────────────────

def synthesize_and_save(params_path: str, midi: int, vel: int,
                        out_path: str = None,
                        duration: float = None,
                        sr: int = 44100,
                        soundboard_strength: float = 0.0,
                        beat_scale: float = 1.0,
                        pan_spread: float = 0.55,
                        eq_strength: float = 1.0) -> str:
    with open(params_path) as f:
        data = json.load(f)

    key = f"m{midi:03d}_vel{vel}"
    if key not in data['samples']:
        raise ValueError(f"Key {key} not found in {params_path}")

    sample = data['samples'][key]
    audio  = synthesize_note(sample, duration=duration, sr=sr,
                             soundboard_strength=soundboard_strength,
                             beat_scale=beat_scale, pan_spread=pan_spread,
                             eq_strength=eq_strength)

    if out_path is None:
        Path('analysis').mkdir(exist_ok=True)
        out_path = f'analysis/synth_{key}.wav'

    sf.write(out_path, audio, sr, subtype='PCM_16')
    n_str = n_strings_for_midi(midi)
    print(f"Synthesized {key} ({n_str} strings/harmonic, sb={soundboard_strength:.2f}, beat_scale={beat_scale:.1f}) -> {out_path}")
    return out_path


def compare_to_original(params_path: str, bank_dir: str,
                         midi: int, vel: int,
                         soundboard_strength: float = 0.0,
                         beat_scale: float = 1.0,
                         out_dir: str = 'analysis'):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    synth_path = str(out_dir / f'synth_m{midi:03d}_vel{vel}.wav')
    synthesize_and_save(params_path, midi, vel, synth_path, duration=6.0,
                        soundboard_strength=soundboard_strength,
                        beat_scale=beat_scale)

    orig_name = f'm{midi:03d}-vel{vel}-f44.wav'
    orig_path = Path(bank_dir) / orig_name
    if orig_path.exists():
        audio_orig, sr_orig = sf.read(str(orig_path), dtype='float32', always_2d=True)
        if audio_orig.shape[1] > 1:
            audio_orig = audio_orig.mean(axis=1)
        else:
            audio_orig = audio_orig[:, 0]
        n = min(len(audio_orig), int(6.0 * sr_orig))
        audio_orig = audio_orig[:n]
        peak = np.abs(audio_orig).max()
        if peak > 1e-10:
            audio_orig = audio_orig / peak * 0.9
        orig_out = str(out_dir / f'orig_m{midi:03d}_vel{vel}.wav')
        sf.write(orig_out, audio_orig, sr_orig, subtype='PCM_16')
        print(f"Original -> {orig_out}")


def synthesize_preview_set(params_path: str,
                           out_dir: str = 'analysis/synth_preview',
                           soundboard_strength: float = 0.0,
                           beat_scale: float = 1.0,
                           duration: float = 5.0,
                           target_rms: float = 0.06,
                           vel_gamma: float = 0.7):
    """Synthesize representative notes for listening evaluation.

    Applies gamma velocity curve: rms = target_rms * ((vel+1)/8)^gamma
    so velocity layers are correctly scaled relative to each other.
    """
    notes = [
        (45, 2), (45, 5),  # A2 bichord
        (60, 2), (60, 5),  # C4 trichord
        (69, 2), (69, 5),  # A4 trichord
        (81, 2), (81, 5),  # A5 trichord
        (93, 2), (93, 5),  # A6 trichord
    ]
    with open(params_path) as f:
        data = json.load(f)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for midi, vel in notes:
        key = f"m{midi:03d}_vel{vel}"
        if key not in data['samples']:
            continue
        vel_rms = target_rms * ((vel + 1) / 8.0) ** vel_gamma
        audio = synthesize_note(data['samples'][key], duration=duration,
                                soundboard_strength=soundboard_strength,
                                beat_scale=beat_scale,
                                target_rms=vel_rms)
        out = f'{out_dir}/{key}.wav'
        sf.write(out, audio, 44100, subtype='PCM_16')
        n_str = n_strings_for_midi(midi)
        rms_actual = float(__import__('numpy').sqrt(__import__('numpy').mean(audio**2)))
        print(f"  {key} ({n_str}str)  vel_rms={vel_rms:.4f}  actual={rms_actual:.4f}  -> {out}")
    print(f"Preview written to {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params',    default='analysis/params.json')
    parser.add_argument('--midi',      type=int,   default=60)
    parser.add_argument('--vel',       type=int,   default=3)
    parser.add_argument('--out',       default=None)
    parser.add_argument('--duration',  type=float, default=None)
    parser.add_argument('--sr',        type=int,   default=44100)
    parser.add_argument('--soundboard', type=float, default=0.0,
                        help='Soundboard strength 0.0 (default/bypass) to 1.0 (full). PARKED.')
    parser.add_argument('--beat-scale', type=float, default=1.0,
                        help='Beat frequency multiplier (1.0=extracted, 1.5-2.5=more vivid)')
    parser.add_argument('--compare',   action='store_true')
    parser.add_argument('--preview',   action='store_true')
    parser.add_argument('--bank',      default='C:/SoundBanks/IthacaPlayer/ks-grand')
    args = parser.parse_args()

    if args.preview:
        synthesize_preview_set(args.params, soundboard_strength=args.soundboard,
                               beat_scale=args.beat_scale,
                               duration=args.duration or 5.0)
    elif args.compare:
        compare_to_original(args.params, args.bank, args.midi, args.vel,
                            soundboard_strength=args.soundboard,
                            beat_scale=args.beat_scale)
    else:
        synthesize_and_save(args.params, args.midi, args.vel,
                            out_path=args.out, duration=args.duration,
                            sr=args.sr, soundboard_strength=args.soundboard,
                            beat_scale=args.beat_scale)


if __name__ == '__main__':
    main()
