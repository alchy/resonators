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

Usage:
    python analysis/physics_synth.py --params analysis/params.json
                                     --midi 60 --vel 3 --duration 6
                                     [--soundboard 0.0]
                                     [--beat-scale 1.5]
                                     [--compare]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


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


# ── Core synthesis ────────────────────────────────────────────────────────────

def synthesize_note(params: dict, duration: float = None,
                    sr: int = 44100,
                    soundboard_strength: float = 0.0,
                    beat_scale: float = 1.0,
                    fade_out: float = 0.5,
                    rng_seed: int = None) -> np.ndarray:
    """
    Synthesize a piano note using physically correct multi-string beating.

    Args:
        params:              dict from params.json['samples'][key]
        duration:            total duration in seconds (None = extracted)
        sr:                  output sample rate
        soundboard_strength: [0..1] virtual soundboard mix (PARKED: default 0.0)
        beat_scale:          multiplier on beat_hz — increase for more pronounced
                             string interference (try 1.5-2.5); default 1.0
        fade_out:            fade-out duration at end (seconds)
        rng_seed:            for reproducible output

    Returns:
        mono audio (float32, normalized to ~0.9)
    """
    rng = np.random.default_rng(rng_seed)

    if duration is None:
        duration = min(params.get('duration_s', 4.0), 8.0)

    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr

    audio = np.zeros(n_samples, dtype=np.float64)

    partials = params.get('partials', [])
    if not partials:
        return audio.astype(np.float32)

    # Reference amplitude (k=1)
    A0_ref = next((p['A0'] for p in partials if p['A0'] and p['A0'] > 1e-10), 1.0)

    midi = params.get('midi', 60)
    n_str = n_strings_for_midi(midi)

    # ── Harmonic sum with N_strings independent oscillators ──────────────────
    for p in partials:
        f_hz  = p['f_hz']
        A0    = p['A0']
        k     = p['k']

        if A0 is None or A0 < 1e-10 or f_hz > sr / 2 * 0.99:
            continue

        amp_norm = A0 / A0_ref

        # Bi-exponential decay envelope
        tau1 = p.get('tau1') or 3.0
        tau2 = p.get('tau2')
        a1   = p.get('a1')
        if a1 is None:
            a1 = 1.0

        if tau2 is not None and not p.get('mono', True):
            env = a1 * np.exp(-t / tau1) + (1 - a1) * np.exp(-t / tau2)
        else:
            env = np.exp(-t / tau1)

        # ── Multi-string oscillators ─────────────────────────────────────────
        beat_hz = (p.get('beat_hz', 0.0) or 0.0) * beat_scale

        if n_str == 1 or beat_hz < 0.05:
            # Single oscillator
            phi = rng.uniform(0, 2 * math.pi)
            osc = np.cos(2 * math.pi * f_hz * t + phi)
            audio += amp_norm * env * osc

        elif n_str == 2:
            # Two independent strings at f +/- beat_hz/2
            # Beating emerges from their sum naturally (full-depth)
            phi_a = rng.uniform(0, 2 * math.pi)
            phi_b = rng.uniform(0, 2 * math.pi)  # independent phase -> random beat start

            osc_a = np.cos(2 * math.pi * (f_hz + beat_hz / 2) * t + phi_a)
            osc_b = np.cos(2 * math.pi * (f_hz - beat_hz / 2) * t + phi_b)
            osc   = (osc_a + osc_b) * 0.5  # average: same energy as single osc

            audio += amp_norm * env * osc

        else:  # n_str == 3
            # Three strings: f-delta, f, f+delta (symmetric spacing)
            phi_a = rng.uniform(0, 2 * math.pi)
            phi_b = rng.uniform(0, 2 * math.pi)  # center string
            phi_c = rng.uniform(0, 2 * math.pi)

            delta = beat_hz / 2  # spacing from center to outer strings
            osc_a = np.cos(2 * math.pi * (f_hz - delta) * t + phi_a)  # string low
            osc_b = np.cos(2 * math.pi * f_hz * t + phi_b)             # string center
            osc_c = np.cos(2 * math.pi * (f_hz + delta) * t + phi_c)  # string high
            osc   = (osc_a + osc_b + osc_c) / 3.0

            audio += amp_norm * env * osc

    # ── Noise (attack burst only — no persistent floor) ───────────────────────
    # Persistent noise floor was causing audible high-frequency droning.
    # Noise is now purely transient: strong at attack, decays with tau_noise.
    noise_params = params.get('noise', {})
    tau_noise = noise_params.get('attack_tau_s', 0.05) or 0.05
    centroid  = noise_params.get('centroid_hz', 3000.0) or 3000.0
    # Use extracted centroid directly -- encodes hammer brightness per MIDI.
    # The 2000 Hz cap was killing the metallic attack ("cink") in treble.

    noise_raw = rng.standard_normal(n_samples)

    # First-order IIR low-pass at centroid_hz (capped at Nyquist safety margin)
    alpha_lp = 1.0 - math.exp(-2 * math.pi * min(centroid, sr * 0.45) / sr)
    noise_shaped = np.zeros_like(noise_raw)
    y = 0.0
    for i in range(n_samples):
        y = alpha_lp * noise_raw[i] + (1 - alpha_lp) * y
        noise_shaped[i] = y

    # Pure attack envelope — no constant floor
    noise_env    = np.exp(-t / max(tau_noise, 0.001))
    noise_signal = noise_shaped * noise_env
    noise_level  = 0.06  # 0.04 was too quiet; bright attack needs presence
    audio       += noise_level * noise_signal

    # ── Soundboard (parked) ───────────────────────────────────────────────────
    if soundboard_strength > 0.005:
        audio = apply_soundboard(audio, sr, soundboard_strength)

    # ── Fade out ─────────────────────────────────────────────────────────────
    if fade_out > 0 and duration > fade_out:
        n_fade = int(fade_out * sr)
        audio[-n_fade:] *= np.linspace(1.0, 0.0, n_fade)

    # ── Normalize ─────────────────────────────────────────────────────────────
    peak = np.abs(audio).max()
    if peak > 1e-10:
        audio = audio / peak * 0.9

    return audio.astype(np.float32)


# ── Convenience functions ─────────────────────────────────────────────────────

def synthesize_and_save(params_path: str, midi: int, vel: int,
                        out_path: str = None,
                        duration: float = None,
                        sr: int = 44100,
                        soundboard_strength: float = 0.0,
                        beat_scale: float = 1.0) -> str:
    with open(params_path) as f:
        data = json.load(f)

    key = f"m{midi:03d}_vel{vel}"
    if key not in data['samples']:
        raise ValueError(f"Key {key} not found in {params_path}")

    sample = data['samples'][key]
    audio  = synthesize_note(sample, duration=duration, sr=sr,
                             soundboard_strength=soundboard_strength,
                             beat_scale=beat_scale)

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
                           duration: float = 5.0):
    """Synthesize representative notes for listening evaluation."""
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
        audio = synthesize_note(data['samples'][key], duration=duration,
                                soundboard_strength=soundboard_strength,
                                beat_scale=beat_scale)
        out = f'{out_dir}/{key}.wav'
        sf.write(out, audio, 44100, subtype='PCM_16')
        n_str = n_strings_for_midi(midi)
        print(f"  {key} ({n_str}str) -> {out}")
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
