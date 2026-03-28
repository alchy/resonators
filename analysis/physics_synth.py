"""
analysis/physics_synth.py
─────────────────────────
Phase 1: Pure physical synthesizer using extracted parameters.

Synthesizes a piano note from params.json parameters.
No neural network — pure physics equations.

This is a benchmark / reference synthesizer that tells us:
  a) How good analytical parameter extraction is
  b) What the ceiling quality is for the chosen physical model
  c) What physics is missing (heard by comparing to real sample)

Synthesis model:
  Audio(t) = Σ_k A_k(t) · cos(2π·f_k·t + φ_k)  [harmonic sum]
           + noise(t)                              [shaped noise]

Where:
  f_k   = k · f0 · √(1 + B·k²)                  [inharmonicity]
  A_k(t) = [a1_k·exp(-t/τ1_k) + (1-a1_k)·exp(-t/τ2_k)]  [bi-exp decay]
           · [1 + m_k · cos(2π·Δf_k·t + ψ_k)]             [beating modulation]
  noise(t) = A_noise · exp(-t/τ_noise) · N(t)             [attack burst]

Usage:
    python analysis/physics_synth.py --params analysis/params.json
                                     --midi 60 --vel 3
                                     [--out analysis/synth_m060_vel3.wav]
                                     [--duration 4.0]
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import soundfile as sf


# ── Core synthesis ────────────────────────────────────────────────────────────

def synthesize_note(params: dict, duration: float = None,
                    sr: int = 44100,
                    fade_out: float = 0.5) -> np.ndarray:
    """
    Synthesize a single piano note from extracted physical parameters.

    Args:
        params: dict from params.json['samples'][key]
        duration: total duration in seconds (None = use extracted duration)
        sr: output sample rate
        fade_out: fade-out length at end in seconds

    Returns:
        mono audio array (float32, normalized)
    """
    if duration is None:
        duration = min(params.get('duration_s', 4.0), 8.0)  # cap at 8s for playback

    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr

    audio = np.zeros(n_samples, dtype=np.float64)

    partials = params.get('partials', [])
    if not partials:
        return audio.astype(np.float32)

    # Normalize partial amplitudes by peak A0 of k=1
    A0_ref = partials[0]['A0'] if partials[0]['A0'] else 1.0
    if A0_ref is None or A0_ref < 1e-10:
        A0_ref = max((p['A0'] or 0) for p in partials)
    if A0_ref < 1e-10:
        A0_ref = 1.0

    # --- Harmonic sum ---
    for p in partials:
        k = p['k']
        f_hz = p['f_hz']
        A0 = p['A0']

        if A0 is None or A0 < 1e-10 or f_hz > sr / 2 * 0.99:
            continue

        # Amplitude normalization (relative to k=1)
        amp_norm = A0 / A0_ref

        # Bi-exponential decay envelope
        tau1 = p.get('tau1') or 3.0
        tau2 = p.get('tau2')
        a1   = p.get('a1', 1.0)
        if a1 is None:
            a1 = 1.0

        if tau2 is not None and not p.get('mono', True):
            # Bi-exponential
            a2 = 1.0 - a1
            env = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
        else:
            # Single exponential
            env = np.exp(-t / tau1)

        # Beating modulation
        beat_hz = p.get('beat_hz', 0.0) or 0.0
        beat_depth = p.get('beat_depth', 0.0) or 0.0
        if beat_hz > 0 and beat_depth > 0.02:
            psi = np.random.uniform(0, 2 * math.pi)  # random phase for beating
            beat_env = 1.0 + beat_depth * np.cos(2 * math.pi * beat_hz * t + psi)
            env = env * beat_env

        # Phase (random initial phase for each partial)
        phi = np.random.uniform(0, 2 * math.pi)
        oscillator = np.cos(2 * math.pi * f_hz * t + phi)

        audio += amp_norm * env * oscillator

    # --- Noise component (attack burst) ---
    noise_params = params.get('noise', {})
    tau_noise = noise_params.get('attack_tau_s', 0.05) or 0.05
    floor_rms = noise_params.get('floor_rms', 0.001) or 0.001

    # Attack burst: broadband noise with exponential decay
    noise_raw = np.random.randn(n_samples)

    # Spectral shaping: apply gentle low-pass to make it less white
    centroid = noise_params.get('centroid_hz', 2000.0) or 2000.0
    # Simple first-order IIR low-pass approximation
    alpha = 1.0 - math.exp(-2 * math.pi * centroid / sr)
    noise_shaped = np.zeros_like(noise_raw)
    y = 0.0
    for i in range(n_samples):
        y = alpha * noise_raw[i] + (1 - alpha) * y
        noise_shaped[i] = y

    # Noise envelope: attack burst + floor
    noise_env = np.exp(-t / tau_noise) + floor_rms
    noise_signal = noise_shaped * noise_env

    # Mix noise at ~10% of harmonic level
    noise_level = 0.1
    audio += noise_level * noise_signal

    # --- Fade out ---
    if fade_out > 0 and duration > fade_out:
        fade_samples = int(fade_out * sr)
        fade_win = np.linspace(1.0, 0.0, fade_samples)
        audio[-fade_samples:] *= fade_win

    # --- Normalize ---
    peak = np.abs(audio).max()
    if peak > 1e-10:
        audio = audio / peak * 0.9

    return audio.astype(np.float32)


def synthesize_and_save(params_path: str, midi: int, vel: int,
                        out_path: str = None, duration: float = None,
                        sr: int = 44100) -> str:
    """Load params and synthesize one note. Returns output path."""
    with open(params_path) as f:
        data = json.load(f)

    key = f"m{midi:03d}_vel{vel}"
    if key not in data['samples']:
        raise ValueError(f"Key {key} not found in {params_path}")

    sample = data['samples'][key]
    audio = synthesize_note(sample, duration=duration, sr=sr)

    if out_path is None:
        Path('analysis').mkdir(exist_ok=True)
        out_path = f'analysis/synth_{key}.wav'

    sf.write(out_path, audio, sr, subtype='PCM_16')
    print(f"Synthesized {key} -> {out_path}  (dur={len(audio)/sr:.2f}s)")
    return out_path


def synthesize_bank_subset(params_path: str, out_dir: str = 'analysis/synth_preview',
                           sample_midis: list = None, velocities: list = None,
                           duration: float = 4.0, sr: int = 44100):
    """
    Synthesize a subset of notes for listening evaluation.
    Default: A2, C4, C5, C6, A6 at vel 2,5 — covers register and dynamics.
    """
    if sample_midis is None:
        sample_midis = [45, 57, 60, 69, 72, 81, 93]  # A2, A3, C4, A4, C5, A5, A6
    if velocities is None:
        velocities = [2, 5]

    with open(params_path) as f:
        data = json.load(f)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for midi in sample_midis:
        for vel in velocities:
            key = f"m{midi:03d}_vel{vel}"
            if key not in data['samples']:
                print(f"  [skip] {key} not in params")
                continue
            sample = data['samples'][key]
            audio = synthesize_note(sample, duration=duration, sr=sr)
            out_path = f'{out_dir}/{key}.wav'
            sf.write(out_path, audio, sr, subtype='PCM_16')
            print(f"  {key} -> {out_path}")

    print(f"\nPreview set written to {out_dir}/")


# ── Comparison utility ────────────────────────────────────────────────────────

def compare_to_original(params_path: str, bank_dir: str,
                         midi: int, vel: int, out_dir: str = 'analysis'):
    """
    Write both original sample and synthesized version for A/B comparison.
    """
    sr_target = 44100
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Synthesize
    synth_path = str(out_dir / f'synth_m{midi:03d}_vel{vel}.wav')
    synthesize_and_save(params_path, midi, vel, synth_path, duration=6.0, sr=sr_target)

    # Copy/trim original
    orig_name = f'm{midi:03d}-vel{vel}-f44.wav'
    orig_path = Path(bank_dir) / orig_name
    if orig_path.exists():
        import soundfile as sf
        audio_orig, sr_orig = sf.read(str(orig_path), dtype='float32', always_2d=True)
        if audio_orig.shape[1] > 1:
            audio_orig = audio_orig.mean(axis=1)
        else:
            audio_orig = audio_orig[:, 0]
        # Trim to 6s
        n = min(len(audio_orig), int(6.0 * sr_orig))
        audio_orig = audio_orig[:n]
        # Normalize
        peak = np.abs(audio_orig).max()
        if peak > 1e-10:
            audio_orig = audio_orig / peak * 0.9
        orig_out = str(out_dir / f'orig_m{midi:03d}_vel{vel}.wav')
        sf.write(orig_out, audio_orig, sr_orig, subtype='PCM_16')
        print(f"Original -> {orig_out}")
    else:
        print(f"[WARN] Original not found: {orig_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Physics-based piano synthesizer from extracted params')
    parser.add_argument('--params', default='analysis/params.json')
    parser.add_argument('--midi', type=int, default=60)
    parser.add_argument('--vel', type=int, default=3)
    parser.add_argument('--out', default=None, help='Output WAV path')
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--preview', action='store_true',
                        help='Synthesize preview set across keyboard')
    parser.add_argument('--compare', action='store_true',
                        help='Also output original sample for A/B comparison')
    parser.add_argument('--bank', default='C:/SoundBanks/IthacaPlayer/ks-grand')
    args = parser.parse_args()

    if args.preview:
        synthesize_bank_subset(args.params, duration=args.duration or 4.0, sr=args.sr)
    elif args.compare:
        compare_to_original(args.params, args.bank, args.midi, args.vel)
    else:
        synthesize_and_save(args.params, args.midi, args.vel,
                            out_path=args.out, duration=args.duration, sr=args.sr)


if __name__ == '__main__':
    main()
