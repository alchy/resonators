"""
data/prepare.py
Preprocess Salamander piano soundbank for EGRB training.

For each m{MIDI}-vel{VEL}-f48.wav:
  1. Parse MIDI note and velocity index
  2. Compute f0 = 440 * 2^((midi-69)/12)
  3. Load 48kHz stereo audio
  4. Split into FRAME_SIZE-sample frames
  5. Compute per-frame RMS envelope
  6. Detect envelope phases: 0=attack 1=sustain 2=decay 3=release
  7. Save NPZ to data/prepared/

Also writes data/manifest.json.

Usage:
    python data/prepare.py
    python data/prepare.py --config config.json --source C:/SoundBanks/ddsp/salamander
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

PHASE_ATTACK  = 0
PHASE_SUSTAIN = 1
PHASE_DECAY   = 2
PHASE_RELEASE = 3

SR         = 48000
FRAME_SIZE = 256


def midi_to_f0(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def parse_filename(name: str):
    """m{MIDI}-vel{VEL}-f{SR_KHZ}.wav  →  (midi, vel_idx, sr_khz)  or None"""
    m = re.match(r'm(\d{3})-vel(\d)-f(\d+)\.wav', name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compute_rms_frames(audio: np.ndarray, frame_size: int) -> np.ndarray:
    """audio: (2, T)  →  rms per frame (T_frames,)"""
    mono = audio.mean(axis=0)
    T = mono.shape[0]
    n_frames = T // frame_size
    mono_cut = mono[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt((mono_cut ** 2).mean(axis=1) + 1e-12)
    return rms


def detect_envelope_phases(rms: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Returns int64 array of phase labels per frame.

    Strategy:
      1. Smooth RMS with a box filter.
      2. Find the peak frame → everything before it is ATTACK.
      3. After the peak compute the normalised derivative.
         While |deriv| < sustain_deriv_thresh  →  SUSTAIN
         Once  |deriv| >= threshold            →  DECAY
      4. Once RMS drops below release_rms_thresh  →  RELEASE (rest of file).
    """
    smooth_n       = int(cfg.get('smooth_frames',        5))
    release_thresh = float(cfg.get('release_rms_thresh', 0.005))
    sus_thresh     = float(cfg.get('sustain_deriv_thresh', 0.008))

    kernel = np.ones(smooth_n) / smooth_n
    rms_s  = np.convolve(rms, kernel, mode='same')
    rms_max = rms_s.max() + 1e-12

    n = len(rms)
    phases = np.zeros(n, dtype=np.int64)

    peak_idx = int(np.argmax(rms_s))
    phases[: peak_idx + 1] = PHASE_ATTACK

    # Normalised first-difference after peak
    deriv_norm = np.diff(rms_s / rms_max, prepend=rms_s[0] / rms_max)

    in_decay = False
    for i in range(peak_idx + 1, n):
        if rms_s[i] < release_thresh:
            phases[i:] = PHASE_RELEASE
            break

        if not in_decay and abs(deriv_norm[i]) > sus_thresh:
            in_decay = True

        phases[i] = PHASE_DECAY if in_decay else PHASE_SUSTAIN

    return phases


def load_stereo_48k(path: str) -> np.ndarray:
    """Load audio as float32 stereo (2, T) at 48 kHz."""
    audio, file_sr = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.T  # (C, T)

    if file_sr != SR:
        import librosa
        audio = np.stack([
            librosa.resample(ch, orig_sr=file_sr, target_sr=SR)
            for ch in audio
        ])

    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])
    elif audio.shape[0] > 2:
        audio = audio[:2]

    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak

    return audio.astype(np.float32)


def process_file(wav_path: str, out_dir: str, env_cfg: dict) -> dict | None:
    stem   = Path(wav_path).stem
    parsed = parse_filename(Path(wav_path).name)
    if parsed is None:
        print(f"  skip (bad name): {wav_path}")
        return None

    midi_note, vel_idx, _ = parsed
    f0       = midi_to_f0(midi_note)
    vel_norm = vel_idx / 7.0

    audio    = load_stereo_48k(wav_path)
    rms      = compute_rms_frames(audio, FRAME_SIZE)
    n_frames = rms.shape[0]
    phases   = detect_envelope_phases(rms, env_cfg)

    audio_trimmed = audio[:, : n_frames * FRAME_SIZE]

    out_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(
        out_path,
        audio    = audio_trimmed,
        rms      = rms.astype(np.float32),
        phases   = phases,
        f0       = np.float32(f0),
        vel_norm = np.float32(vel_norm),
        midi_note = np.int64(midi_note),
        vel_idx   = np.int64(vel_idx),
    )

    counts = {
        'attack':  int((phases == PHASE_ATTACK).sum()),
        'sustain': int((phases == PHASE_SUSTAIN).sum()),
        'decay':   int((phases == PHASE_DECAY).sum()),
        'release': int((phases == PHASE_RELEASE).sum()),
    }
    return {
        'stem':       stem,
        'npz':        out_path,
        'midi_note':  midi_note,
        'vel_idx':    vel_idx,
        'f0':         float(f0),
        'vel_norm':   float(vel_norm),
        'n_frames':   int(n_frames),
        'duration_s': float(n_frames * FRAME_SIZE / SR),
        'phase_counts': counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--source', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    source_dir = args.source or cfg['source_dir']
    out_dir    = cfg['data_dir']
    env_cfg    = cfg['envelope']

    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(r'm\d{3}-vel\d-f\d+\.wav')
    wav_files = sorted([
        os.path.join(source_dir, fn)
        for fn in os.listdir(source_dir)
        if pattern.match(fn)
    ])

    print(f"Found {len(wav_files)} WAV files in {source_dir}")
    if not wav_files:
        print("Nothing to process.")
        sys.exit(1)

    manifest = []
    for i, wav_path in enumerate(wav_files):
        name = Path(wav_path).name
        print(f"  [{i+1:3d}/{len(wav_files)}] {name}", end='  ', flush=True)
        meta = process_file(wav_path, out_dir, env_cfg)
        if meta:
            manifest.append(meta)
            pc = meta['phase_counts']
            print(
                f"f0={meta['f0']:7.1f}Hz  "
                f"A={pc['attack']:3d} S={pc['sustain']:3d} "
                f"D={pc['decay']:3d} R={pc['release']:3d}  "
                f"({meta['duration_s']:.2f}s)"
            )

    manifest_path = os.path.join(os.path.dirname(out_dir) or '.', 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} files -> {out_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == '__main__':
    main()
