"""
generate.py
Synthesise a full piano sample bank from a trained EGRB model.

For each MIDI note (21–108) × velocity (0–7):
  1. Compute f0 from MIDI, vel_norm from velocity index.
  2. Build a synthetic loudness envelope from a reference NPZ template
     (nearest MIDI note + velocity in the training set).
  3. Run controller + resonator bank to generate audio.
  4. Peak-normalise and save as 16-bit WAV.

Output filename convention: m{MIDI:03d}-vel{VEL}-f48.wav

Usage:
    python generate.py --checkpoint checkpoints/best.pt
    python generate.py --checkpoint checkpoints/best.pt --output generated/ --notes 60 72 --vels 4
"""

import os
import sys
import json
import argparse
import re
import numpy as np
import soundfile as sf
import torch

from models.controller    import GRUController
from models.resonator_bank import ResonatorBank


SR         = 48000
FRAME_SIZE = 256


def midi_to_f0(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def nearest_template(manifest: list, midi_note: int, vel_idx: int) -> dict:
    """Find the manifest entry closest in (midi_note, vel_idx) to the requested note."""
    def dist(e):
        return abs(e['midi_note'] - midi_note) * 10 + abs(e['vel_idx'] - vel_idx)
    return min(manifest, key=dist)


def build_envelope_from_template(template_path: str, target_frames: int) -> np.ndarray:
    """
    Load RMS and phase labels from a template NPZ and resample to target_frames.
    Returns: rms (target_frames,), phases (target_frames,)
    """
    d = np.load(template_path)
    rms    = d['rms'].astype(np.float32)
    phases = d['phases'].astype(np.int64)

    src_n = len(rms)
    if src_n == target_frames:
        return rms, phases

    # Resample via linear interpolation
    idx_src = np.linspace(0, src_n - 1, target_frames)
    rms_r   = np.interp(idx_src, np.arange(src_n), rms).astype(np.float32)

    # Nearest-neighbour for discrete phases
    idx_nn  = np.clip(idx_src.round().astype(int), 0, src_n - 1)
    phases_r = phases[idx_nn].astype(np.int64)

    return rms_r, phases_r


@torch.no_grad()
def synthesise_note(
    controller: GRUController,
    bank:       ResonatorBank,
    f0:         float,
    vel_norm:   float,
    rms:        np.ndarray,     # (T_frames,)
    phases_arr: np.ndarray,     # (T_frames,) int
    device:     torch.device,
) -> np.ndarray:
    """Returns (2, T_samples) float32 audio."""
    T_frames = len(rms)

    f0_t       = torch.tensor([f0],       device=device)
    vel_t      = torch.tensor([vel_norm], device=device)
    rms_t      = torch.from_numpy(rms).unsqueeze(0).to(device)      # (1, T_frames)
    phases_t   = torch.from_numpy(phases_arr).unsqueeze(0).to(device)  # (1, T_frames)

    control, gates = controller(f0_t, vel_t, rms_t, phases_t)
    audio = bank(f0_t, vel_t, control, gates, phases_t)  # (1, 2, T_samples)

    audio_np = audio.squeeze(0).cpu().numpy()  # (2, T_samples)

    peak = np.abs(audio_np).max()
    if peak > 1e-6:
        audio_np = audio_np / peak * 0.95

    return audio_np.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--config',     default=None,
                        help='Override config (default: taken from checkpoint)')
    parser.add_argument('--output',     default='generated')
    parser.add_argument('--duration',   type=float, default=3.0,
                        help='Duration in seconds (default 3.0)')
    parser.add_argument('--notes', nargs='*', type=int, default=None,
                        help='MIDI notes to generate (default: 21-108 step 3)')
    parser.add_argument('--vels',  nargs='*', type=int, default=None,
                        help='Velocity indices 0-7 (default: all 8)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg  = ckpt['cfg'] if args.config is None else json.load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    controller = GRUController(cfg).to(device)
    bank       = ResonatorBank(cfg).to(device)
    controller.load_state_dict(ckpt['controller'])
    bank.load_state_dict(ckpt['bank'])
    controller.eval()
    bank.eval()

    # Load manifest for templates
    manifest_path = os.path.join(
        os.path.dirname(cfg.get('data_dir', 'data/prepared')) or '.', 'manifest.json'
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    notes = args.notes or list(range(21, 109, 3))
    vels  = args.vels  or list(range(8))
    T_frames = int(args.duration * SR / FRAME_SIZE)

    total = len(notes) * len(vels)
    done  = 0
    for midi_note in notes:
        for vel_idx in vels:
            f0       = midi_to_f0(midi_note)
            vel_norm = vel_idx / 7.0

            tmpl = nearest_template(manifest, midi_note, vel_idx)
            rms, phases_arr = build_envelope_from_template(tmpl['npz'], T_frames)

            audio = synthesise_note(
                controller, bank, f0, vel_norm, rms, phases_arr, device
            )

            name    = f"m{midi_note:03d}-vel{vel_idx}-f48.wav"
            out_path = os.path.join(args.output, name)
            sf.write(out_path, audio.T, SR, subtype='PCM_16')

            done += 1
            print(f"  [{done:4d}/{total}]  {name}  f0={f0:.1f}Hz")

    print(f"\nDone. {done} files → {args.output}/")


if __name__ == '__main__':
    main()
