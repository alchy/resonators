"""
generate.py
Synthesise a full piano sample bank from a trained EGRB model.

For each MIDI note (21–108) × velocity (0–7):
  1. Compute f0 from MIDI, vel_norm from velocity index.
  2. EnvelopeNet predicts RMS curve + ADSR phase labels from (midi_norm, vel_norm).
     No training samples are needed at inference — the envelope is fully generalised.
  3. GRUController + ResonatorBank synthesise audio conditioned on the envelope.
  4. Peak-normalise and save as 16-bit WAV.

Requires:
  checkpoints/best.pt      — main model checkpoint
  checkpoints/envelope.pt  — EnvelopeNet checkpoint (produced by train.py)

Output filename convention: m{MIDI:03d}-vel{VEL}-f48.wav

Usage:
    python generate.py --checkpoint checkpoints/best.pt
    python generate.py --checkpoint checkpoints/best.pt --output generated/ --notes 60 72 --vels 4
"""

import os
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import torch

from models.controller     import GRUController
from models.resonator_bank import ResonatorBank
from models.envelope_net   import EnvelopeNet


SR         = 48000
FRAME_SIZE = 256


def midi_to_f0(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


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
                        help='MIDI notes to generate (default: 21-108, all 88 keys)')
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
    # Activate all resonators for generation (curriculum state is not saved in checkpoint)
    bank.set_active_resonators(bank.n_harmonic, bank.n_noise, bank.n_transient)

    # Load EnvelopeNet
    env_ckpt_path = os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'envelope.pt')
    if not os.path.exists(env_ckpt_path):
        print(f"EnvelopeNet checkpoint not found: {env_ckpt_path}")
        print("Run:  python train.py  (or just the envelope pre-training step)")
        sys.exit(1)
    env_ckpt = torch.load(env_ckpt_path, map_location='cpu')
    env_net  = EnvelopeNet(env_ckpt['cfg']).to(device)
    env_net.load_state_dict(env_ckpt['state_dict'])
    env_net.eval()
    print(f"EnvelopeNet loaded from {env_ckpt_path}")

    os.makedirs(args.output, exist_ok=True)

    notes    = args.notes or list(range(21, 109))
    vels     = args.vels  or list(range(8))
    T_frames = int(args.duration * SR / FRAME_SIZE)

    total = len(notes) * len(vels)
    done  = 0
    for midi_note in notes:
        for vel_idx in vels:
            f0         = midi_to_f0(midi_note)
            vel_norm   = vel_idx / 7.0
            midi_norm  = midi_note / 127.0

            rms, phases_arr = env_net.predict(midi_norm, vel_norm, T_frames)

            audio = synthesise_note(
                controller, bank, f0, vel_norm, rms, phases_arr, device
            )

            name    = f"m{midi_note:03d}-vel{vel_idx}-f48.wav"
            out_path = os.path.join(args.output, name)
            sf.write(out_path, audio.T, SR, subtype='PCM_16')

            done += 1
            print(f"  [{done:4d}/{total}]  {name}  f0={f0:.1f}Hz")

    print(f"\nDone. {done} files -> {args.output}/")


if __name__ == '__main__':
    main()
