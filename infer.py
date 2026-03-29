"""
infer.py  —  quick inference from checkpoint
Usage:
    python infer.py [--checkpoint checkpoints/best.pt] [--notes 48,60,72] [--vels 2,4,6]
Renders selected notes/velocities from the trained EGRB model and saves WAV files.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from models.controller    import GRUController
from models.resonator_bank import ResonatorBank


def load_npz(data_dir: str, midi: int, vel: int) -> dict | None:
    """Load prepared NPZ for a given note/velocity (tries 44 and 48 kHz variants)."""
    for suffix in ("f44", "f48"):
        path = os.path.join(data_dir, f"m{midi:03d}-vel{vel}-{suffix}.npz")
        if os.path.exists(path):
            d = np.load(path)
            return {
                "audio":    d["audio"].astype(np.float32),    # (2, T)
                "rms":      d["rms"].astype(np.float32),       # (T_frames,)
                "phases":   d["phases"].astype(np.int64),      # (T_frames,)
                "f0":       float(d["f0"]),
                "vel_norm": float(d["vel_norm"]),
            }
    return None


@torch.no_grad()
def render(
    controller: GRUController,
    bank: ResonatorBank,
    data: dict,
    device: torch.device,
    max_seconds: float = 4.0,
    sr: int = 48000,
) -> np.ndarray:
    """Run full model on one note, return (2, T) float32 numpy."""
    frame_size = bank.frame_size
    max_frames = int(max_seconds * sr / frame_size)

    f0       = torch.tensor([data["f0"]],       dtype=torch.float32, device=device)
    vel_norm = torch.tensor([data["vel_norm"]], dtype=torch.float32, device=device)

    rms_full    = torch.tensor(data["rms"],    device=device).unsqueeze(0)   # (1, T_all)
    phases_full = torch.tensor(data["phases"], device=device).unsqueeze(0)   # (1, T_all)

    T_frames = min(rms_full.shape[1], max_frames)
    rms    = rms_full[:, :T_frames]
    phases = phases_full[:, :T_frames]

    control, gates = controller(f0, vel_norm, rms, phases)
    audio = bank(f0, vel_norm, control, gates, phases)  # (1, 2, T_samples)

    return audio[0].cpu().numpy()  # (2, T_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config",     default="config.json")
    parser.add_argument("--notes", default="36,48,60,72,84",
                        help="Comma-separated MIDI note numbers")
    parser.add_argument("--vels",  default="1,3,5",
                        help="Comma-separated velocity indices (0–7)")
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="Max render duration per note")
    parser.add_argument("--out", default="generated/infer",
                        help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device   = torch.device("cpu")
    data_dir = cfg.get("data_dir", "data/prepared")
    sr       = int(cfg.get("output_sr", cfg.get("sr") or 48000))

    # Load models
    controller = GRUController(cfg).to(device)
    bank       = ResonatorBank(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Support both EMA and plain state_dicts
    ctrl_sd = ckpt.get("ema_controller") or ckpt.get("controller")
    bank_sd = ckpt.get("ema_bank")       or ckpt.get("bank")
    controller.load_state_dict(ctrl_sd)
    bank.load_state_dict(bank_sd)
    controller.eval()
    bank.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: epoch {epoch}")

    # Activate all resonators for inference (phase3 counts)
    rc = cfg.get("resonator_curriculum", {})
    n_h = rc.get("p3_n_harmonic", cfg.get("n_harmonic", 48))
    n_n = rc.get("p3_n_noise",    cfg.get("n_noise",    8))
    n_t = rc.get("p3_n_transient",cfg.get("n_transient",8))
    bank.set_active_resonators(n_h, n_n, n_t)

    os.makedirs(args.out, exist_ok=True)

    notes = [int(x) for x in args.notes.split(",")]
    vels  = [int(x) for x in args.vels.split(",")]

    for midi in notes:
        for vel in vels:
            data = load_npz(data_dir, midi, vel)
            if data is None:
                print(f"  SKIP  m{midi:03d}-vel{vel}  (no NPZ found)")
                continue

            audio = render(controller, bank, data, device,
                           max_seconds=args.seconds, sr=sr)

            # Normalize to -3 dBFS peak
            peak = np.abs(audio).max()
            if peak > 1e-6:
                audio = audio * (0.708 / peak)

            audio_interleaved = audio.T  # (T, 2)
            fname = os.path.join(args.out, f"m{midi:03d}-vel{vel}.wav")
            sf.write(fname, audio_interleaved, sr)
            dur = audio_interleaved.shape[0] / sr
            print(f"  wrote {fname}  ({dur:.2f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
