"""
analysis/train_ddsp.py
──────────────────────
DDSP (Differentiable DSP) training: learn synthesis parameters directly from
original WAV recordings without an intermediate extraction step.

Pipeline:
  WAV files (88×8) → load segments → NN(midi, vel) → differentiable synthesis
                   → multi-scale spectral loss vs original WAV → backprop

The NN architecture is identical to InstrumentProfile (train_instrument_profile.py).
Supervision comes from the original audio instead of extracted parameters.

This avoids all extraction artifacts (noisy tau, wrong partial tracking) because
the model is directly optimised to reproduce the spectrogram.

Usage:
  python analysis/train_ddsp.py
         --wav-dir  C:/SoundBanks/IthacaPlayer/ks-grand
         --out      analysis/params_profile.json
         --model    analysis/profile_ddsp.pt
         [--init    analysis/profile.pt]   # warm-start from extracted profile
         [--epochs  300]
         [--seg     0.5]                   # segment length in seconds
         [--kmax    16]                    # max partials per note
         [--batch   8]                     # notes per gradient step
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim

# Import shared helpers and NN from train_instrument_profile
sys.path.insert(0, str(Path(__file__).parent))
from train_instrument_profile import (
    InstrumentProfile,
    midi_feat, vel_feat, k_feat,
    midi_to_hz, generate_profile,
)


# ── Feature tables (precomputed, no gradients) ────────────────────────────────

def build_feature_tables(sr: int = 44100, k_max: int = 32) -> dict:
    """
    Precompute all input feature vectors once.
    Returns dict of tensors (no grad, reused every step).
    """
    midi_feats = {m: midi_feat(m) for m in range(21, 109)}
    vel_feats  = {v: vel_feat(v)  for v in range(8)}
    k_feats    = {k: k_feat(k, k_max=k_max) for k in range(1, k_max + 1)}
    f0s        = {m: midi_to_hz(m) for m in range(21, 109)}
    # Max K per MIDI (partials below Nyquist)
    k_limits   = {m: min(k_max, max(1, int((sr / 2) / f0s[m]))) for m in range(21, 109)}
    return dict(midi_feats=midi_feats, vel_feats=vel_feats, k_feats=k_feats,
                f0s=f0s, k_limits=k_limits, k_max=k_max, sr=sr)


# ── WAV loading ───────────────────────────────────────────────────────────────

def load_wav_bank(wav_dir: str, sr_target: int = 44100,
                  seg_secs: float = 0.5) -> dict:
    """
    Load WAV segments from a sample bank directory.
    Returns {key: mono_tensor} where key = 'm064_vel4'.
    """
    wav_dir = Path(wav_dir)
    seg_len = int(seg_secs * sr_target)
    bank: dict = {}

    for midi in range(21, 109):
        for vel in range(8):
            fname = wav_dir / f"m{midi:03d}-vel{vel}-f44.wav"
            if not fname.exists():
                continue
            audio, sr = sf.read(str(fname), dtype='float32')
            if sr != sr_target:
                continue
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if len(audio) >= seg_len:
                audio = audio[:seg_len]
            else:
                audio = np.pad(audio, (0, seg_len - len(audio)))
            bank[f"m{midi:03d}_vel{vel}"] = torch.from_numpy(audio)

    print(f"Loaded {len(bank)} WAV segments ({seg_secs:.1f}s, {sr_target} Hz)")
    return bank


# ── Batched differentiable synthesis ─────────────────────────────────────────

def synth_batch(model: InstrumentProfile,
                midi_list: list, vel_list: list,
                feat: dict, seg_len: int) -> torch.Tensor:
    """
    Synthesise a batch of notes simultaneously.
    All NN calls are vectorised over the batch.

    Returns: [N, seg_len] waveform tensor (differentiable)
    """
    sr    = feat["sr"]
    k_max = feat["k_max"]
    N     = len(midi_list)

    # Use the maximum K across the batch (zero-mask the rest)
    K = max(feat["k_limits"][m] for m in midi_list)
    K = min(K, k_max)

    # Feature tensors — [N, D]
    mf_n  = torch.stack([feat["midi_feats"][m] for m in midi_list])  # [N, MIDI_DIM]
    vf_n  = torch.stack([feat["vel_feats"][v]  for v in vel_list])   # [N, VEL_DIM]
    kf_k  = torch.stack([feat["k_feats"][k]    for k in range(1, K + 1)])  # [K, K_DIM]

    # Expand to [N*K, D] for batched MLP calls
    mf_nk = mf_n.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)   # [N*K, MIDI_DIM]
    vf_nk = vf_n.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)   # [N*K, VEL_DIM]
    kf_nk = kf_k.unsqueeze(0).expand(N, K, -1).reshape(N * K, -1)   # [N*K, K_DIM]

    # NN forward — single pass per sub-network (clamp prevents exp overflow)
    B_n       = torch.exp(model.B_net(mf_n).clamp(-20, 0)).squeeze(-1)           # [N] >0
    tau1_k1_n = torch.exp(model.forward_tau1_k1(mf_n, vf_n).clamp(-5, 4)).squeeze(-1)  # [N] 0.007..55s
    tau_ratio = model.forward_tau_ratio(mf_nk, kf_nk).squeeze(-1).reshape(N, K)  # [N, K]
    A0_nk     = torch.exp(model.forward_A0(mf_nk, kf_nk, vf_nk).clamp(-10, 8)).squeeze(-1).reshape(N, K)
    df_nk     = torch.exp(model.forward_df(mf_nk, kf_nk).clamp(-5, 4)).squeeze(-1).reshape(N, K)

    # tau1 per partial: k=1 from tau1_k1_net; k>1 scaled by ratio
    k1_zero       = torch.zeros(N, 1)
    ratio_clamped = torch.cat([k1_zero, tau_ratio[:, 1:].clamp(-4.0, 0.0)], dim=1)  # [N, K]
    tau1_nk       = (tau1_k1_n.unsqueeze(1) * torch.exp(ratio_clamped)).clamp(0.005)  # [N, K]

    # Inharmonic frequencies: f_k = k * f0 * sqrt(1 + B * k²)
    k_vals = torch.arange(1, K + 1, dtype=torch.float32)    # [K]
    f0_n   = torch.tensor([feat["f0s"][m] for m in midi_list])  # [N]
    f_k_nk = k_vals * f0_n.unsqueeze(1) * torch.sqrt(
        1.0 + B_n.unsqueeze(1) * k_vals ** 2)               # [N, K]

    # Nyquist mask [N, K]
    mask = (f_k_nk < sr / 2).float()
    # Also mask k above each note's k_limit
    k_lim_n = torch.tensor([feat["k_limits"][m] for m in midi_list], dtype=torch.float32)  # [N]
    k_idx   = k_vals.unsqueeze(0).expand(N, -1)              # [N, K]
    mask    = mask * (k_idx <= k_lim_n.unsqueeze(1)).float()

    # Time axis [T]
    t = torch.linspace(0.0, seg_len / sr, seg_len)

    # Envelopes: [N, K, T]
    envelopes = A0_nk.unsqueeze(2) * torch.exp(
        -t / tau1_nk.unsqueeze(2))

    # Sinusoidal phases: [N, K, T]
    phase1 = 2.0 * math.pi * f_k_nk.unsqueeze(2) * t
    phase2 = 2.0 * math.pi * (f_k_nk + df_nk).unsqueeze(2) * t
    components = 0.5 * (torch.cos(phase1) + torch.cos(phase2))

    # Sum over K, apply mask: [N, T]
    signals = (mask.unsqueeze(2) * envelopes * components).sum(1)
    return signals


# ── Spectral loss ─────────────────────────────────────────────────────────────

def multiscale_spectral_loss(synth: torch.Tensor, target: torch.Tensor,
                              fft_sizes: tuple = (256, 2048)) -> torch.Tensor:
    """
    Multi-scale log-magnitude spectral loss for a batch of waveforms.
    synth, target: [N, T] or [T]
    """
    if synth.dim() == 1:
        synth  = synth.unsqueeze(0)
        target = target.unsqueeze(0)
    N = synth.shape[0]

    # Per-note amplitude normalisation (shape comparison)
    synth  = synth  / (synth.std(dim=-1, keepdim=True)  + 1e-8)
    target = target / (target.std(dim=-1, keepdim=True) + 1e-8)

    loss = torch.tensor(0.0)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft)
        # torch.stft supports [B, T] input in PyTorch >= 1.8
        S = torch.stft(synth.reshape(-1, synth.shape[-1]),
                       n_fft=n_fft, hop_length=hop, window=win,
                       return_complex=True).abs()   # [N, F, frames]
        T = torch.stft(target.reshape(-1, target.shape[-1]),
                       n_fft=n_fft, hop_length=hop, window=win,
                       return_complex=True).abs()
        loss = loss + (S.log1p() - T.log1p()).pow(2).mean()
    return loss / len(fft_sizes)


# ── Training ──────────────────────────────────────────────────────────────────

def train_ddsp(model: InstrumentProfile, wav_bank: dict,
               epochs: int = 300, lr: float = 3e-3,
               sr: int = 44100, seg_len: int = 22050,
               k_max: int = 16, batch_size: int = 8,
               fft_sizes: tuple = (256, 2048),
               verbose: bool = True,
               progress: dict | None = None) -> list:
    """
    Train InstrumentProfile end-to-end on WAV segments using batched synthesis.

    progress (optional shared dict for GUI polling):
      progress["epoch"]  — updated each epoch
      progress["loss"]   — current average loss
      progress["cancel"] — set True externally to stop early
    """
    feat  = build_feature_tables(sr=sr, k_max=k_max)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    keys   = sorted(wav_bank.keys())
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        perm = torch.randperm(len(keys)).tolist()
        # Group into batches
        for i in range(0, len(keys), batch_size):
            batch_keys = [keys[perm[j]] for j in range(i, min(i + batch_size, len(keys)))]
            midi_list, vel_list, wav_refs = [], [], []
            for key in batch_keys:
                p = key.split('_')
                midi_list.append(int(p[0][1:]))
                vel_list.append(int(p[1][3:]))
                wav_refs.append(wav_bank[key])

            wav_target = torch.stack(wav_refs)  # [B, T]

            opt.zero_grad()
            synth = synth_batch(model, midi_list, vel_list, feat, seg_len)  # [B, T]
            loss  = multiscale_spectral_loss(synth, wav_target, fft_sizes=fft_sizes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            epoch_loss += float(loss.detach())

        sched.step()
        avg = epoch_loss / math.ceil(len(keys) / batch_size)
        losses.append(avg)

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

        if progress is not None:
            current_lr = sched.get_last_lr()[0]
            progress["epoch"] = epoch
            progress["loss"]  = round(avg, 5)
            progress["lr"]    = float(current_lr)
            # Append log line every 10 epochs
            if epoch % 10 == 0:
                line = f"epoch {epoch:4d}/{epochs}  loss={avg:.5f}  lr={current_lr:.2e}"
                logs = progress.setdefault("log_lines", [])
                logs.append(line)
                if len(logs) > 20:
                    logs.pop(0)
            if progress.get("cancel"):
                break

    return losses


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DDSP instrument profile training")
    ap.add_argument("--wav-dir",  default="C:/SoundBanks/IthacaPlayer/ks-grand")
    ap.add_argument("--out",      default="analysis/params_profile.json")
    ap.add_argument("--model",    default="analysis/profile_ddsp.pt")
    ap.add_argument("--init",     default=None,
                    help="Warm-start from existing profile.pt")
    ap.add_argument("--epochs",   type=int,   default=300)
    ap.add_argument("--hidden",   type=int,   default=64)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--seg",      type=float, default=0.5,
                    help="Segment length in seconds (0.5 recommended)")
    ap.add_argument("--kmax",     type=int,   default=16)
    ap.add_argument("--batch",    type=int,   default=8,
                    help="Notes per gradient step")
    ap.add_argument("--sr",       type=int,   default=44100)
    ap.add_argument("--midi-from", type=int,  default=21)
    ap.add_argument("--midi-to",   type=int,  default=108)
    ap.add_argument("--no-preserve-orig", action="store_true")
    args = ap.parse_args()

    seg_len = int(args.seg * args.sr)

    wav_bank = load_wav_bank(args.wav_dir, sr_target=args.sr, seg_secs=args.seg)
    if not wav_bank:
        print("ERROR: No WAV files found. Check --wav-dir.")
        return

    model = InstrumentProfile(hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())

    if args.init and Path(args.init).exists():
        ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Warm-started from {args.init}")
    else:
        print("Training from random initialisation (use --init for warm-start)")

    print(f"Model parameters: {n_params:,}")
    print(f"Training notes: {len(wav_bank)}  seg={args.seg:.1f}s  "
          f"k_max={args.kmax}  batch={args.batch}")
    print(f"Training {args.epochs} epochs ...")

    train_ddsp(model, wav_bank, epochs=args.epochs, lr=args.lr,
               sr=args.sr, seg_len=seg_len, k_max=args.kmax,
               batch_size=args.batch)

    torch.save({
        "state_dict": model.state_dict(),
        "hidden": args.hidden,
        "eq_freqs": None,
    }, args.model)
    print(f"Model saved -> {args.model}")

    print("Generating full parameter profile ...")
    orig_samples = None
    if not args.no_preserve_orig:
        params_path = Path("analysis/params.json")
        if params_path.exists():
            orig_samples = json.loads(params_path.read_text()).get("samples")
            print(f"  Preserving originals from {params_path}")

    ds = {"eq_freqs": None, "batches": {}}
    profile_samples = generate_profile(
        model, ds,
        midi_from=args.midi_from, midi_to=args.midi_to,
        sr=args.sr, orig_samples=orig_samples,
    )
    n_nn   = sum(1 for s in profile_samples.values() if s.get("_from_profile"))
    n_orig = sum(1 for s in profile_samples.values() if not s.get("_from_profile"))
    print(f"  NN-generated: {n_nn}  |  Preserved originals: {n_orig}")

    out_data = {"samples": profile_samples}
    Path(args.out).write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"Written -> {args.out}")


if __name__ == "__main__":
    main()
