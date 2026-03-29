"""
train_e2e.py
─────────────
End-to-end differentiable piano synthesizer training.

Three-phase curriculum (Simionato 2024 insight):
  Phase 0 — Warm-start (supervised on analytically extracted params)
             Only SetterNN trained; no synthesizer involved.
             Sub-phase 0a: B_net only (F0 anchor, Simionato stage 1).
             Sub-phase 0b: all SetterNN branches.

  Phase 1 — Spectral audio loss
             SetterNN → DiffPianoSynth → audio segment → MRSTFT + L1 + RMS.
             Gradients flow end-to-end through synthesizer to SetterNN.

  Phase 2 — Full loss
             Adds attack-weighted MRSTFT, F0 logMSE, physics regularisation.

Usage:
    python train_e2e.py --config config_e2e.json \\
                        --params analysis/params-salamander.json \\
                        --bank   C:/SoundBanks/IthacaPlayer/salamander \\
                        --out    checkpoints/e2e

    # Resume from checkpoint:
    python train_e2e.py --config config_e2e.json \\
                        --params analysis/params-salamander.json \\
                        --bank   C:/SoundBanks/IthacaPlayer/salamander \\
                        --out    checkpoints/e2e \\
                        --resume checkpoints/e2e/best.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Project imports
from models.setter_nn   import SetterNN, load_params_tensors
from models.diff_synth  import DifferentiablePianoSynth
from losses.piano_loss  import PianoLoss
from data.piano_dataset import PianoDataset, collate_fn


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_log(out_dir: Path) -> None:
    log_path = out_dir / "train_e2e.log"
    class _Tee:
        def __init__(self, *s): self.streams = s
        def write(self, t):
            for s in self.streams: s.write(t)
        def flush(self):
            for s in self.streams: s.flush()
    sys.stdout = _Tee(sys.__stdout__, open(log_path, "w", encoding="utf-8", buffering=1))


# ── Device setup ──────────────────────────────────────────────────────────────

def _get_device(cfg: dict) -> torch.device:
    d = cfg.get('device', 'auto')
    if d == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(d)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save(path: Path, setter: SetterNN, synth: DifferentiablePianoSynth,
          optimizer: torch.optim.Optimizer, epoch: int, phase: int,
          best_loss: float):
    torch.save({
        'setter_state':  setter.state_dict(),
        'synth_state':   synth.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'epoch':         epoch,
        'phase':         phase,
        'best_loss':     best_loss,
    }, path)


def _load(path: Path, setter: SetterNN, synth: DifferentiablePianoSynth,
          optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    setter.load_state_dict(ckpt['setter_state'])
    if 'synth_state' in ckpt and ckpt['synth_state']:
        synth.load_state_dict(ckpt['synth_state'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('phase', 0), ckpt.get('best_loss', 1e9)


# ── Phase 0: warm-start (supervised) ─────────────────────────────────────────

def phase0(
    setter:     SetterNN,
    data:       dict,           # from load_params_tensors()
    cfg:        dict,
    device:     torch.device,
    out_dir:    Path,
    sub_phase0a_epochs: int = 20,   # B_net only
):
    """
    Train SetterNN supervised against analytically extracted params.
    Sub-phase 0a: only B_net trainable (Simionato stage 1: anchor inharmonicity).
    Sub-phase 0b: all branches trainable.
    """
    tc      = cfg['training']
    n_total = tc['phase0_epochs']
    lr      = tc['lr_phase0']

    # Move data to device
    f0  = data['f0'].to(device)
    vel = data['vel_norm'].to(device)
    targets = {k: v.to(device) for k, v in data.items() if k not in ('f0', 'vel_norm')}

    # Sub-phase 0a: B_net only
    print(f"\n{'='*60}")
    print(f"Phase 0a: warm-start B_net only ({sub_phase0a_epochs} epochs)")
    print('='*60)
    opt_a = torch.optim.Adam(setter.B_net.parameters(), lr=lr)
    setter.train()
    setter.to(device)
    for ep in range(1, sub_phase0a_epochs + 1):
        opt_a.zero_grad()
        pred = setter(f0, vel)
        B_p = pred['B'].clamp(1e-7)
        B_t = targets['B'].squeeze(1).clamp(1e-7)
        loss = torch.nn.functional.mse_loss(torch.log(B_p), torch.log(B_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(setter.B_net.parameters(), 1.0)
        opt_a.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  0a epoch {ep:3d}/{sub_phase0a_epochs}  B_loss={loss.item():.6f}")

    # Sub-phase 0b: all branches
    n_0b = n_total - sub_phase0a_epochs
    print(f"\n{'='*60}")
    print(f"Phase 0b: warm-start all branches ({n_0b} epochs)")
    print('='*60)
    opt_b = torch.optim.Adam(setter.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=n_0b, eta_min=lr * 0.05)

    best_loss = 1e9
    for ep in range(1, n_0b + 1):
        setter.train()
        opt_b.zero_grad()

        pred = setter(f0, vel)
        total, ld = setter.warm_start_loss(pred, targets)
        total.backward()
        torch.nn.utils.clip_grad_norm_(setter.parameters(), tc['grad_clip'])
        opt_b.step()
        sched.step()

        if total.item() < best_loss:
            best_loss = total.item()
            _save(out_dir / 'phase0_best.pt', setter,
                  DifferentiablePianoSynth(), opt_b, ep, 0, best_loss)

        if ep % 10 == 0 or ep == 1 or ep == n_0b:
            comp = ' '.join(f'{k}={v:.4f}' for k, v in ld.items() if k != 'total')
            print(f"  0b epoch {ep:3d}/{n_0b}  total={ld['total']:.4f}  {comp}")

    print(f"\nPhase 0 done. Best warm-start loss: {best_loss:.4f}")
    return opt_b   # return optimizer so phase1 can re-use it


# ── Phase 1 & 2: audio loss ───────────────────────────────────────────────────

def phase_audio(
    phase:      int,            # 1 or 2
    setter:     SetterNN,
    synth:      DifferentiablePianoSynth,
    loss_fn:    PianoLoss,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    cfg:        dict,
    device:     torch.device,
    out_dir:    Path,
    resume_epoch: int = 0,
    best_loss:  float = 1e9,
):
    tc  = cfg['training']
    lr  = tc[f'lr_phase{phase}']
    n_epochs = tc[f'phase{phase}_epochs']
    seg_dur  = tc['segment_duration']

    opt  = torch.optim.Adam(setter.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs,
        eta_min=lr * tc.get('lr_min_factor', 0.05)
    )

    # Advance scheduler for resumed epochs
    for _ in range(resume_epoch):
        sched.step()

    print(f"\n{'='*60}")
    print(f"Phase {phase}: audio loss ({'spectral' if phase==1 else 'full'}), "
          f"{n_epochs} epochs")
    print('='*60)

    for ep in range(resume_epoch + 1, n_epochs + 1):
        t0 = time.time()
        setter.train()
        train_loss = 0.0
        n_batches  = 0

        for batch in train_dl:
            f0       = batch['f0'].to(device)
            vel_norm = batch['vel_norm'].to(device)
            audio_gt = batch['audio'].to(device)   # (B, 2, T)
            wf       = batch['width_factor'].to(device)
            params_gt = {k: v.to(device) for k, v in batch['params'].items()}

            # Forward: SetterNN → DiffSynth
            params = setter(f0, vel_norm)
            audio_synth = synth(params, f0, seg_dur, width_factor=wf)

            # Align lengths (synthesizer may produce slightly different length)
            T = min(audio_gt.shape[-1], audio_synth.shape[-1])
            audio_gt    = audio_gt[..., :T]
            audio_synth = audio_synth[..., :T]

            loss, ld = loss_fn(
                audio_synth, audio_gt,
                params=params, f0=f0,
                phase=phase,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(setter.parameters(), tc['grad_clip'])
            opt.step()

            train_loss += ld['total']
            n_batches  += 1

        sched.step()
        train_loss /= max(n_batches, 1)

        # Validation
        val_loss = _validate(setter, synth, loss_fn, val_dl,
                             seg_dur, device, phase)

        dt = time.time() - t0
        print(f"  epoch {ep:4d}/{n_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {dt:.0f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            _save(out_dir / 'best.pt', setter, synth, opt, ep, phase, best_loss)
            print(f"    ✓ saved best (val={best_loss:.4f})")

        if ep % 50 == 0:
            _save(out_dir / f'phase{phase}_epoch{ep:04d}.pt',
                  setter, synth, opt, ep, phase, best_loss)

    return best_loss


def _validate(
    setter:   SetterNN,
    synth:    DifferentiablePianoSynth,
    loss_fn:  PianoLoss,
    val_dl:   DataLoader,
    seg_dur:  float,
    device:   torch.device,
    phase:    int,
) -> float:
    setter.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_dl:
            f0       = batch['f0'].to(device)
            vel_norm = batch['vel_norm'].to(device)
            audio_gt = batch['audio'].to(device)
            wf       = batch['width_factor'].to(device)

            params       = setter(f0, vel_norm)
            audio_synth  = synth(params, f0, seg_dur, width_factor=wf)

            T = min(audio_gt.shape[-1], audio_synth.shape[-1])
            _, ld = loss_fn(
                audio_synth[..., :T], audio_gt[..., :T],
                params=params, f0=f0, phase=phase,
            )
            total += ld['total']
            n += 1
    return total / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='End-to-end piano synthesizer training')
    parser.add_argument('--config',  default='config_e2e.json')
    parser.add_argument('--params',  default='analysis/params-salamander.json')
    parser.add_argument('--bank',    default='C:/SoundBanks/IthacaPlayer/salamander')
    parser.add_argument('--out',     default='checkpoints/e2e')
    parser.add_argument('--resume',  default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--phase',   type=int, default=0,
                        help='Start from this phase (0/1/2). Default 0 = run all.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_log(out_dir)

    device   = _get_device(cfg)
    K        = cfg['n_partials']
    tc       = cfg['training']

    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Params: {args.params}")
    print(f"Bank:   {args.bank}")
    print(f"Output: {args.out}")

    # ── Build models ──────────────────────────────────────────────────
    setter = SetterNN(K=K, hidden=cfg['setter_nn']['hidden_dim']).to(device)
    synth  = DifferentiablePianoSynth(
        sr=cfg['sample_rate'],
        frame_size=cfg['frame_size'],
        noise_order=cfg['diff_synth']['noise_order'],
    ).to(device)
    loss_fn = PianoLoss(cfg).to(device)

    n_params = sum(p.numel() for p in setter.parameters() if p.requires_grad)
    print(f"SetterNN parameters: {n_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = PianoDataset(
        bank_dir=args.bank,
        params_json=args.params,
        segment_duration=tc['segment_duration'],
        target_sr=cfg['sample_rate'],
        K=K,
    )
    n_val   = max(1, int(len(dataset) * tc.get('val_split', 0.1)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=tc['batch_size'],
                          shuffle=True,  collate_fn=collate_fn,
                          num_workers=0, pin_memory=(device.type=='cuda'))
    val_dl   = DataLoader(val_ds,   batch_size=tc['batch_size'],
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=0)
    print(f"Train: {n_train} samples, Val: {n_val} samples")

    # ── Resume ────────────────────────────────────────────────────────
    start_phase = args.phase
    best_loss   = 1e9
    resume_epoch = 0
    dummy_opt   = torch.optim.Adam(setter.parameters())
    if args.resume and Path(args.resume).exists():
        resume_epoch, start_phase, best_loss = _load(
            Path(args.resume), setter, synth, dummy_opt
        )
        print(f"Resumed from {args.resume} (epoch={resume_epoch}, "
              f"phase={start_phase}, best_loss={best_loss:.4f})")

    # ── Run phases ────────────────────────────────────────────────────

    # Phase 0: warm-start
    if start_phase <= 0:
        warmup_data = load_params_tensors(args.params, K, device='cpu')
        opt_0 = phase0(setter, warmup_data, cfg, device, out_dir)

    # Phase 1: spectral audio loss
    if start_phase <= 1:
        best_loss = phase_audio(
            phase=1,
            setter=setter, synth=synth, loss_fn=loss_fn,
            train_dl=train_dl, val_dl=val_dl,
            cfg=cfg, device=device, out_dir=out_dir,
            resume_epoch=(resume_epoch if start_phase == 1 else 0),
            best_loss=best_loss,
        )

    # Phase 2: full loss
    if start_phase <= 2:
        best_loss = phase_audio(
            phase=2,
            setter=setter, synth=synth, loss_fn=loss_fn,
            train_dl=train_dl, val_dl=val_dl,
            cfg=cfg, device=device, out_dir=out_dir,
            resume_epoch=(resume_epoch if start_phase == 2 else 0),
            best_loss=best_loss,
        )

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == '__main__':
    main()
