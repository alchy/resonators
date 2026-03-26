"""
train.py
EGRB training loop.

Curriculum
──────────
Phase 1 (0 … phase2_start-1):
  8 harmonic resonators, no noise/transients.
  Clip length 32 frames (~170 ms) — only attack region.
  Model learns basic harmonic structure without frequency drift.

Phase 2 (phase2_start … phase3_start-1):
  24 harmonic + 4 noise resonators.
  Clip length 128 frames (~680 ms).
  Envelope gating active, energy + kinetics losses added.

Phase 3 (phase3_start … epochs):
  Full 48 + 8 + 8 resonators.
  Clip length 375 frames (2 s).
  Sparsity loss active. Full composite loss.

Usage:
    python train.py
    python train.py --config config.json --resume checkpoints/last.pt
    python train.py --skip-envelope
"""

import os
import sys
import json
import math
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset          import EGRBDataset
from models.controller     import GRUController
from models.resonator_bank import ResonatorBank
from models.envelope_net   import EnvelopeNet, train_envelope_net
from losses.losses         import EGRBLoss


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    pref = cfg.get('device', 'auto')
    if pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(pref)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, eta_min: float = 0.05):
        self.opt        = optimizer
        self.warmup     = warmup_steps
        self.total      = total_steps
        self.eta_min    = eta_min
        self.base_lrs   = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        s = self.step_count
        for i, pg in enumerate(self.opt.param_groups):
            if s < self.warmup:
                pg['lr'] = self.base_lrs[i] * s / max(self.warmup, 1)
            else:
                progress = (s - self.warmup) / max(self.total - self.warmup, 1)
                cos_val  = 0.5 * (1 + math.cos(math.pi * progress))
                pg['lr'] = self.base_lrs[i] * (self.eta_min + (1 - self.eta_min) * cos_val)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.clone().detach()
            else:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum
# ──────────────────────────────────────────────────────────────────────────────

def _phase(epoch: int, cfg: dict) -> int:
    """Returns 1, 2, or 3."""
    p2 = cfg['curriculum']['phase2_start']
    p3 = cfg['curriculum']['phase3_start']
    if epoch < p2:
        return 1
    if epoch < p3:
        return 2
    return 3


def apply_curriculum(
    epoch:    int,
    cfg:      dict,
    train_ds: EGRBDataset,
    bank:     ResonatorBank,
    loss_fn:  EGRBLoss,
):
    """
    Called at the start of each epoch.
    Updates: resonator active counts, clip_frames, loss weights.
    """
    phase   = _phase(epoch, cfg)
    rc      = cfg['resonator_curriculum']
    tc      = cfg['training']

    # ── Resonator counts ─────────────────────────────────────────────
    n_h = rc[f'p{phase}_n_harmonic']
    n_n = rc[f'p{phase}_n_noise']
    n_t = rc[f'p{phase}_n_transient']
    bank.set_active_resonators(n_h, n_n, n_t)

    # ── Clip length ───────────────────────────────────────────────────
    clip_key = f'clip_frames_p{phase}'
    new_clip = int(tc.get(clip_key, tc['clip_frames']))
    if train_ds.clip_frames != new_clip:
        train_ds.clip_frames = new_clip

    # ── Loss weights ─────────────────────────────────────────────────
    if phase == 1:
        loss_fn.w_kin    = 0.0
        loss_fn.w_eng    = 0.0
        loss_fn.w_sparse = 0.0
    elif phase == 2:
        loss_fn.w_kin    = float(cfg['loss']['w_kin'])
        loss_fn.w_eng    = float(cfg['loss']['w_eng'])
        loss_fn.w_sparse = 0.0
    else:
        loss_fn.w_kin    = float(cfg['loss']['w_kin'])
        loss_fn.w_eng    = float(cfg['loss']['w_eng'])
        loss_fn.w_sparse = float(cfg['loss']['w_sparsity'])

    return phase, n_h, n_n, n_t


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval one epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    controller: GRUController,
    bank:       ResonatorBank,
    loss_fn:    EGRBLoss,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer | None,
    scheduler:  WarmupCosineScheduler | None,
    ema:        EMA | None,
    device:     torch.device,
    cfg:        dict,
    epoch:      int,
    is_train:   bool,
    env_net:    EnvelopeNet | None = None,
) -> dict:
    controller.train(is_train)
    bank.train(is_train)

    grad_clip = float(cfg['training']['grad_clip'])
    max_steps = int(cfg['training']['steps_per_epoch'])
    env_mix   = float(cfg['envelope_net']['env_mix'])

    totals: dict = {}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            if is_train and step >= max_steps:
                break

            audio    = batch['audio'].to(device)     # (B, 2, T)
            rms      = batch['rms'].to(device)        # (B, T_frames)
            phases   = batch['phases'].to(device)     # (B, T_frames) int64
            f0       = batch['f0'].to(device)         # (B,)
            vel_norm = batch['vel_norm'].to(device)   # (B,)

            # ── Coupled training: replace envelope with EnvelopeNet ───
            if is_train and env_net is not None and random.random() < env_mix:
                T_frames  = rms.shape[1]
                midi_norm = (69.0 + 12.0 * torch.log2(
                    f0.clamp(min=1.0) / 440.0
                )).clamp(0.0, 127.0) / 127.0
                rms_list, ph_list = [], []
                for b in range(f0.shape[0]):
                    r, p = env_net.predict(
                        float(midi_norm[b]), float(vel_norm[b]), T_frames
                    )
                    rms_list.append(torch.from_numpy(r))
                    ph_list.append(torch.from_numpy(p))
                rms    = torch.stack(rms_list).to(device)
                phases = torch.stack(ph_list).to(device)

            # ── Forward ──────────────────────────────────────────────
            control, gates = controller(f0, vel_norm, rms, phases)
            pred           = bank(f0, vel_norm, control, gates, phases)

            T_out      = min(pred.shape[-1], audio.shape[-1])
            pred_cut   = pred[:, :, :T_out]
            target_cut = audio[:, :, :T_out]

            T_frames_used = T_out // cfg['frame_size']
            phases_cut    = phases[:, :T_frames_used]
            gates_cut     = gates[:, :T_frames_used, :]

            total, ld = loss_fn(pred_cut, target_cut, phases_cut, gates_cut)

            if is_train:
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(
                    list(controller.parameters()) + list(bank.parameters()),
                    grad_clip,
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()
                if ema:
                    ema.update(controller)
                    ema.update(bank)

            for k, v in ld.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        default='config.json')
    parser.add_argument('--resume',        default=None)
    parser.add_argument('--skip-envelope', action='store_true',
                        help='Skip EnvelopeNet pre-training (use existing checkpoint)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device   = get_device(cfg)
    ckpt_dir = cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Device: {device}")

    # ── Manifest ────────────────────────────────────────────────────
    manifest_path = os.path.join(
        os.path.dirname(cfg['data_dir']) or '.', 'manifest.json'
    )
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        print("Run:  python data/prepare.py")
        sys.exit(1)

    # ── EnvelopeNet pre-training ─────────────────────────────────────
    env_ckpt = os.path.join(ckpt_dir, 'envelope.pt')
    if args.skip_envelope and os.path.exists(env_ckpt):
        print(f"Loading existing EnvelopeNet from {env_ckpt}")
        env_net = EnvelopeNet(cfg).to(device)
        env_net.load_state_dict(torch.load(env_ckpt, map_location=device)['state_dict'])
        env_net.eval()
    else:
        print("Pre-training EnvelopeNet ...")
        env_net = train_envelope_net(manifest_path, cfg, device)

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds = EGRBDataset(manifest_path, cfg, split='train')
    val_ds   = EGRBDataset(manifest_path, cfg, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'],
        shuffle=True, num_workers=0,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['training']['batch_size'],
        shuffle=False, num_workers=0,
    )

    # ── Models ───────────────────────────────────────────────────────
    controller = GRUController(cfg).to(device)
    bank       = ResonatorBank(cfg).to(device)
    loss_fn    = EGRBLoss(cfg).to(device)

    n_params = (sum(p.numel() for p in controller.parameters()) +
                sum(p.numel() for p in bank.parameters()))
    print(f"Parameters: {n_params:,}  "
          f"(controller={sum(p.numel() for p in controller.parameters()):,}  "
          f"bank={sum(p.numel() for p in bank.parameters()):,})")

    # ── Optimizer ────────────────────────────────────────────────────
    lr        = float(cfg['training']['learning_rate'])
    epochs    = int(cfg['training']['epochs'])
    spe       = int(cfg['training']['steps_per_epoch'])
    total_stp = epochs * spe
    warmup    = int(cfg['training']['warmup_steps'])

    optimizer = torch.optim.AdamW(
        list(controller.parameters()) + list(bank.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=0.01,
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_stp)
    ema       = EMA(controller, cfg['training']['ema_decay'])

    start_epoch = 0
    best_val    = float('inf')

    # ── Resume ───────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        controller.load_state_dict(ckpt['controller'])
        bank.load_state_dict(ckpt['bank'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch - 1}  (best_val={best_val:.5f})")

    # ── Training loop ────────────────────────────────────────────────
    log_path  = os.path.join(ckpt_dir, 'train.log')
    prev_phase = 0

    with open(log_path, 'a') as logf:
        for epoch in range(start_epoch, epochs):

            phase, n_h, n_n, n_t = apply_curriculum(
                epoch, cfg, train_ds, bank, loss_fn
            )

            if phase != prev_phase:
                print(f"\n  == Curriculum phase {phase}  "
                      f"resonators={n_h}h+{n_n}n+{n_t}t  "
                      f"clip={train_ds.clip_frames}f ==\n")
                prev_phase = phase

            t0 = time.time()
            train_ld = run_epoch(
                controller, bank, loss_fn,
                train_loader, optimizer, scheduler, ema,
                device, cfg, epoch, is_train=True, env_net=env_net,
            )
            val_ld = run_epoch(
                controller, bank, loss_fn,
                val_loader, None, None, None,
                device, cfg, epoch, is_train=False,
            )
            elapsed = time.time() - t0

            phase_label = f'phase{phase}'
            train_str   = '  '.join(f"{k}={v:.4f}" for k, v in train_ld.items())
            val_str     = '  '.join(f"{k}={v:.4f}" for k, v in val_ld.items())
            line = (f"Epoch {epoch:4d}/{epochs}  [{phase_label}]  "
                    f"train: {train_str}  ||  val: {val_str}  ({elapsed:.0f}s)")
            print(line)
            logf.write(line + '\n')
            logf.flush()

            val_total = val_ld.get('total', float('inf'))

            if val_total < best_val:
                best_val = val_total
                is_best = True
            else:
                is_best = False

            ckpt_data = {
                'epoch':      epoch,
                'controller': controller.state_dict(),
                'bank':       bank.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'best_val':   best_val,
                'cfg':        cfg,
            }

            torch.save(ckpt_data, os.path.join(ckpt_dir, 'last.pt'))

            if is_best:
                torch.save(ckpt_data, os.path.join(ckpt_dir, 'best.pt'))
                print(f"  ** new best  val_total={best_val:.5f}")

    print("Training complete.")


if __name__ == '__main__':
    main()
