"""
train.py
EGRB training loop.

Curriculum
──────────
Phase 1 (0 … phase2_start-1):
  Full control density, MRSTFT + L1 only.
  Resonator bank learns basic harmonic structure.

Phase 2 (phase2_start … phase3_start-1):
  Envelope gating active, energy loss + kinetics added.
  Model learns to respect A/S/D/R density constraints.

Phase 3 (phase3_start … epochs):
  Sparsity loss enabled. Gate temperature annealing pushes
  gates toward binary. Full composite loss.

Usage:
    python train.py
    python train.py --config config.json --resume checkpoints/last.pt
"""

import os
import sys
import json
import math
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset      import EGRBDataset
from models.controller import GRUController
from models.resonator_bank import ResonatorBank
from losses.losses     import EGRBLoss


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
        self.opt          = optimizer
        self.warmup       = warmup_steps
        self.total        = total_steps
        self.eta_min      = eta_min
        self.base_lrs     = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count   = 0

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
# Curriculum helpers
# ──────────────────────────────────────────────────────────────────────────────

def curriculum_loss_weights(epoch: int, cfg: dict) -> dict:
    """Returns loss weight overrides for current epoch."""
    p2 = cfg['curriculum']['phase2_start']
    p3 = cfg['curriculum']['phase3_start']

    if epoch < p2:
        return {'w_kin': 0.0, 'w_eng': 0.0, 'w_sparsity': 0.0}
    if epoch < p3:
        return {'w_sparsity': 0.0}
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval one epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    controller:  GRUController,
    bank:        ResonatorBank,
    loss_fn:     EGRBLoss,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer | None,
    scheduler:   WarmupCosineScheduler | None,
    ema:         EMA | None,
    device:      torch.device,
    cfg:         dict,
    epoch:       int,
    is_train:    bool,
) -> dict:
    controller.train(is_train)
    bank.train(is_train)

    grad_clip = float(cfg['training']['grad_clip'])
    max_steps = int(cfg['training']['steps_per_epoch'])

    totals: dict = {}
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            if is_train and step >= max_steps:
                break

            audio        = batch['audio'].to(device)        # (B, 2, T)
            rms          = batch['rms'].to(device)           # (B, T_frames)
            phases       = batch['phases'].to(device)        # (B, T_frames) int64
            f0           = batch['f0'].to(device)            # (B,)
            vel_norm     = batch['vel_norm'].to(device)      # (B,)

            # Controller
            control, gates = controller(f0, vel_norm, rms, phases)
            # control: (B, T_frames, N, 2)
            # gates:   (B, T_frames, N)

            # Resonator bank synthesis
            pred = bank(f0, vel_norm, control, gates, phases)
            # pred: (B, 2, T_samples)  — may differ from audio length
            T_out = min(pred.shape[-1], audio.shape[-1])
            pred_cut   = pred[:, :, :T_out]
            target_cut = audio[:, :, :T_out]

            # Trim phase_labels to match frame count of pred_cut
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
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device   = get_device(cfg)
    ckpt_dir = cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────
    manifest_path = os.path.join(
        os.path.dirname(cfg['data_dir']) or '.', 'manifest.json'
    )
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        print("Run:  python data/prepare.py")
        sys.exit(1)

    train_ds = EGRBDataset(manifest_path, cfg, split='train')
    val_ds   = EGRBDataset(manifest_path, cfg, split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg['training']['batch_size'],
        shuffle     = True,
        num_workers = 0,
        pin_memory  = device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg['training']['batch_size'],
        shuffle     = False,
        num_workers = 0,
    )

    # ── Models ─────────────────────────────────────────────────────────
    controller = GRUController(cfg).to(device)
    bank       = ResonatorBank(cfg).to(device)
    loss_fn    = EGRBLoss(cfg).to(device)

    n_params = (
        sum(p.numel() for p in controller.parameters()) +
        sum(p.numel() for p in bank.parameters())
    )
    print(f"Parameters: {n_params:,}  "
          f"(controller={sum(p.numel() for p in controller.parameters()):,}  "
          f"bank={sum(p.numel() for p in bank.parameters()):,})")

    # ── Optimizer ──────────────────────────────────────────────────────
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

    # ── Resume ─────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        controller.load_state_dict(ckpt['controller'])
        bank.load_state_dict(ckpt['bank'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch - 1}  (best_val={best_val:.5f})")

    # ── Training loop ──────────────────────────────────────────────────
    log_path = os.path.join(ckpt_dir, 'train.log')
    with open(log_path, 'a') as logf:
        for epoch in range(start_epoch, epochs):
            # Apply curriculum weight overrides to loss_fn
            ow = curriculum_loss_weights(epoch, cfg)
            for k, v in ow.items():
                if hasattr(loss_fn, k.replace('w_', 'w_').lstrip('w_')):
                    attr = 'w_' + k.lstrip('w_')
                    if hasattr(loss_fn, attr):
                        setattr(loss_fn, attr, v)

            t0 = time.time()
            train_ld = run_epoch(
                controller, bank, loss_fn,
                train_loader, optimizer, scheduler, ema,
                device, cfg, epoch, is_train=True,
            )
            val_ld = run_epoch(
                controller, bank, loss_fn,
                val_loader, None, None, None,
                device, cfg, epoch, is_train=False,
            )
            elapsed = time.time() - t0

            # Determine curriculum phase label
            p2 = cfg['curriculum']['phase2_start']
            p3 = cfg['curriculum']['phase3_start']
            phase_label = (
                'phase1' if epoch < p2 else
                'phase2' if epoch < p3 else 'phase3'
            )

            # Log
            train_str = '  '.join(f"{k}={v:.4f}" for k, v in train_ld.items())
            val_str   = '  '.join(f"{k}={v:.4f}" for k, v in val_ld.items())
            line = (
                f"Epoch {epoch:4d}/{epochs}  [{phase_label}]  "
                f"train: {train_str}  ||  val: {val_str}  "
                f"({elapsed:.0f}s)"
            )
            print(line)
            logf.write(line + '\n')
            logf.flush()

            # Checkpoints
            val_total = val_ld.get('total', float('inf'))
            ckpt_data = {
                'epoch':      epoch,
                'controller': controller.state_dict(),
                'bank':       bank.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'best_val':   best_val,
                'cfg':        cfg,
            }

            torch.save(ckpt_data, os.path.join(ckpt_dir, 'last.pt'))

            if val_total < best_val:
                best_val = val_total
                torch.save(ckpt_data, os.path.join(ckpt_dir, 'best.pt'))
                print(f"  ** new best  val_total={best_val:.5f}")

    print("Training complete.")


if __name__ == '__main__':
    main()
