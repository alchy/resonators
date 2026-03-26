"""
models/envelope_net.py
Tiny MLP: (midi_norm, vel_norm) → (warped RMS curve, ADSR fractions)

Learns instrument-specific envelopes independently of the resonator bank.
The RMS shape uses a power-law warped time axis so that early control points
cover the attack region at fine resolution, while later points are coarser.

ADSR fractions are predicted via softmax over 4 logits, giving the proportion
of total clip duration in each phase (attack / decay / sustain / release).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvelopeNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ec         = cfg['envelope_net']
        self.n_env = int(ec['n_env'])      # warped RMS control points
        self.warp  = float(ec['warp'])     # power-law exponent

        hidden = int(ec['hidden'])
        # output: n_env RMS values  +  4 ADSR fraction logits
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_env + 4),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        midi_norm: torch.Tensor,   # (B,)  in [0, 1]
        vel_norm:  torch.Tensor,   # (B,)  in [0, 1]
    ):
        """
        Returns:
          rms_shape   (B, n_env) — RMS at warped time steps  (≥ 0, softplus)
          adsr_fracs  (B, 4)     — fraction per A/D/S/R phase (sums to 1)
        """
        x          = torch.stack([midi_norm, vel_norm], dim=-1)  # (B, 2)
        out        = self.net(x)                                  # (B, n_env+4)
        rms_shape  = F.softplus(out[:, :self.n_env])              # (B, n_env)
        adsr_fracs = F.softmax(out[:, self.n_env:], dim=-1)       # (B, 4)
        return rms_shape, adsr_fracs

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        midi_norm: float,
        vel_norm:  float,
        T_frames:  int,
    ):
        """
        Predict envelope for a single note.

        Returns:
          rms    (T_frames,)  float32
          phases (T_frames,)  int64   (0=attack, 1=decay, 2=sustain, 3=release)
        """
        device  = next(self.parameters()).device
        midi_t  = torch.tensor([midi_norm], dtype=torch.float32, device=device)
        vel_t   = torch.tensor([vel_norm],  dtype=torch.float32, device=device)

        rms_shape, adsr_fracs = self(midi_t, vel_t)

        # ── RMS: warped grid → uniform T_frames via linear interpolation ──
        shape_np = rms_shape[0].cpu().numpy()                      # (n_env,)
        t_store  = np.power(np.linspace(0.0, 1.0, self.n_env), self.warp)
        t_query  = np.linspace(0.0, 1.0, T_frames)
        rms_out  = np.interp(t_query, t_store, shape_np).astype(np.float32)

        # ── Phases: ADSR fractions → label array ─────────────────────────
        fracs      = adsr_fracs[0].cpu().numpy()                   # (4,) sums to 1
        boundaries = np.cumsum(fracs)[:3]                          # [p_ad, p_ds, p_sr]
        t_norm     = np.arange(T_frames) / max(T_frames - 1, 1)
        phases_out = np.digitize(t_norm, boundaries).astype(np.int64)  # 0–3

        return rms_out, phases_out


# ──────────────────────────────────────────────────────────────────────────────
# Standalone pre-training
# ──────────────────────────────────────────────────────────────────────────────

def train_envelope_net(
    manifest_path: str,
    cfg:           dict,
    device:        torch.device,
) -> EnvelopeNet:
    """
    Train EnvelopeNet on NPZ loudness curves and ADSR phase labels.
    Saves checkpoint to  cfg['checkpoint_dir']/envelope.pt
    Returns the trained model (in eval mode).
    """
    ec           = cfg['envelope_net']
    epochs       = int(ec['epochs'])
    lr           = float(ec['lr'])
    attack_w     = float(ec['attack_weight'])
    warp         = float(ec['warp'])
    n_env        = int(ec['n_env'])
    n_atk        = max(1, n_env // 25)          # ~4 % → attack region

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  EnvelopeNet: loading {len(manifest)} samples ...")

    # Warped time axis shared across all samples
    t_store = np.power(np.linspace(0.0, 1.0, n_env), warp)

    midi_list, vel_list, rms_list, adsr_list = [], [], [], []

    for entry in manifest:
        d         = np.load(entry['npz'])
        rms_raw   = d['rms'].astype(np.float32)             # (n_frames,)
        n_frames  = len(rms_raw)

        # Warp RMS onto n_env control points
        xs      = np.linspace(0.0, 1.0, max(n_frames, 1))
        rms_warp = np.interp(t_store, xs, rms_raw).astype(np.float32)

        # ADSR fractions: order matches PHASE_ATTACK=0, PHASE_SUSTAIN=1, PHASE_DECAY=2, PHASE_RELEASE=3
        pc    = entry['phase_counts']
        total = sum(pc.values()) or 1
        fracs = np.array([
            pc.get('attack',  0),
            pc.get('sustain', 0),
            pc.get('decay',   0),
            pc.get('release', 0),
        ], dtype=np.float32) / total

        midi_list.append(entry['midi_note'] / 127.0)
        vel_list.append(float(entry['vel_norm']))
        rms_list.append(rms_warp)
        adsr_list.append(fracs)

    midi_t  = torch.tensor(midi_list, dtype=torch.float32, device=device)
    vel_t   = torch.tensor(vel_list,  dtype=torch.float32, device=device)
    rms_t   = torch.tensor(np.stack(rms_list),  dtype=torch.float32, device=device)
    adsr_t  = torch.tensor(np.stack(adsr_list), dtype=torch.float32, device=device)

    # Loss weight: extra weight on first n_atk warped points (attack region)
    loss_w         = torch.ones(n_env, dtype=torch.float32, device=device)
    loss_w[:n_atk] = attack_w

    model = EnvelopeNet(cfg).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    print(f"  EnvelopeNet: training {epochs} epochs  n_env={n_env}  warp={warp}  "
          f"attack_weight={attack_w}  device={device}")

    for epoch in range(1, epochs + 1):
        pred_rms, pred_adsr = model(midi_t, vel_t)

        # Weighted MSE on warped RMS
        err_rms   = (pred_rms - rms_t) ** 2               # (B, n_env)
        loss_rms  = (err_rms * loss_w).mean()

        # KL-divergence on ADSR fractions (target treated as distribution)
        loss_adsr = F.kl_div(
            torch.log(pred_adsr.clamp(min=1e-8)),
            adsr_t.clamp(min=1e-8),
            reduction='batchmean',
        )

        loss = loss_rms + loss_adsr
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 100 == 0 or epoch == epochs:
            print(f"    ep {epoch:4d}/{epochs}  loss={loss.item():.5f}"
                  f"  (rms={loss_rms.item():.5f}  adsr={loss_adsr.item():.5f})")

    ckpt_dir  = cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, 'envelope.pt')
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg}, save_path)
    print(f"  EnvelopeNet saved -> {save_path}")

    model.eval()
    return model
