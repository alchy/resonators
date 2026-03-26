"""
models/controller.py
GRU neural controller for the resonator bank.

Input per frame (concatenated):
  f0_enc       – sinusoidal log-F0 embedding       (f0_bins  dims)
  vel_enc      – sinusoidal velocity embedding      (vel_bins dims)
  phase_onehot – envelope phase one-hot             (4 dims)
  log_rms      – log RMS of current frame           (1 dim)
  frame_t      – fractional frame position in clip  (1 dim)

Pre-MLP → GRU → Post-MLP

Output per frame per resonator:
  delta_f    – inharmonicity/tuning correction (±0.5 % relative, via tanh in bank)
  raw_amp    – raw amplitude prediction; bank applies softplus × amp_scale
  raw_gate   – before sigmoid; caller applies sigmoid
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUController(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        cc = cfg['controller']
        rc = cfg['resonators']

        self.N        = rc['n_harmonic'] + rc['n_noise'] + rc['n_transient']
        self.f0_bins  = int(cc['f0_bins'])
        self.vel_bins = int(cc['vel_bins'])

        input_dim = self.f0_bins + self.vel_bins + 4 + 1 + 1
        pre_dim   = int(cc['pre_mlp_dim'])
        gru_h     = int(cc['gru_hidden'])
        gru_l     = int(cc['gru_layers'])
        post_dim  = int(cc['post_mlp_dim'])

        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, pre_dim),
            nn.ReLU(),
            nn.Linear(pre_dim, pre_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size  = pre_dim,
            hidden_size = gru_h,
            num_layers  = gru_l,
            batch_first = True,
            dropout     = 0.1 if gru_l > 1 else 0.0,
        )

        self.post_mlp = nn.Sequential(
            nn.Linear(gru_h, post_dim),
            nn.ReLU(),
            nn.Linear(post_dim, self.N * 3),   # (delta_f, raw_amp, raw_gate) × N
        )

        # Initialise output layer small
        nn.init.normal_(self.post_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.post_mlp[-1].bias)

        # Gate outputs (index 2, 5, 8, …): strong negative bias → gates start ~0.02
        # This forces the model to learn to open gates explicitly rather than
        # having all resonators active from epoch 1.
        with torch.no_grad():
            self.post_mlp[-1].bias[2::3].fill_(-4.0)

            # raw_amp outputs (index 1, 4, 7, …): mild negative bias so that
            # softplus(bias) × amp_scale starts at a small, reasonable level.
            self.post_mlp[-1].bias[1::3].fill_(-3.0)

    # ──────────────────────────────────────────────────────────────────
    # Sinusoidal positional encoding  (fixed, not learnable)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _sinenc(x: torch.Tensor, n_bins: int) -> torch.Tensor:
        """
        x: (B,) scalar in [0, 1]
        Returns (B, n_bins) via alternating sin/cos at exponentially spaced freqs.
        """
        half   = n_bins // 2
        exps   = torch.arange(half, device=x.device).float()
        freqs  = 2.0 ** exps
        angles = 2.0 * math.pi * freqs * x.unsqueeze(1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        f0:           torch.Tensor,   # (B,)
        vel_norm:     torch.Tensor,   # (B,)
        rms_frames:   torch.Tensor,   # (B, T_frames)
        phase_labels: torch.Tensor,   # (B, T_frames) int64
    ):
        """
        Returns:
          control  (B, T_frames, N, 2)  stack of [delta_f, raw_amp]
          gates    (B, T_frames, N)     sigmoid gate ∈ (0, 1)
        """
        B, T = rms_frames.shape
        device = f0.device

        LOG_LO  = math.log(27.5)
        LOG_HI  = math.log(4186.0)
        f0_norm = ((torch.log(f0.clamp(min=1.0)) - LOG_LO) / (LOG_HI - LOG_LO)).clamp(0.0, 1.0)

        f0_enc  = self._sinenc(f0_norm,  self.f0_bins)   # (B, f0_bins)
        vel_enc = self._sinenc(vel_norm, self.vel_bins)  # (B, vel_bins)

        f0_t  = f0_enc.unsqueeze(1).expand(-1, T, -1)
        vel_t = vel_enc.unsqueeze(1).expand(-1, T, -1)

        phase_oh = F.one_hot(phase_labels, num_classes=4).float()   # (B, T, 4)
        log_rms  = torch.log(rms_frames.clamp(min=1e-7)).unsqueeze(-1)  # (B, T, 1)
        frame_t  = (torch.arange(T, device=device).float() / max(T - 1, 1)
                    ).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)       # (B, T, 1)

        feat = torch.cat([f0_t, vel_t, phase_oh, log_rms, frame_t], dim=-1)
        feat = self.pre_mlp(feat)

        gru_out, _ = self.gru(feat)

        out  = self.post_mlp(gru_out)               # (B, T, N*3)
        out  = out.reshape(B, T, self.N, 3)

        delta_f  = out[..., 0]                      # (B, T, N)
        raw_amp  = out[..., 1]                      # (B, T, N)
        gate     = torch.sigmoid(out[..., 2])       # (B, T, N)  ∈ (0, 1)

        control = torch.stack([delta_f, raw_amp], dim=-1)   # (B, T, N, 2)
        return control, gate
