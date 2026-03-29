"""
models/setter_nn.py
───────────────────
SetterNN: maps (midi_norm, vel_norm) → per-note physical synthesizer parameters.

Architecture (inspired by Simionato 2024, extended):
  B_net:     Linear(1→64→1)  + Softplus   — inharmonicity (f0-only, velocity-independent)
  beat_net:  Linear(2→64→K*2)+ [Softplus, Sigmoid]  — beat_hz, beat_depth
  decay_net: Linear(2→256→256→K*3) — tau1, tau2, a1 (bi-exponential decay)
  amp_net:   Linear(2→256→256→K)   — partial amplitudes A0
  noise_net: Linear(2→64→4)        — noise params [attack_tau, floor_rms, centroid, slope]

Input normalisation:
  f0_norm  = (log(f0) - log(f0_A0)) / (log(f0_C8) - log(f0_A0))  ∈ [0, 1]
  vel_norm = vel_idx / 7.0                                          ∈ [0, 1]

Warm-start: init_from_params(params_json) trains SetterNN supervised against
  analytically extracted params (extract-params.py output). This avoids cold-start
  divergence when switching to audio loss.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── MIDI / frequency constants ────────────────────────────────────────────────

MIDI_LO  = 21      # A0
MIDI_HI  = 108     # C8
F0_LO    = 27.5    # A0 Hz
F0_HI    = 4186.0  # C8 Hz
LOG_F0_LO = math.log(F0_LO)
LOG_F0_HI = math.log(F0_HI)

# Physical bounds (used for output clamping and physics regularisation)
B_MIN,   B_MAX   = 1e-6,  0.4
TAU1_MIN, TAU1_MAX = 0.005, 5.0
TAU2_MIN, TAU2_MAX = 0.02,  80.0
BEAT_MIN, BEAT_MAX = 0.05,  10.0


def f0_to_norm(f0: torch.Tensor) -> torch.Tensor:
    """Log-linear normalisation: f0 [Hz] → [0, 1]."""
    return (torch.log(f0.clamp(F0_LO, F0_HI)) - LOG_F0_LO) / (LOG_F0_HI - LOG_F0_LO)


def midi_to_f0(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * torch.pow(2.0, (midi.float() - 69.0) / 12.0)


# ── Building block ────────────────────────────────────────────────────────────

def _mlp(in_dim: int, *dims: int) -> nn.Sequential:
    """Build MLP: Linear(in→d0) ReLU Linear(d0→d1) ReLU … Linear(dn-1→dn)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for i, d in enumerate(dims):
        layers.append(nn.Linear(prev, d))
        if i < len(dims) - 1:
            layers.append(nn.ReLU())
        prev = d
    return nn.Sequential(*layers)


# ── SetterNN ──────────────────────────────────────────────────────────────────

class SetterNN(nn.Module):
    """
    Per-note physical parameter predictor.

    Args:
        K         : number of partials (default 64)
        hidden    : MLP hidden dimension (default 256)

    Forward:
        f0        : (B,) Hz  — fundamental frequency
        vel_norm  : (B,) [0,1] — normalised velocity

    Returns dict of tensors (all shape (B, K) or (B,) as noted):
        B           : (B,)    inharmonicity coefficient
        f0_offset   : (B,)    tuning offset in cents (tanh * 50)
        A0          : (B, K)  partial amplitudes (normalised, sum=K)
        tau1        : (B, K)  fast decay time [s]
        tau2        : (B, K)  slow decay time [s]
        a1          : (B, K)  fast/slow mixing ratio ∈ (0, 1)
        beat_hz     : (B, K)  string detuning [Hz]
        beat_depth  : (B, K)  beating depth ∈ (0, 1)
        noise       : (B, 4)  [attack_tau_s, floor_rms, centroid_norm, slope_db_oct]
    """

    def __init__(self, K: int = 64, hidden: int = 256):
        super().__init__()
        self.K = K
        H = hidden

        # B: only f0_norm as input (velocity-independent, Simionato insight)
        self.B_net = _mlp(1, H // 4, H // 4, 1)

        # f0_offset: tuning correction (cents), (f0, vel) dependent
        self.tune_net = _mlp(2, H // 4, 1)

        # Amplitude spectrum (f0, vel) → K amplitudes
        self.amp_net = _mlp(2, H, H, K)

        # Decay (f0, vel) → K * 3  (log_tau1, log_tau2, logit_a1)
        self.decay_net = _mlp(2, H, H, K * 3)

        # Beating: (f0, vel) → K * 2  (log_beat_hz, logit_beat_depth)
        self.beat_net = _mlp(2, H // 2, H // 2, K * 2)

        # Noise: (f0, vel) → 4
        self.noise_net = _mlp(2, H // 4, H // 4, 4)

        self._init_weights()

    def _init_weights(self):
        """Sensible zero-point initialisation before warm-start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, f0: torch.Tensor, vel_norm: torch.Tensor) -> dict:
        B_batch = f0.shape[0]
        K = self.K

        f0n = f0_to_norm(f0).unsqueeze(1)      # (B, 1)
        vn  = vel_norm.unsqueeze(1)             # (B, 1)
        x2  = torch.cat([f0n, vn], dim=1)      # (B, 2)

        # ── B: inharmonicity ─────────────────────────────────────────
        B_raw = self.B_net(f0n).squeeze(1)      # (B,)
        B = F.softplus(B_raw) * 0.02            # scale into typical range [0, 0.4]

        # ── f0 offset (cents) ─────────────────────────────────────────
        f0_offset = torch.tanh(self.tune_net(x2).squeeze(1)) * 50.0  # (B,) ±50 cents

        # ── Amplitude spectrum ────────────────────────────────────────
        amp_raw = self.amp_net(x2)              # (B, K)
        # Softmax → normalised weights, scaled so mean ≈ 1.0
        A0 = F.softmax(amp_raw, dim=1) * K      # (B, K)

        # ── Decay params ──────────────────────────────────────────────
        decay_raw = self.decay_net(x2)          # (B, K*3)
        lt1 = decay_raw[:, :K]                  # log_tau1
        lt2 = decay_raw[:, K:2*K]              # log_tau2
        la1 = decay_raw[:, 2*K:3*K]            # logit_a1

        # tau1 physically shorter (attack), tau2 longer (sustain)
        # Initialise towards tau1~0.1s, tau2~3s via offset in softplus
        tau1 = F.softplus(lt1 - 1.0).clamp(TAU1_MIN, TAU1_MAX)   # (B, K)
        tau2 = F.softplus(lt2 + 1.0).clamp(TAU2_MIN, TAU2_MAX)   # (B, K)
        a1   = torch.sigmoid(la1)                                   # (B, K)

        # ── Beating ───────────────────────────────────────────────────
        beat_raw = self.beat_net(x2)            # (B, K*2)
        lbhz  = beat_raw[:, :K]
        lbdep = beat_raw[:, K:]

        beat_hz    = F.softplus(lbhz - 1.0).clamp(BEAT_MIN, BEAT_MAX)  # (B, K)
        beat_depth = torch.sigmoid(lbdep) * 0.5                          # (B, K) ∈ (0, 0.5)

        # ── Noise ─────────────────────────────────────────────────────
        noise_raw = self.noise_net(x2)          # (B, 4)
        # [attack_tau_s, floor_rms, centroid_norm, slope_db_oct]
        attack_tau   = F.softplus(noise_raw[:, 0]).clamp(0.002, 0.5)
        floor_rms    = torch.sigmoid(noise_raw[:, 1]) * 0.1
        centroid_norm = torch.sigmoid(noise_raw[:, 2])   # 0=low, 1=high
        slope_db_oct = -F.softplus(noise_raw[:, 3]) * 6  # negative slope (LPF)
        noise = torch.stack([attack_tau, floor_rms, centroid_norm, slope_db_oct], dim=1)

        return {
            'B':           B,
            'f0_offset':   f0_offset,
            'A0':          A0,
            'tau1':        tau1,
            'tau2':        tau2,
            'a1':          a1,
            'beat_hz':     beat_hz,
            'beat_depth':  beat_depth,
            'noise':       noise,
        }

    # ── Warm-start ────────────────────────────────────────────────────────────

    def warm_start_loss(
        self,
        params_pred: dict,
        params_target: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Supervised loss between predicted and analytically extracted params.
        Each component normalised to roughly unit scale.

        Returns (total_loss, component_dict).
        """
        losses: dict[str, torch.Tensor] = {}

        # B: MSE in log space (prevents large values dominating)
        B_p = params_pred['B'].clamp(1e-7)
        B_t = params_target['B'].clamp(1e-7)
        losses['B'] = F.mse_loss(torch.log(B_p), torch.log(B_t))

        # A0: MSE on log-amplitude (relative shape matters, not absolute scale)
        A0_p = params_pred['A0'].clamp(1e-6)
        A0_t = params_target['A0'].clamp(1e-6)
        losses['A0'] = F.mse_loss(torch.log(A0_p), torch.log(A0_t))

        # tau1, tau2: MSE in log space
        losses['tau1'] = F.mse_loss(
            torch.log(params_pred['tau1'].clamp(1e-4)),
            torch.log(params_target['tau1'].clamp(1e-4))
        )
        losses['tau2'] = F.mse_loss(
            torch.log(params_pred['tau2'].clamp(1e-4)),
            torch.log(params_target['tau2'].clamp(1e-4))
        )

        # a1: MSE (already in [0,1])
        losses['a1'] = F.mse_loss(params_pred['a1'], params_target['a1'])

        # beat_hz: MSE in log space
        losses['beat_hz'] = F.mse_loss(
            torch.log(params_pred['beat_hz'].clamp(1e-3)),
            torch.log(params_target['beat_hz'].clamp(1e-3))
        )

        # beat_depth: MSE
        losses['beat_depth'] = F.mse_loss(
            params_pred['beat_depth'], params_target['beat_depth']
        )

        # Weighted sum — B and tau are most critical for pitch / decay accuracy
        weights = {
            'B':          2.0,
            'A0':         1.0,
            'tau1':       1.5,
            'tau2':       1.5,
            'a1':         0.5,
            'beat_hz':    1.0,
            'beat_depth': 0.5,
        }
        total = sum(weights[k] * v for k, v in losses.items())
        losses['total'] = total
        return total, {k: v.item() for k, v in losses.items()}


# ── Params JSON → tensor dataset helper ──────────────────────────────────────

def load_params_tensors(params_json: str, K: int, device: str = 'cpu') -> dict:
    """
    Load params.json and return dict of tensors ready for warm-start training.

    Returns {
        'f0':         (N,)    Hz
        'vel_norm':   (N,)    [0,1]
        'B':          (N,)
        'A0':         (N, K)
        'tau1':       (N, K)
        'tau2':       (N, K)
        'a1':         (N, K)
        'beat_hz':    (N, K)
        'beat_depth': (N, K)
    }
    """
    with open(params_json, 'r') as f:
        data = json.load(f)
    samples = data['samples']

    rows_f0, rows_vel = [], []
    rows_B    = []
    rows_A0   = []
    rows_tau1, rows_tau2, rows_a1 = [], [], []
    rows_beat_hz, rows_beat_depth = [], []

    for key, s in samples.items():
        midi = int(s['midi'])
        vel  = int(s['vel'])
        f0   = float(s.get('f0_fitted_hz') or s.get('f0_nominal_hz', 440.0))
        B    = float(s.get('B', 1e-4))

        partials = s.get('partials', [])

        A0_vec   = [0.0] * K
        tau1_vec = [0.1] * K
        tau2_vec = [3.0] * K
        a1_vec   = [0.25] * K
        bHz_vec  = [0.3] * K
        bDep_vec = [0.05] * K

        def _f(v, default): return float(v) if v is not None else default

        for p in partials:
            k_idx = int(p['k']) - 1
            if k_idx >= K:
                continue
            A0_vec[k_idx]   = max(_f(p.get('A0'),          0.0),  1e-6)
            tau1_vec[k_idx] = max(_f(p.get('tau1'),        0.1),  TAU1_MIN)
            tau2_vec[k_idx] = max(_f(p.get('tau2'),        3.0),  TAU2_MIN)
            a1_vec[k_idx]   = max(0.01, min(0.99, _f(p.get('a1'), 0.25)))
            bHz_vec[k_idx]  = max(_f(p.get('beat_hz'),     0.3),  BEAT_MIN)
            bDep_vec[k_idx] = max(0.001, min(0.499, _f(p.get('beat_depth'), 0.05)))

        # Normalize A0 so mean = 1.0 (SetterNN output convention)
        A0_t = torch.tensor(A0_vec, dtype=torch.float32)
        if A0_t.sum() > 0:
            A0_t = A0_t / (A0_t.mean() + 1e-8)

        rows_f0.append(f0)
        rows_vel.append(vel / 7.0)
        rows_B.append(B)
        rows_A0.append(A0_t.tolist())
        rows_tau1.append(tau1_vec)
        rows_tau2.append(tau2_vec)
        rows_a1.append(a1_vec)
        rows_beat_hz.append(bHz_vec)
        rows_beat_depth.append(bDep_vec)

    def t(lst): return torch.tensor(lst, dtype=torch.float32, device=device)

    return {
        'f0':         t(rows_f0),
        'vel_norm':   t(rows_vel),
        'B':          t(rows_B),
        'A0':         t(rows_A0),
        'tau1':       t(rows_tau1),
        'tau2':       t(rows_tau2),
        'a1':         t(rows_a1),
        'beat_hz':    t(rows_beat_hz),
        'beat_depth': t(rows_beat_depth),
    }
