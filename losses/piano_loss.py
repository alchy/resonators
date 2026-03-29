"""
losses/piano_loss.py
─────────────────────
PianoLoss: composite audio loss for end-to-end piano synthesizer training.

Components (phase-gated — enabled progressively through training):
  mrstft   Multi-resolution STFT (spectral convergence + log-mag L1)
           FFT sizes [256, 1024, 4096, 16384]  (bass→transient coverage)
  l1       Time-domain L1
  attack   Attack-weighted MRSTFT (5× emphasis on rising loudness)
           Technique from ddsp-base (alchy/ddsp-base)
  f0       Log-MSE on predicted first partial vs. target f0
           (Simionato 2024 Eq.6 — anchors harmonic structure)
  rms      Per-frame RMS envelope MSE (Simionato 2024)
  physics  Soft physical plausibility constraints on SetterNN params

Phase usage:
  phase 1 (spectral):  mrstft + l1 + rms
  phase 2 (full):      all components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PianoLoss(nn.Module):
    """
    Args (cfg['loss'] dict):
        w_mrstft    float  weight for MRSTFT loss              (default 1.0)
        w_l1        float  weight for time-domain L1           (default 0.2)
        w_attack    float  weight for attack-weighted MRSTFT   (default 0.3)
        w_f0        float  weight for F0 logMSE                (default 0.05)
        w_rms       float  weight for RMS envelope             (default 0.05)
        w_physics   float  weight for physics regularisation   (default 0.02)
        mrstft_ffts list   FFT sizes for MRSTFT                (default [256,1024,4096,16384])
        attack_alpha float peak multiplier for attack weight    (default 4.0)
        attack_sigma float Gaussian smoothing frames            (default 2.0)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg.get('loss', {})
        self.w_mrstft  = float(lc.get('w_mrstft',  1.0))
        self.w_l1      = float(lc.get('w_l1',      0.2))
        self.w_attack  = float(lc.get('w_attack',  0.3))
        self.w_f0      = float(lc.get('w_f0',      0.05))
        self.w_rms     = float(lc.get('w_rms',     0.05))
        self.w_physics = float(lc.get('w_physics', 0.02))
        self.alpha     = float(lc.get('attack_alpha', 4.0))
        self.sigma     = float(lc.get('attack_sigma', 2.0))

        raw_ffts = lc.get('mrstft_ffts', [256, 1024, 4096, 16384])
        # (n_fft, hop, win_length) — hop = n_fft/4, win = n_fft
        self.mrstft_cfgs = [(n, n // 4, n) for n in raw_ffts]

        self.sr = int(cfg.get('sample_rate', 48000))

    # ── MRSTFT ────────────────────────────────────────────────────────────────

    @staticmethod
    def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        """(B*C, T) → magnitude spectrogram (B*C, F, frames)."""
        window = torch.hann_window(win, device=x.device)
        st = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                        window=window, return_complex=True)
        return (st.real.pow(2) + st.imag.pow(2) + 1e-8).sqrt()

    def _mrstft(
        self,
        pred:   torch.Tensor,  # (B, 2, T)
        target: torch.Tensor,  # (B, 2, T)
        w_map:  torch.Tensor | None = None,  # (B, 1, T) or None
    ) -> torch.Tensor:
        B, C, T = pred.shape

        if w_map is not None:
            pred   = pred   * w_map
            target = target * w_map

        p_flat = pred.reshape(B * C, T)
        t_flat = target.reshape(B * C, T)

        total   = pred.new_tensor(0.0)
        n_valid = 0
        for n_fft, hop, win in self.mrstft_cfgs:
            if n_fft >= T:
                continue
            S_p = self._stft_mag(p_flat, n_fft, hop, win)
            S_t = self._stft_mag(t_flat, n_fft, hop, win)

            # Spectral convergence
            sc = ((S_p - S_t).pow(2).sum() + 1e-8).sqrt() / \
                 (S_t.pow(2).sum() + 1e-8).sqrt()
            # Log-magnitude L1
            lm = F.l1_loss(torch.log(S_p + 1e-7), torch.log(S_t + 1e-7))
            total = total + sc + lm
            n_valid += 1

        return total / max(n_valid, 1)

    # ── Attack weight ─────────────────────────────────────────────────────────

    def _attack_weight(
        self,
        target: torch.Tensor,  # (B, 2, T)
        rms_frame: int = 512,
    ) -> torch.Tensor:
        """
        Per-sample weight map: 1.0 base + alpha on rising-RMS frames.
        Follows ddsp-base method: Gaussian-smooth RMS, take positive derivative.
        Returns (B, 1, T).
        """
        B, C, T = target.shape
        # Mono RMS
        mono = target.mean(dim=1)  # (B, T)
        T_trim = (T // rms_frame) * rms_frame
        if T_trim == 0:
            return torch.ones(B, 1, T, device=target.device)

        # RMS per frame: (B, n_frames)
        frames = mono[:, :T_trim].reshape(B, -1, rms_frame)
        rms = (frames.pow(2).mean(dim=-1) + 1e-8).sqrt()   # (B, n_frames)

        # Gaussian smooth (approximate with uniform box filter, sigma frames)
        radius = max(1, int(self.sigma * 2))
        kernel_size = 2 * radius + 1
        kernel = torch.ones(1, 1, kernel_size, device=rms.device) / kernel_size
        rms_s  = F.conv1d(
            rms.unsqueeze(1),
            kernel,
            padding=radius,
        ).squeeze(1)   # (B, n_frames)

        # Positive derivative → rising loudness = attack
        d_rms  = rms_s[:, 1:] - rms_s[:, :-1]   # (B, n_frames-1)
        d_pos  = d_rms.clamp(min=0.0)

        d_max  = d_pos.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        w_frames = 1.0 + self.alpha * (d_pos / d_max)   # (B, n_frames-1), peak=1+alpha

        # Pad to n_frames (first frame = 1.0, rest follow derivative)
        w_frames = F.pad(w_frames, (1, 0), value=1.0)   # (B, n_frames)

        # Upsample from frame-level to sample-level
        w_up = w_frames.unsqueeze(1).repeat_interleave(rms_frame, dim=2)  # (B,1, T_trim)
        if T_trim < T:
            pad_w = torch.ones(B, 1, T - T_trim, device=target.device)
            w_up  = torch.cat([w_up, pad_w], dim=2)

        return w_up   # (B, 1, T)

    # ── F0 logMSE (Simionato 2024 Eq.6) ──────────────────────────────────────

    @staticmethod
    def _f0_logmse(
        B_inh:    torch.Tensor,  # (B,)  inharmonicity from SetterNN
        f0:       torch.Tensor,  # (B,)  fundamental Hz
        f0_off:   torch.Tensor,  # (B,)  offset cents
    ) -> torch.Tensor:
        """
        logMSE between predicted first partial and nominal f0.
        f1_pred = f0_adj * sqrt(1 + B * 1²)
        """
        f0_adj  = f0 * (2.0 ** (f0_off / 1200.0))
        f1_pred = f0_adj * torch.sqrt(1.0 + B_inh)
        # log2 scale (Simionato: log2 because of cents/octave metric)
        eps = 1.0  # Simionato uses ε=1
        return F.mse_loss(torch.log2(f1_pred + eps), torch.log2(f0 + eps))

    # ── RMS envelope (Simionato 2024) ─────────────────────────────────────────

    @staticmethod
    def _rms_envelope(
        pred:   torch.Tensor,  # (B, 2, T)
        target: torch.Tensor,
        frame:  int = 512,
    ) -> torch.Tensor:
        """MSE of per-frame RMS (mono mix)."""
        B, C, T = pred.shape
        T_trim  = (T // frame) * frame
        if T_trim == 0:
            return pred.new_tensor(0.0)

        p_mono = pred[:, :, :T_trim].mean(dim=1).reshape(B, -1, frame)
        t_mono = target[:, :, :T_trim].mean(dim=1).reshape(B, -1, frame)

        rms_p = (p_mono.pow(2).mean(dim=-1) + 1e-8).sqrt()
        rms_t = (t_mono.pow(2).mean(dim=-1) + 1e-8).sqrt()
        return F.mse_loss(rms_p, rms_t)

    # ── Physics regularisation ────────────────────────────────────────────────

    @staticmethod
    def _physics_reg(params: dict) -> torch.Tensor:
        """
        Soft plausibility constraints on SetterNN output params.
        All terms non-negative, sum to physics penalty.
        """
        B_inh    = params['B']          # (N,)
        tau1     = params['tau1']       # (N, K)
        tau2     = params['tau2']       # (N, K)
        beat_hz  = params['beat_hz']    # (N, K)

        # B ∈ [0, 0.4]  (beyond 0.4 is physically unrealistic for piano)
        l_B = F.relu(-B_inh).mean() + F.relu(B_inh - 0.4).mean()

        # τ1 < τ2 (fast attack shorter than sustain) — per partial
        l_tau_order = F.relu(tau1 - tau2).mean()

        # τ2 should decrease with partial number k (higher harmonics decay faster)
        # relu(tau2[:, k+1] - tau2[:, k]) penalises non-monotonic tau2
        l_tau_mono = F.relu(tau2[:, 1:] - tau2[:, :-1]).mean()

        # beat_hz ∈ [0.05, 10] Hz (physically bounded string detuning)
        l_beat = (F.relu(beat_hz - 10.0) + F.relu(0.05 - beat_hz)).mean()

        return l_B + l_tau_order + l_tau_mono + l_beat

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        pred:    torch.Tensor,  # (B, 2, T) synthesized audio
        target:  torch.Tensor,  # (B, 2, T) original recording
        params:  dict | None = None,   # SetterNN output dict (for f0/physics losses)
        f0:      torch.Tensor | None = None,  # (B,) Hz, needed for f0 loss
        phase:   int = 2,       # 1 = spectral only, 2 = full
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred    : synthesized audio (B, 2, T)
            target  : original WAV segment (B, 2, T)
            params  : SetterNN output dict (optional, needed for f0 + physics)
            f0      : fundamental frequencies (B,) Hz
            phase   : 1 = mrstft+l1+rms, 2 = all components

        Returns (total_loss, component_dict).
        """
        ld: Dict[str, float] = {}
        total = pred.new_tensor(0.0)

        # ── Phase 1 & 2: core spectral losses ────────────────────────
        if self.w_mrstft > 0:
            l = self._mrstft(pred, target)
            total = total + self.w_mrstft * l
            ld['mrstft'] = l.item()

        if self.w_l1 > 0:
            l = F.l1_loss(pred, target)
            total = total + self.w_l1 * l
            ld['l1'] = l.item()

        if self.w_rms > 0:
            l = self._rms_envelope(pred, target)
            total = total + self.w_rms * l
            ld['rms'] = l.item()

        # ── Phase 2 only: perceptual + physics ───────────────────────
        if phase >= 2:

            if self.w_attack > 0:
                w_map = self._attack_weight(target)
                l = self._mrstft(pred, target, w_map)
                total = total + self.w_attack * l
                ld['attack'] = l.item()

            if self.w_f0 > 0 and params is not None and f0 is not None:
                l = self._f0_logmse(params['B'], f0, params['f0_offset'])
                total = total + self.w_f0 * l
                ld['f0'] = l.item()

            if self.w_physics > 0 and params is not None:
                l = self._physics_reg(params)
                total = total + self.w_physics * l
                ld['physics'] = l.item()

        ld['total'] = total.item()
        return total, ld
