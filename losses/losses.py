"""
losses/losses.py
Composite loss for EGRB synthesis.

Components
──────────
mrstft   Multi-resolution STFT (spectral convergence + log-magnitude L1)
         FFT sizes 256 / 1024 / 4096 — matches WaveSim2 + ddsp-base
l1       Sample-level L1
kin      Kinetics: 1st and 2nd derivative matching (from WaveSim2)
eng      Energy: RMS per 256-sample block (from WaveSim2)
sparse   L1 on gate tensor — reward for sparse control in sustain/decay
attack   Attack frames weighted by attack_weight (from ddsp-base EnvelopeNet trick)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


PHASE_ATTACK  = 0
PHASE_SUSTAIN = 1
PHASE_DECAY   = 2
PHASE_RELEASE = 3


class EGRBLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg['loss']

        self.w_mrstft  = float(lc.get('w_mrstft',   1.0))
        self.w_l1      = float(lc.get('w_l1',        0.5))
        self.w_kin     = float(lc.get('w_kin',       0.4))
        self.w_eng     = float(lc.get('w_eng',       0.4))
        self.w_sparse  = float(lc.get('w_sparsity',  0.03))
        self.atk_wt    = float(lc.get('attack_weight', 5.0))

        self.frame_size = int(cfg['frame_size'])  # 256

        raw_ffts = lc.get('mrstft_ffts', [256, 1024, 4096])
        self.mrstft_cfgs = [(n, n // 4, n) for n in raw_ffts]

    # ──────────────────────────────────────────────────────────────────
    # Attack weight map
    # ──────────────────────────────────────────────────────────────────

    def _attack_weight_map(
        self,
        phase_labels: torch.Tensor,  # (B, T_frames) int64
        T_samples: int,
    ) -> torch.Tensor:
        """
        Returns per-sample weight (B, 1, T_samples).
        Attack frames → attack_weight, others → 1.0.
        """
        B, T_f = phase_labels.shape
        fs      = self.frame_size

        # Frame-level weights (B, T_f)
        frame_w = torch.where(
            phase_labels == PHASE_ATTACK,
            torch.full_like(phase_labels, self.atk_wt, dtype=torch.float32),
            torch.ones_like(phase_labels, dtype=torch.float32),
        )

        # Expand to sample level: repeat each frame weight fs times
        # (B, T_f) → (B, T_f, fs) → (B, T_f*fs) → (B, 1, T_samples)
        w = frame_w.unsqueeze(-1).expand(-1, -1, fs).reshape(B, -1)

        if w.shape[-1] > T_samples:
            w = w[:, :T_samples]
        elif w.shape[-1] < T_samples:
            pad = T_samples - w.shape[-1]
            w = F.pad(w, (0, pad), value=1.0)

        return w.unsqueeze(1)  # (B, 1, T_samples)

    # ──────────────────────────────────────────────────────────────────
    # MRSTFT
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        """(B*C, T) → magnitude spectrogram (B*C, F, frames)"""
        window = torch.hann_window(win, device=x.device)
        st = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                        window=window, return_complex=True)
        return (st.real.pow(2) + st.imag.pow(2) + 1e-8).sqrt()

    def _mrstft(
        self,
        pred:   torch.Tensor,   # (B, 2, T)
        target: torch.Tensor,
        w_map:  torch.Tensor,   # (B, 1, T)
    ) -> torch.Tensor:
        B, C, T = pred.shape
        pred_w   = pred   * w_map
        target_w = target * w_map

        p_flat = pred_w.reshape(B * C, T)
        t_flat = target_w.reshape(B * C, T)

        total = pred.sum() * 0.0
        n_valid = 0
        for n_fft, hop, win in self.mrstft_cfgs:
            if n_fft >= T:
                continue
            S_p = self._stft_mag(p_flat, n_fft, hop, win)
            S_t = self._stft_mag(t_flat, n_fft, hop, win)

            diff = S_p - S_t
            sc   = (diff.pow(2).sum() + 1e-8).sqrt() / (S_t.pow(2).sum() + 1e-8).sqrt()
            lm   = F.l1_loss(torch.log(S_p + 1e-7), torch.log(S_t + 1e-7))
            total = total + sc + lm
            n_valid += 1

        return total / max(n_valid, 1)

    # ──────────────────────────────────────────────────────────────────
    # L1
    # ──────────────────────────────────────────────────────────────────

    def _l1(self, pred, target, w_map):
        return (w_map * (pred - target).abs()).mean()

    # ──────────────────────────────────────────────────────────────────
    # Kinetics (1st + 2nd derivative)
    # ──────────────────────────────────────────────────────────────────

    def _kinetics(self, pred, target, w_map):
        d1_p = pred[:, :, 1:] - pred[:, :, :-1]
        d1_t = target[:, :, 1:] - target[:, :, :-1]
        w1   = w_map[:, :, 1:]
        l1   = (w1 * (d1_p - d1_t).abs()).mean()

        d2_p = d1_p[:, :, 1:] - d1_p[:, :, :-1]
        d2_t = d1_t[:, :, 1:] - d1_t[:, :, :-1]
        w2   = w1[:, :, 1:]
        l2   = (w2 * (d2_p - d2_t).abs()).mean()

        return l1 + 0.5 * l2

    # ──────────────────────────────────────────────────────────────────
    # Energy (RMS per block)
    # ──────────────────────────────────────────────────────────────────

    def _energy(self, pred, target, block: int = 256):
        B, C, T = pred.shape
        T_trim  = (T // block) * block
        if T_trim == 0:
            return pred.sum() * 0.0

        p = pred[:, :, :T_trim].reshape(B, C, -1, block)
        t = target[:, :, :T_trim].reshape(B, C, -1, block)

        rms_p = (p.pow(2).mean(-1) + 1e-8).sqrt()
        rms_t = (t.pow(2).mean(-1) + 1e-8).sqrt()
        return F.l1_loss(rms_p, rms_t)

    # ──────────────────────────────────────────────────────────────────
    # Sparsity (gate penalty in non-attack frames)
    # ──────────────────────────────────────────────────────────────────

    def _sparsity(
        self,
        gates:        torch.Tensor,   # (B, T_frames, N)
        phase_labels: torch.Tensor,   # (B, T_frames)
    ) -> torch.Tensor:
        # Only penalise gates during sustain / decay / release
        non_attack = (phase_labels != PHASE_ATTACK).float().unsqueeze(-1)  # (B, T, 1)
        return (gates * non_attack).mean()

    # ──────────────────────────────────────────────────────────────────
    # Main entry
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        pred:         torch.Tensor,   # (B, 2, T_samples)
        target:       torch.Tensor,   # (B, 2, T_samples)
        phase_labels: torch.Tensor,   # (B, T_frames) int64
        gates:        torch.Tensor,   # (B, T_frames, N)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        T_samples = pred.shape[-1]
        w_map     = self._attack_weight_map(phase_labels, T_samples).to(pred.device)

        ld: Dict[str, float] = {}
        total = pred.sum() * 0.0

        if self.w_mrstft > 0:
            l = self._mrstft(pred, target, w_map)
            total = total + self.w_mrstft * l
            ld['mrstft'] = l.item()

        if self.w_l1 > 0:
            l = self._l1(pred, target, w_map)
            total = total + self.w_l1 * l
            ld['l1'] = l.item()

        if self.w_kin > 0:
            l = self._kinetics(pred, target, w_map)
            total = total + self.w_kin * l
            ld['kin'] = l.item()

        if self.w_eng > 0:
            l_f = self._energy(pred, target, 256)
            l_c = self._energy(pred, target, 2048)
            l   = 0.6 * l_f + 0.4 * l_c
            total = total + self.w_eng * l
            ld['eng'] = l.item()

        if self.w_sparse > 0:
            l = self._sparsity(gates, phase_labels)
            total = total + self.w_sparse * l
            ld['sparse'] = l.item()

        ld['total'] = total.item()
        return total, ld
