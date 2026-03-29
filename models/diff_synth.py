"""
models/diff_synth.py
────────────────────
DifferentiablePianoSynth: physics-based piano synthesizer in pure PyTorch.

All operations are differentiable → gradients flow back to SetterNN.

Signal model per partial k:
  f_k(t)   = k · f0_adj · √(1 + B·k²)                    inharmonicity
  env_k(t) = A0_k · (a1_k·exp(−t/τ1_k) + (1−a1_k)·exp(−t/τ2_k))
  osc_k(t) = cos(2π·f_k·t + φ_k)                          main string
           + beat_depth_k · cos(2π·(f_k+beat_hz_k)·t + φ_k')  detuned string
  signal(t) = Σ_k env_k(t) · osc_k(t)
  noise(t)  = FIR_shaped_noise · exp(−t/attack_tau) · floor_rms

Frame-based synthesis (Simionato 2024 insight):
  Envelope computed per frame (frame_size samples), phases accumulated across
  frames → continuous sinusoids with no phase discontinuity.

Memory optimisation:
  Full-length synthesis is O(B·K·T). For T=96000 (2s@48kHz), K=64, B=8:
  ≈ 49M floats ≈ 196MB — acceptable on GPU. If memory is tight, synthesize
  in overlapping segments and sum (not implemented here for simplicity).

Stereo:
  Mid channel = mono synthesis.
  Side channel = mono · width_factor (from spectral_eq).
  L = M + S,  R = M − S  (normalised).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── DifferentiablePianoSynth ──────────────────────────────────────────────────

class DifferentiablePianoSynth(nn.Module):
    """
    Stateless differentiable synthesizer — no learnable parameters here.
    All parameters come from SetterNN at forward time.

    Args (constructor):
        sr         : sample rate (default 48000)
        frame_size : synthesis frame in samples (default 240 = 5ms @ 48kHz)
        noise_order: FIR noise shaping filter length (default 64)
    """

    def __init__(self, sr: int = 48000, frame_size: int = 240, noise_order: int = 64):
        super().__init__()
        self.sr          = sr
        self.frame_size  = frame_size
        self.noise_order = noise_order

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        params:          dict,
        f0:              torch.Tensor,   # (B,) Hz
        duration_s:      float,
        width_factor:    torch.Tensor | None = None,  # (B,) stereo width
    ) -> torch.Tensor:
        """
        Synthesize piano audio from SetterNN params.

        Returns (B, 2, n_samples) stereo float32.
        n_samples = int(duration_s * sr)
        """
        device  = f0.device
        B_batch = f0.shape[0]
        sr      = self.sr

        n_samples = int(duration_s * sr)
        n_frames  = math.ceil(n_samples / self.frame_size)
        T         = n_frames * self.frame_size  # padded length (trimmed after)

        # ── Unpack params ────────────────────────────────────────────
        B_inh    = params['B']            # (B,)
        f0_off   = params['f0_offset']    # (B,) cents
        A0       = params['A0']           # (B, K)
        tau1     = params['tau1']         # (B, K)
        tau2     = params['tau2']         # (B, K)
        a1       = params['a1']           # (B, K)
        beat_hz  = params['beat_hz']      # (B, K)
        beat_dep = params['beat_depth']   # (B, K)
        noise_p  = params['noise']        # (B, 4)

        K = A0.shape[1]

        # ── Adjusted f0 (tuning offset in cents) ─────────────────────
        f0_adj = f0 * (2.0 ** (f0_off / 1200.0))  # (B,)

        # ── Inharmonic partial frequencies ────────────────────────────
        ks    = torch.arange(1, K + 1, device=device, dtype=f0.dtype)   # (K,)
        f0_e  = f0_adj.unsqueeze(1)      # (B, 1)
        B_e   = B_inh.unsqueeze(1)       # (B, 1)
        ks_e  = ks.unsqueeze(0)          # (1, K)
        f_k   = f0_e * ks_e * torch.sqrt(1.0 + B_e * ks_e ** 2)  # (B, K)

        # ── Beat partner frequencies ──────────────────────────────────
        f_k_beat = f_k + beat_hz         # (B, K) detuned string

        # ── Time axis ─────────────────────────────────────────────────
        t = torch.arange(T, device=device, dtype=f0.dtype) / sr  # (T,)

        # ── Bi-exponential amplitude envelope ────────────────────────
        # env_k(t) = A0_k * (a1_k * exp(-t/tau1_k) + (1-a1_k) * exp(-t/tau2_k))
        # Shapes: A0 (B,K), tau1 (B,K), t (T,)
        # Broadcast: (B, K, 1) × (1, 1, T) → (B, K, T)
        t3      = t.view(1, 1, -1)
        tau1_3  = tau1.unsqueeze(2)       # (B, K, 1)
        tau2_3  = tau2.unsqueeze(2)
        a1_3    = a1.unsqueeze(2)
        A0_3    = A0.unsqueeze(2)

        e_fast  = torch.exp(-t3 / tau1_3)                        # (B, K, T)
        e_slow  = torch.exp(-t3 / tau2_3)
        env     = A0_3 * (a1_3 * e_fast + (1.0 - a1_3) * e_slow)  # (B, K, T)

        # ── Phase-continuous oscillators ──────────────────────────────
        # Random initial phases (same seed for evaluation reproducibility).
        # In production inference, phases should be reset per note-on.
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        phi_main = (torch.rand(B_batch, K, device=device, generator=gen) * 2 * math.pi
                    ).unsqueeze(2)   # (B, K, 1)
        phi_beat = (torch.rand(B_batch, K, device=device, generator=gen) * 2 * math.pi
                    ).unsqueeze(2)

        f_k3      = f_k.unsqueeze(2)        # (B, K, 1)
        f_k_beat3 = f_k_beat.unsqueeze(2)
        beat_dep3 = beat_dep.unsqueeze(2)   # (B, K, 1)

        osc_main = torch.cos(2.0 * math.pi * f_k3 * t3 + phi_main)
        osc_beat = torch.cos(2.0 * math.pi * f_k_beat3 * t3 + phi_beat)
        osc      = osc_main + beat_dep3 * osc_beat   # (B, K, T)

        # ── Sum partials ──────────────────────────────────────────────
        harmonic = (env * osc).sum(dim=1)   # (B, T)

        # Normalise so max absolute value ≤ 0.9 (prevents clipping)
        peak = harmonic.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
        harmonic = harmonic / peak * 0.9

        # ── Noise ─────────────────────────────────────────────────────
        noise_out = self._synthesize_noise(noise_p, t, device, B_batch, T)
        mono = harmonic + noise_out          # (B, T)

        # ── Trim to exact length ──────────────────────────────────────
        mono = mono[:, :n_samples]          # (B, n_samples)

        # ── Stereo (Mid/Side) ─────────────────────────────────────────
        if width_factor is None:
            width_factor = torch.ones(B_batch, device=device)

        w  = width_factor.clamp(0.0, 2.0).unsqueeze(1)  # (B, 1)
        mid  = mono
        side = mono * w * 0.3               # side = attenuated copy of mid

        left  = (mid + side).clamp(-1.0, 1.0)
        right = (mid - side).clamp(-1.0, 1.0)

        return torch.stack([left, right], dim=1)  # (B, 2, n_samples)

    # ── Noise synthesis ───────────────────────────────────────────────────────

    def _synthesize_noise(
        self,
        noise_p:  torch.Tensor,  # (B, 4): [attack_tau, floor_rms, centroid_norm, slope_db_oct]
        t:        torch.Tensor,  # (T,)
        device:   torch.device,
        B_batch:  int,
        T:        int,
    ) -> torch.Tensor:
        """
        Coloured noise with exponentially decaying attack envelope.
        Returns (B, T).
        """
        attack_tau   = noise_p[:, 0].unsqueeze(1)   # (B, 1)
        floor_rms    = noise_p[:, 1].unsqueeze(1)   # (B, 1)
        centroid_norm = noise_p[:, 2]               # (B,)
        slope_db_oct  = noise_p[:, 3]               # (B,)

        # White noise
        white = torch.randn(B_batch, T, device=device)

        # Per-batch FIR shaping filter derived from centroid and slope
        # Design: 1-pole IIR approximation (simple, differentiable)
        # cutoff = F0_LO * 2^(centroid_norm * log2(nyquist/F0_LO))
        nyquist = self.sr / 2.0
        f_lo    = 27.5
        log_range = math.log2(nyquist / f_lo)
        cutoff_hz = f_lo * (2.0 ** (centroid_norm * log_range))     # (B,)
        cutoff_norm = (cutoff_hz / nyquist).clamp(0.01, 0.99)        # (B,)

        # Pole of 1st-order IIR LPF: pole = exp(-2π * fc/fs)
        pole = torch.exp(-2.0 * math.pi * cutoff_norm)               # (B,)

        # Apply IIR filter batch-wise (loop over samples — unavoidable for IIR)
        alpha = (1.0 - pole).unsqueeze(1)    # (B, 1)
        pole_ = pole.unsqueeze(1)            # (B, 1)
        out   = torch.zeros(B_batch, T, device=device)
        prev  = torch.zeros(B_batch, device=device)
        for i in range(T):
            prev = alpha.squeeze(1) * white[:, i] + pole_.squeeze(1) * prev
            out[:, i] = prev

        # Attack envelope
        t3 = t.view(1, -1)
        env = torch.exp(-t3 / attack_tau.clamp(0.002)) * floor_rms

        return out * env    # (B, T)

    # ── Convenience: synthesize single note (no grad) ─────────────────────────

    @torch.no_grad()
    def synthesize_note(
        self,
        setter_nn:   nn.Module,
        midi:        int,
        vel_idx:     int,
        duration_s:  float = 3.0,
        device:      str   = 'cpu',
    ) -> torch.Tensor:
        """
        Synthesize a single note. Returns (2, n_samples) CPU float32.
        Uses setter_nn to get params.
        """
        self.eval()
        setter_nn.eval()
        f0  = torch.tensor([440.0 * 2.0 ** ((midi - 69) / 12.0)], device=device)
        vel = torch.tensor([vel_idx / 7.0], device=device)
        params = setter_nn(f0, vel)
        audio = self.forward(params, f0, duration_s)
        return audio.squeeze(0).cpu()   # (2, n_samples)
