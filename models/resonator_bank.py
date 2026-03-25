"""
models/resonator_bank.py
Differentiable bank of sinusoidal resonators.

Resonator groups
────────────────
0 … n_h-1          Harmonic    f_i = (i+1)·f0·√(1 + inh·(i+1)²)
n_h … n_h+n_n-1    Noise       fixed log-spaced freqs, phase randomised each frame
n_h+n_n … N-1      Transient   very high decay, only updated during attack phase

Per-frame evolution
───────────────────
1.  GRU controller outputs (Δf_rel, ΔA, raw_gate) for every resonator.
2.  Envelope-phase hard mask limits which resonators may be updated:
      attack  → all N
      sustain → harmonics 0..n_active_sustain-1
      decay   → harmonics 0..n_active_decay-1
      release → none
3.  Effective gate = sigmoid(raw_gate) × phase_mask   (differentiable)
4.  Frequency and amplitude updated where gate > 0.
5.  All amplitudes decay: A(t+1) = A(t) · exp(–decay_i · dt)
6.  256 audio samples synthesised by summing cosines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonatorBank(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        rc = cfg['resonators']
        dc = cfg.get('density', {})

        self.n_harmonic  = int(rc['n_harmonic'])   # 48
        self.n_noise     = int(rc['n_noise'])       #  8
        self.n_transient = int(rc['n_transient'])   #  8
        self.N           = self.n_harmonic + self.n_noise + self.n_transient

        self.sr         = int(cfg['sample_rate'])   # 48 000
        self.frame_size = int(cfg['frame_size'])    #    256
        self.dt         = self.frame_size / self.sr # seconds per frame

        self.n_active_sustain = int(dc.get('n_active_sustain', 12))
        self.n_active_decay   = int(dc.get('n_active_decay',    6))

        # ── Learnable parameters ──────────────────────────────────────

        # Inharmonicity scalar (log-space, always positive)
        inh_init = float(rc.get('inharmonicity_init', 1e-4))
        self.log_inharmonicity = nn.Parameter(
            torch.tensor(math.log(max(inh_init, 1e-8)))
        )

        # Per-resonator log-decay (always positive after exp)
        d_h = float(rc.get('decay_harmonic_init',   0.5))
        d_n = float(rc.get('decay_noise_init',       3.0))
        d_t = float(rc.get('decay_transient_init', 40.0))
        decay_init = torch.cat([
            torch.full((self.n_harmonic,),  math.log(d_h)),
            torch.full((self.n_noise,),     math.log(d_n)),
            torch.full((self.n_transient,), math.log(d_t)),
        ])
        self.log_decay = nn.Parameter(decay_init)  # (N,)

        # Stereo pan per resonator: L = cos(pan), R = sin(pan), pan ∈ [0, π/2]
        pan_init = torch.full((self.N,), math.pi / 4)
        # Gently spread harmonics across the stereo field
        spread = torch.linspace(-0.25, 0.25, self.n_harmonic)
        pan_init[: self.n_harmonic] = math.pi / 4 + spread
        self.pan_params = nn.Parameter(pan_init)  # (N,)

        # ── Fixed buffers ─────────────────────────────────────────────

        # Harmonic indices [1, 2, …, n_harmonic]
        self.register_buffer('harm_idx',
            torch.arange(1, self.n_harmonic + 1, dtype=torch.float32))

        # Noise resonator centre frequencies (log-spaced 200 Hz … 10 kHz)
        noise_f = torch.exp(
            torch.linspace(math.log(200.0), math.log(10000.0), self.n_noise)
        )
        self.register_buffer('noise_freqs', noise_f)

        self.noise_phase_scale = float(rc.get('noise_phase_scale', 0.4))

        TWO_PI = torch.tensor(2.0 * math.pi)
        self.register_buffer('TWO_PI', TWO_PI)

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def inharmonicity(self) -> torch.Tensor:
        return self.log_inharmonicity.exp()

    @property
    def decay(self) -> torch.Tensor:
        return self.log_decay.exp()  # (N,) all > 0

    # ──────────────────────────────────────────────────────────────────
    # State initialisation
    # ──────────────────────────────────────────────────────────────────

    def init_state(
        self,
        f0:       torch.Tensor,   # (B,)
        vel_norm: torch.Tensor,   # (B,)
    ):
        """
        Returns (freqs, amps, phases), each (B, N).
        """
        B      = f0.shape[0]
        device = f0.device
        inh    = self.inharmonicity

        # Harmonic frequencies with inharmonicity stretch
        hi = self.harm_idx  # (n_h,)
        harm_freqs = (
            f0.unsqueeze(1) * hi
            * torch.sqrt(1.0 + inh * hi.pow(2))
        )  # (B, n_h)

        # Noise frequencies — fixed, same for all batch elements
        noise_freqs = self.noise_freqs.unsqueeze(0).expand(B, -1)  # (B, n_n)

        # Transient frequencies: log-spaced from f0*2 to f0*8
        ti = torch.exp(
            torch.linspace(math.log(2.0), math.log(8.0),
                           self.n_transient, device=device)
        )
        trans_freqs = f0.unsqueeze(1) * ti.unsqueeze(0)  # (B, n_t)

        freqs = torch.cat([harm_freqs, noise_freqs, trans_freqs], dim=1)  # (B, N)

        # Initial amplitudes: 1/i spectral roll-off scaled by velocity
        harm_amps  = vel_norm.unsqueeze(1) * (0.08 / hi.unsqueeze(0))
        noise_amps = vel_norm.unsqueeze(1) * torch.full((1, self.n_noise),  0.005, device=device)
        trans_amps = vel_norm.unsqueeze(1) * torch.full((1, self.n_transient), 0.03, device=device)
        amps = torch.cat([harm_amps, noise_amps, trans_amps], dim=1)  # (B, N)

        # Random initial phases
        phases = torch.rand(B, self.N, device=device) * self.TWO_PI

        return freqs, amps, phases

    # ──────────────────────────────────────────────────────────────────
    # Per-frame evolution
    # ──────────────────────────────────────────────────────────────────

    def _phase_mask(
        self,
        phase_label: torch.Tensor,  # (B,) int
    ) -> torch.Tensor:
        """
        Returns float mask (B, N) ∈ {0, 1}.
        Differentiable in the sense that gradient passes through
        the product eff_gate = sigmoid_gate * phase_mask.
        """
        B      = phase_label.shape[0]
        N      = self.N
        device = phase_label.device

        # Start with all ones
        mask = torch.ones(B, N, device=device)

        # Resonator index vector for comparisons
        res_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)

        is_sustain = (phase_label == 1).float().unsqueeze(1)  # (B, 1)
        is_decay   = (phase_label == 2).float().unsqueeze(1)
        is_release = (phase_label == 3).float().unsqueeze(1)
        is_attack  = (phase_label == 0).float().unsqueeze(1)

        n_sus = self.n_active_sustain
        n_dcy = self.n_active_decay
        n_h   = self.n_harmonic
        n_n   = self.n_noise

        # Block resonators >= n_active_* during sustain / decay / release
        block_sus = is_sustain * (res_idx >= n_sus).float()
        block_dcy = is_decay   * (res_idx >= n_dcy).float()
        block_rel = is_release * torch.ones(1, N, device=device)

        # Transients only during attack
        block_trans = (1.0 - is_attack) * (res_idx >= n_h + n_n).float()

        mask = (mask - block_sus - block_dcy - block_rel - block_trans).clamp(0.0, 1.0)
        return mask

    def evolve(
        self,
        freqs:       torch.Tensor,  # (B, N)
        amps:        torch.Tensor,  # (B, N)
        phases:      torch.Tensor,  # (B, N)
        delta_f:     torch.Tensor,  # (B, N) from controller
        delta_a:     torch.Tensor,  # (B, N) from controller
        gate:        torch.Tensor,  # (B, N) sigmoid output from controller
        phase_label: torch.Tensor,  # (B,)  int64
    ):
        """One frame of evolution.  Returns (freqs, amps, phases)."""
        device = freqs.device
        decay  = self.decay.to(device)        # (N,)
        n_h    = self.n_harmonic
        n_n    = self.n_noise

        # Effective gate: soft × hard
        eff_gate = gate * self._phase_mask(phase_label)  # (B, N)

        # ── Frequency update (harmonic resonators only) ───────────────
        df_rel   = 0.01 * torch.tanh(delta_f[:, :n_h])         # (B, n_h) small rel. corr.
        freqs_new = freqs.clone()
        freqs_new[:, :n_h] = freqs[:, :n_h] * (1.0 + df_rel * eff_gate[:, :n_h])

        # ── Amplitude update ──────────────────────────────────────────
        # Gate scales the amplitude delta; then free decay is applied.
        amps_updated = amps + delta_a * eff_gate             # (B, N)
        decay_factor = torch.exp(-decay * self.dt)           # (N,)
        amps_new = F.softplus(amps_updated * decay_factor.unsqueeze(0))

        # ── Phase advance ─────────────────────────────────────────────
        phases_new = (phases + self.TWO_PI * freqs_new * self.dt) % self.TWO_PI

        # Noise resonators: add random phase jitter (simulates band noise)
        if n_n > 0:
            jitter = torch.randn(phases_new.shape[0], n_n, device=device) * self.noise_phase_scale
            phases_new[:, n_h : n_h + n_n] = (
                phases_new[:, n_h : n_h + n_n] + jitter
            ) % self.TWO_PI

        return freqs_new, amps_new, phases_new

    # ──────────────────────────────────────────────────────────────────
    # Frame synthesis
    # ──────────────────────────────────────────────────────────────────

    def synthesize_frame(
        self,
        freqs:  torch.Tensor,  # (B, N)
        amps:   torch.Tensor,  # (B, N)
        phases: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """
        Generates frame_size samples.
        Returns (B, 2, frame_size) stereo audio.
        """
        device = freqs.device
        Fsz    = self.frame_size

        # Local time within frame [0, Fsz-1] / sr  → seconds
        t = torch.arange(Fsz, device=device).float() / self.sr  # (Fsz,)

        # Instantaneous phase: φ_i + 2π·f_i·t_local  →  (B, N, Fsz)
        inst = phases.unsqueeze(-1) + self.TWO_PI * freqs.unsqueeze(-1) * t

        # Sine waves weighted by amplitude: (B, N, Fsz)
        waves = amps.unsqueeze(-1) * torch.sin(inst)

        # Stereo panning
        pan   = self.pan_params.clamp(0.0, math.pi / 2)  # (N,)
        pan_l = torch.cos(pan)  # (N,)
        pan_r = torch.sin(pan)  # (N,)

        audio_l = (waves * pan_l.view(1, -1, 1)).sum(dim=1)  # (B, Fsz)
        audio_r = (waves * pan_r.view(1, -1, 1)).sum(dim=1)  # (B, Fsz)

        return torch.stack([audio_l, audio_r], dim=1)  # (B, 2, Fsz)

    # ──────────────────────────────────────────────────────────────────
    # Full forward pass
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        f0:           torch.Tensor,  # (B,)
        vel_norm:     torch.Tensor,  # (B,)
        control:      torch.Tensor,  # (B, T_frames, N, 2)  [delta_f, delta_a]
        gates:        torch.Tensor,  # (B, T_frames, N)     sigmoid gate
        phase_labels: torch.Tensor,  # (B, T_frames)        int64
    ) -> torch.Tensor:
        """
        Synthesises T_frames * frame_size audio samples.
        Returns (B, 2, T_samples).
        """
        T_frames = control.shape[1]

        freqs, amps, phases = self.init_state(f0, vel_norm)
        audio_frames = []

        for t in range(T_frames):
            delta_f      = control[:, t, :, 0]   # (B, N)
            delta_a      = control[:, t, :, 1]   # (B, N)
            gate         = gates[:, t, :]         # (B, N)
            phase_label  = phase_labels[:, t]     # (B,)

            freqs, amps, phases = self.evolve(
                freqs, amps, phases,
                delta_f, delta_a, gate,
                phase_label,
            )
            audio_frames.append(self.synthesize_frame(freqs, amps, phases))

        return torch.cat(audio_frames, dim=-1)  # (B, 2, T_samples)
