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
1.  GRU controller outputs (Δf_rel, target_amp, raw_gate) for every resonator.
2.  Envelope-phase hard mask limits which resonators may be updated:
      attack  → all curriculum-active resonators
      sustain → harmonics 0..n_active_sustain-1
      decay   → harmonics 0..n_active_decay-1
      release → none
3.  Effective gate = sigmoid(raw_gate) × phase_mask × curriculum_mask
4.  Harmonic frequencies are re-anchored to ideal positions each frame
    (prevents frequency drift).  delta_f adds a tiny ±0.5 % inharmonicity
    correction.
5.  Amplitudes: gate blends between natural decay and predicted target_amp.
      amps_new = amps_decayed·(1–gate) + target_amp·gate
6.  256 audio samples synthesised by summing cosines.

Resonator curriculum
────────────────────
Call  bank.set_active_resonators(n_h, n_n, n_t)  at each curriculum phase
transition.  Only active resonators receive controller updates and contribute
to the audio output.  Inactive resonators have their amplitudes zeroed.
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

        self.n_harmonic  = int(rc['n_harmonic'])   # max harmonics
        self.n_noise     = int(rc['n_noise'])       # max noise resonators
        self.n_transient = int(rc['n_transient'])   # max transient resonators
        self.N           = self.n_harmonic + self.n_noise + self.n_transient

        self.sr         = int(cfg['sample_rate'])
        self.frame_size = int(cfg['frame_size'])
        self.dt         = self.frame_size / self.sr  # seconds per frame

        self.n_active_sustain = int(dc.get('n_active_sustain', 12))
        self.n_active_decay   = int(dc.get('n_active_decay',    6))

        # ── Curriculum state (non-parameter, updated by train.py) ─────
        rc_cfg = cfg.get('resonator_curriculum', {})
        self._n_active_h = int(rc_cfg.get('p1_n_harmonic',  min(8,  self.n_harmonic)))
        self._n_active_n = int(rc_cfg.get('p1_n_noise',     0))
        self._n_active_t = int(rc_cfg.get('p1_n_transient', 0))

        # ── Learnable parameters ──────────────────────────────────────

        inh_init = float(rc.get('inharmonicity_init', 1e-4))
        self.log_inharmonicity = nn.Parameter(
            torch.tensor(math.log(max(inh_init, 1e-8)))
        )

        # Per-partial systematic tuning deviation (±1% via tanh).
        # Captures piano-like deviations from the ideal inharmonicity formula.
        self.harm_detune = nn.Parameter(torch.zeros(self.n_harmonic))

        d_h = float(rc.get('decay_harmonic_init',   0.5))
        d_n = float(rc.get('decay_noise_init',       3.0))
        d_t = float(rc.get('decay_transient_init', 40.0))
        decay_init = torch.cat([
            torch.full((self.n_harmonic,),  math.log(d_h)),
            torch.full((self.n_noise,),     math.log(d_n)),
            torch.full((self.n_transient,), math.log(d_t)),
        ])
        self.log_decay = nn.Parameter(decay_init)  # (N,)

        # Per-resonator log amplitude scale — learned overall level per resonator
        self.log_amp_scale = nn.Parameter(
            torch.cat([
                torch.log(0.08 / torch.arange(1, self.n_harmonic + 1).float()),
                torch.full((self.n_noise,),     math.log(0.005)),
                torch.full((self.n_transient,), math.log(0.03)),
            ])
        )  # (N,)

        # Stereo pan
        pan_init = torch.full((self.N,), math.pi / 4)
        spread   = torch.linspace(-0.25, 0.25, self.n_harmonic)
        pan_init[: self.n_harmonic] = math.pi / 4 + spread
        self.pan_params = nn.Parameter(pan_init)

        # ── Fixed buffers ─────────────────────────────────────────────

        self.register_buffer('harm_idx',
            torch.arange(1, self.n_harmonic + 1, dtype=torch.float32))

        noise_f = torch.exp(
            torch.linspace(math.log(200.0), math.log(10000.0), self.n_noise)
        ) if self.n_noise > 0 else torch.zeros(0)
        self.register_buffer('noise_freqs', noise_f)

        self.noise_phase_scale = float(rc.get('noise_phase_scale', 0.4))

        self.register_buffer('TWO_PI', torch.tensor(2.0 * math.pi))

    # ──────────────────────────────────────────────────────────────────
    # Curriculum API
    # ──────────────────────────────────────────────────────────────────

    def set_active_resonators(self, n_h: int, n_n: int, n_t: int):
        """Update how many resonators of each type are active."""
        self._n_active_h = min(int(n_h), self.n_harmonic)
        self._n_active_n = min(int(n_n), self.n_noise)
        self._n_active_t = min(int(n_t), self.n_transient)

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def inharmonicity(self) -> torch.Tensor:
        return self.log_inharmonicity.exp()

    @property
    def decay(self) -> torch.Tensor:
        return self.log_decay.exp()

    @property
    def amp_scale(self) -> torch.Tensor:
        return self.log_amp_scale.exp()

    def _ideal_harmonic_freqs(self, f0: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_harmonic) ideal harmonic frequencies."""
        inh    = self.inharmonicity
        hi     = self.harm_idx                                    # (n_h,)
        base   = f0.unsqueeze(1) * hi * torch.sqrt(1.0 + inh * hi.pow(2))
        detune = 0.01 * torch.tanh(self.harm_detune).unsqueeze(0)  # ±1%
        return base * (1.0 + detune)

    # ──────────────────────────────────────────────────────────────────
    # State initialisation
    # ──────────────────────────────────────────────────────────────────

    def init_state(
        self,
        f0:       torch.Tensor,   # (B,)
        vel_norm: torch.Tensor,   # (B,)
    ):
        """Returns (freqs, amps, phases), each (B, N)."""
        B      = f0.shape[0]
        device = f0.device

        harm_freqs = self._ideal_harmonic_freqs(f0)              # (B, n_h)

        noise_freqs = (self.noise_freqs.unsqueeze(0).expand(B, -1)
                       if self.n_noise > 0
                       else torch.zeros(B, 0, device=device))

        if self.n_transient > 0:
            ti = torch.exp(torch.linspace(
                math.log(2.0), math.log(8.0),
                self.n_transient, device=device))
            trans_freqs = f0.unsqueeze(1) * ti.unsqueeze(0)
        else:
            trans_freqs = torch.zeros(B, 0, device=device)

        freqs = torch.cat([harm_freqs, noise_freqs, trans_freqs], dim=1)

        # Amplitudes scaled by vel_norm and learnable amp_scale
        amps = vel_norm.unsqueeze(1) * self.amp_scale.unsqueeze(0).to(device)

        # Zero out inactive resonators so they don't leak into the output
        n_h, n_n = self.n_harmonic, self.n_noise
        amps = amps.clone()
        amps[:, self._n_active_h : n_h]               = 0.0
        amps[:, n_h + self._n_active_n : n_h + n_n]   = 0.0
        amps[:, n_h + n_n + self._n_active_t :]        = 0.0

        phases = torch.rand(B, self.N, device=device) * self.TWO_PI

        return freqs, amps, phases

    # ──────────────────────────────────────────────────────────────────
    # Phase + curriculum mask
    # ──────────────────────────────────────────────────────────────────

    def _phase_mask(
        self,
        phase_label: torch.Tensor,  # (B,) int
    ) -> torch.Tensor:
        """Returns float mask (B, N) ∈ {0, 1}."""
        B      = phase_label.shape[0]
        N      = self.N
        device = phase_label.device
        n_h, n_n = self.n_harmonic, self.n_noise

        mask    = torch.ones(B, N, device=device)
        res_idx = torch.arange(N, device=device).unsqueeze(0)  # (1, N)

        is_sustain = (phase_label == 1).float().unsqueeze(1)
        is_decay   = (phase_label == 2).float().unsqueeze(1)
        is_release = (phase_label == 3).float().unsqueeze(1)
        is_attack  = (phase_label == 0).float().unsqueeze(1)

        # Phase-based density limits
        block_sus   = is_sustain * (res_idx >= self.n_active_sustain).float()
        block_dcy   = is_decay   * (res_idx >= self.n_active_decay).float()
        block_rel   = is_release * torch.ones(1, N, device=device)
        block_trans = (1.0 - is_attack) * (res_idx >= n_h + n_n).float()

        # Curriculum: block resonators beyond current active counts
        block_curr_h = (res_idx >= self._n_active_h).float() * (res_idx < n_h).float()
        block_curr_n = ((res_idx >= n_h + self._n_active_n).float()
                        * (res_idx < n_h + n_n).float())
        block_curr_t = (res_idx >= n_h + n_n + self._n_active_t).float()

        mask = (mask
                - block_sus - block_dcy - block_rel - block_trans
                - block_curr_h - block_curr_n - block_curr_t
                ).clamp(0.0, 1.0)
        return mask

    # ──────────────────────────────────────────────────────────────────
    # Per-frame evolution
    # ──────────────────────────────────────────────────────────────────

    def evolve(
        self,
        freqs:         torch.Tensor,   # (B, N)
        amps:          torch.Tensor,   # (B, N)  ≥ 0
        phases:        torch.Tensor,   # (B, N)
        delta_f:       torch.Tensor,   # (B, N)  inharmonicity correction from controller
        raw_exc:       torch.Tensor,   # (B, N)  excitation energy (before softplus)
        raw_decay_mul: torch.Tensor,   # (B, N)  decay rate modulation (before sigmoid)
        gate:          torch.Tensor,   # (B, N)  sigmoid gate = excitation mask
        phase_label:   torch.Tensor,   # (B,)    int64
        f0:            torch.Tensor,   # (B,)    fundamental frequency
    ):
        """One frame of evolution.  Returns (freqs, amps, phases)."""
        device    = freqs.device
        base_decay = self.decay.to(device)   # (N,)
        n_h        = self.n_harmonic
        n_n        = self.n_noise
        amp_scale  = self.amp_scale.to(device)

        eff_gate = gate * self._phase_mask(phase_label)   # (B, N)

        # ── Frequency: re-anchor harmonics each frame ─────────────────
        ideal_h   = self._ideal_harmonic_freqs(f0)           # (B, n_h)
        df_corr   = 0.005 * torch.tanh(delta_f[:, :n_h])    # (B, n_h) ±0.5%
        freqs_new = freqs.clone()
        freqs_new[:, :n_h] = ideal_h * (1.0 + df_corr)

        # ── Amplitude: branch-specific update ────────────────────────

        # Harmonic branch — physical decay + excitation injection
        # decay_scale ∈ [0.5, 2.0]: modulates how fast each partial decays
        decay_mod_h   = torch.sigmoid(raw_decay_mul[:, :n_h])      # (B, n_h)
        decay_scale_h = 0.5 + 1.5 * decay_mod_h                    # (B, n_h)
        eff_decay_h   = base_decay[:n_h].unsqueeze(0) * decay_scale_h
        decay_fac_h   = torch.exp(-eff_decay_h * self.dt)
        amps_decayed_h = amps[:, :n_h] * decay_fac_h

        excitation_h = F.softplus(raw_exc[:, :n_h]) * amp_scale[:n_h].unsqueeze(0)
        injected_h   = excitation_h * eff_gate[:, :n_h]

        # Noise + transient branches — keep stable blend (gate → target_amp)
        # target_amp for these branches derived from raw_exc via softplus
        nt_slice = slice(n_h, self.N)
        decay_fac_nt  = torch.exp(-base_decay[n_h:].unsqueeze(0) * self.dt)
        amps_decayed_nt = amps[:, n_h:] * decay_fac_nt
        target_nt     = F.softplus(raw_exc[:, n_h:]) * amp_scale[n_h:].unsqueeze(0)
        eff_gate_nt   = eff_gate[:, n_h:]

        amps_new = amps.clone()
        amps_new[:, :n_h]  = amps_decayed_h + injected_h
        amps_new[:, n_h:]  = amps_decayed_nt * (1.0 - eff_gate_nt) + target_nt * eff_gate_nt

        # ── Phase advance ─────────────────────────────────────────────
        phases_new = (phases + self.TWO_PI * freqs_new * self.dt) % self.TWO_PI

        # Noise: random phase jitter (kept for noise branch consistency)
        if n_n > 0 and self._n_active_n > 0:
            jitter = (torch.randn(amps_new.shape[0], n_n, device=device)
                      * self.noise_phase_scale)
            phases_new[:, n_h : n_h + n_n] = (
                phases_new[:, n_h : n_h + n_n] + jitter
            ) % self.TWO_PI

        return freqs_new, amps_new, phases_new

    # ──────────────────────────────────────────────────────────────────
    # Frame synthesis
    # ──────────────────────────────────────────────────────────────────

    def synthesize_frame(
        self,
        freqs:  torch.Tensor,   # (B, N)
        amps:   torch.Tensor,   # (B, N)
        phases: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:
        """Returns (B, 2, frame_size) stereo audio."""
        device = freqs.device
        Fsz    = self.frame_size

        B      = freqs.shape[0]
        n_h    = self._n_active_h
        n_n    = self._n_active_n
        n_h_max = self.n_harmonic

        t     = torch.arange(Fsz, device=device).float() / self.sr  # (Fsz,)

        # ── Harmonic + transient branches: sine oscillators ───────────
        # Exclude noise slots from sinusoidal rendering
        harm_trans_idx = list(range(n_h_max)) + list(range(n_h_max + self.n_noise, self.N))
        freqs_ht  = freqs[:, harm_trans_idx]
        amps_ht   = amps[:,  harm_trans_idx]
        phases_ht = phases[:, harm_trans_idx]

        inst   = phases_ht.unsqueeze(-1) + self.TWO_PI * freqs_ht.unsqueeze(-1) * t
        waves  = amps_ht.unsqueeze(-1) * torch.sin(inst)          # (B, N_ht, Fsz)

        # ── Noise branch: white noise scaled by predicted amplitude ───
        if self.n_noise > 0 and n_n > 0:
            noise_amps = amps[:, n_h_max : n_h_max + self.n_noise]  # (B, n_noise)
            noise_sig  = torch.randn(B, self.n_noise, Fsz, device=device)
            noise_sig  = noise_sig * noise_amps.unsqueeze(-1)
        else:
            noise_sig = torch.zeros(B, self.n_noise, Fsz, device=device)

        # ── Pan + mix ─────────────────────────────────────────────────
        pan_ht = self.pan_params[harm_trans_idx].clamp(0.0, math.pi / 2)
        pan_l  = torch.cos(pan_ht)
        pan_r  = torch.sin(pan_ht)

        pan_n  = self.pan_params[n_h_max : n_h_max + self.n_noise].clamp(0.0, math.pi / 2)
        pan_nl = torch.cos(pan_n)
        pan_nr = torch.sin(pan_n)

        audio_l = (waves * pan_l.view(1, -1, 1)).sum(dim=1)
        audio_r = (waves * pan_r.view(1, -1, 1)).sum(dim=1)
        audio_l = audio_l + (noise_sig * pan_nl.view(1, -1, 1)).sum(dim=1)
        audio_r = audio_r + (noise_sig * pan_nr.view(1, -1, 1)).sum(dim=1)

        return torch.stack([audio_l, audio_r], dim=1)  # (B, 2, Fsz)

    # ──────────────────────────────────────────────────────────────────
    # Full forward pass
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        f0:           torch.Tensor,   # (B,)
        vel_norm:     torch.Tensor,   # (B,)
        control:      torch.Tensor,   # (B, T_frames, N, 3)  [delta_f, raw_exc, raw_decay_mul]
        gates:        torch.Tensor,   # (B, T_frames, N)     sigmoid gate
        phase_labels: torch.Tensor,   # (B, T_frames)        int64
    ) -> torch.Tensor:
        """
        Synthesises T_frames * frame_size audio samples.
        Returns (B, 2, T_samples).
        """
        T_frames = control.shape[1]
        amp_scale = self.amp_scale.to(f0.device)   # (N,)

        freqs, amps, phases = self.init_state(f0, vel_norm)
        audio_frames = []

        for t in range(T_frames):
            delta_f       = control[:, t, :, 0]   # (B, N)
            raw_exc       = control[:, t, :, 1]   # (B, N)
            raw_decay_mul = control[:, t, :, 2]   # (B, N)
            gate          = gates[:, t, :]         # (B, N)
            pl            = phase_labels[:, t]     # (B,)

            freqs, amps, phases = self.evolve(
                freqs, amps, phases,
                delta_f, raw_exc, raw_decay_mul, gate, pl, f0,
            )
            audio_frames.append(self.synthesize_frame(freqs, amps, phases))

        return torch.cat(audio_frames, dim=-1)   # (B, 2, T_samples)
