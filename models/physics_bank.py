"""
models/physics_bank.py
──────────────────────
Physics-Informed Resonator Bank — differentiable, training-ready.

Physical model per harmonic k = 1..K:
  f_k    = k · f0 · √(1 + B·k²)            inharmonicity
  A_k(t) = a1·exp(-t/τ1) + a2·exp(-t/τ2)   bi-exponential decay
  osc    = Σ_s cos(2π·f_ks·t + φ_s)        N_strings independent oscillators
  signal = A_k(t) · osc / N_strings

Beating: strings at f_k + {-Δf, 0, +Δf} (trichord) or {-Δf/2, +Δf/2} (bichord)
  → natural amplitude modulation from phase interference

Soundboard: parametric convolution [0..1]
  0.0 = bypass (physical soundboard present)
  1.0 = full virtual body resonance

All physical parameters are nn.Parameter → trainable via gradient.
Initialized from params.json analytical priors when available.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── MIDI helpers ──────────────────────────────────────────────────────────────

MIDI_LO = 21
MIDI_HI = 108
F0_LO   = 27.5
F0_HI   = 4186.0

def midi_to_hz_tensor(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

def n_strings_for_midi(midi: int) -> int:
    """Number of strings per note (typical grand piano stringing)."""
    if midi <= 27:   return 1   # lowest bass: monochord
    elif midi <= 47: return 2   # low-mid: bichord
    else:            return 3   # upper register: trichord

def _safe_mean(vals: list, default: float) -> float:
    v = [x for x in vals if x is not None and not math.isnan(x) and not math.isinf(x)]
    return float(sum(v) / len(v)) if v else default


# ── Param priors from params.json ─────────────────────────────────────────────

def load_priors(params_path: str, K: int) -> dict:
    """
    Read analytical priors from params.json.
    Returns dict with per-MIDI aggregated parameters.
    """
    if not Path(params_path).exists():
        return {}
    with open(params_path) as f:
        data = json.load(f)
    samples = data.get('samples', {})

    # Collect per-MIDI values (averaged over velocity)
    by_midi = {}
    for key, s in samples.items():
        midi = s['midi']
        if midi not in by_midi:
            by_midi[midi] = {'B': [], 'tau1': [[] for _ in range(K)],
                             'tau2': [[] for _ in range(K)], 'a1': [[] for _ in range(K)],
                             'beat_hz': [[] for _ in range(K)], 'A0': [[] for _ in range(K)]}
        if s['B'] > 0:
            by_midi[midi]['B'].append(s['B'])
        for p in s.get('partials', []):
            k_idx = p['k'] - 1
            if k_idx >= K: continue
            if p.get('tau1'): by_midi[midi]['tau1'][k_idx].append(p['tau1'])
            if p.get('tau2'): by_midi[midi]['tau2'][k_idx].append(p['tau2'])
            if p.get('a1') and 0 < p['a1'] < 1: by_midi[midi]['a1'][k_idx].append(p['a1'])
            if p.get('beat_hz') and p['beat_hz'] > 0.05: by_midi[midi]['beat_hz'][k_idx].append(p['beat_hz'])
            if p.get('A0') and p['A0'] > 0: by_midi[midi]['A0'][k_idx].append(p['A0'])

    # Average
    out = {}
    for midi, d in by_midi.items():
        out[midi] = {
            'B': _safe_mean(d['B'], 1e-4),
            'tau1':    [_safe_mean(v, 0.3 + 0.1 * i) for i, v in enumerate(d['tau1'])],
            'tau2':    [_safe_mean(v, max(5.0 - i * 0.05, 0.1)) for i, v in enumerate(d['tau2'])],
            'a1':      [_safe_mean(v, 0.25) for v in d['a1']],
            'beat_hz': [_safe_mean(v, 0.4 + 0.02 * i) for i, v in enumerate(d['beat_hz'])],
            'A0':      [_safe_mean(v, max(1.0 / (i + 1), 1e-4)) for i, v in enumerate(d['A0'])],
        }
    return out


# ── Physics Bank ──────────────────────────────────────────────────────────────

class PhysicsResonatorBank(nn.Module):
    """
    Differentiable physics-informed piano synthesizer.

    Parameters (all learnable):
      log_B_slope, log_B_intercept : B(midi_norm) = exp(slope*midi_norm + intercept)
      log_tau1[K]  : fast decay time per partial (hammer/early)
      log_tau2[K]  : slow decay time per partial (string natural)
      logit_a1[K]  : fast/slow mixing ratio (sigmoid → [0,1])
      log_beat_hz[K]: string detuning (Hz) for beating
      log_A0[K]    : partial amplitude spectrum (in log space)
      harm_detune[K]: systematic ±0.5% per-partial tuning correction
      log_noise_level: noise amplitude relative to harmonic
      log_tau_noise  : noise attack decay time
      log_soundboard[n_sb_taps]: learnable soundboard IR taps (FIR)
      soundboard_strength       : scalar [0..1] (parametric mix)

    Forward args:
      f0       : (B,)  fundamental Hz
      vel_norm : (B,)  velocity [0..1]
      n_frames : int   number of synthesis frames
      ctrl     : optional dict of per-frame controller corrections
    """

    def __init__(self, cfg: dict, params_path: str = None):
        super().__init__()

        pc = cfg.get('physics', {})
        self.K          = int(pc.get('n_harmonic', 80))
        self.sr         = int(cfg.get('sample_rate', 44100))
        self.frame_size = int(cfg.get('frame_size', 256))
        self.dt         = self.frame_size / self.sr

        # ── Inharmonicity: B(midi_norm) = exp(slope*x + intercept) ───
        self.log_B_slope     = nn.Parameter(torch.tensor(-2.0))
        self.log_B_intercept = nn.Parameter(torch.tensor(-7.5))

        K = self.K

        # ── Bi-exponential decay ──────────────────────────────────────
        # τ1 (fast, 0.01–5s): decreases with harmonic number
        log_tau1_init = torch.linspace(math.log(0.4), math.log(0.05), K)
        # τ2 (slow, 0.05–60s): decreases faster with harmonic number
        log_tau2_init = torch.linspace(math.log(5.0), math.log(0.3), K)
        # a1 (fast mixing, 0.1–0.5): more fast component at high k
        logit_a1_init = torch.linspace(-1.4, 0.0, K)  # sigmoid(-1.4)≈0.20, sigmoid(0)=0.50

        self.log_tau1  = nn.Parameter(log_tau1_init)
        self.log_tau2  = nn.Parameter(log_tau2_init)
        self.logit_a1  = nn.Parameter(logit_a1_init)

        # ── Beating (string-to-string detuning) ──────────────────────
        # Δf range: 0.05–10 Hz, init: gradually increasing with k
        log_beat_init = torch.linspace(math.log(0.3), math.log(1.5), K)
        self.log_beat_hz = nn.Parameter(log_beat_init)

        # ── Amplitude spectrum ────────────────────────────────────────
        # log_A0[k]: unnormalized log amplitude. Softmax → normalized spectrum.
        # Init: ~1/k^0.7 (bright piano)
        log_A0_init = -torch.log(torch.arange(1, K + 1).float()) * 0.7
        self.log_A0 = nn.Parameter(log_A0_init)

        # Per-partial systematic tuning deviation (±0.5%)
        self.harm_detune = nn.Parameter(torch.zeros(K))

        # ── Noise ─────────────────────────────────────────────────────
        self.log_noise_level = nn.Parameter(torch.tensor(math.log(0.06)))
        self.log_tau_noise   = nn.Parameter(torch.tensor(math.log(0.05)))
        # First-order IIR shaping pole (0 = no shaping, near 1 = strong LPF)
        self.noise_pole      = nn.Parameter(torch.tensor(0.85))

        # ── Soundboard (parametric FIR) ───────────────────────────────
        n_sb = int(pc.get('n_soundboard_taps', 256))
        self.n_sb = n_sb
        # Initialize as delta (no effect), then learned
        sb_init = torch.zeros(n_sb)
        sb_init[0] = 1.0  # delta function → unity passthrough initially
        self.soundboard_ir   = nn.Parameter(sb_init)
        # Strength: 0.0 = bypass, 1.0 = full
        # Use sigmoid(raw) for stability
        self.soundboard_raw  = nn.Parameter(torch.tensor(-0.6))  # sigmoid≈0.35

        # ── Load priors ────────────────────────────────────────────────
        if params_path is not None:
            self._init_from_params(params_path)

    # ── Initialization from analytical priors ─────────────────────────────────

    def _init_from_params(self, params_path: str):
        priors = load_priors(params_path, self.K)
        if not priors:
            print(f"[PhysicsBank] No priors found in {params_path}")
            return

        # Fit B(midi) log-linear: log(B) = slope * midi_norm + intercept
        midi_vals = sorted(priors.keys())
        B_vals = [(m, priors[m]['B']) for m in midi_vals if priors[m]['B'] > 1e-7]
        if len(B_vals) >= 4:
            ms = torch.tensor([(m - MIDI_LO) / (MIDI_HI - MIDI_LO) for m, _ in B_vals])
            bs = torch.tensor([math.log(b) for _, b in B_vals])
            # Least squares: [slope, intercept]
            A = torch.stack([ms, torch.ones_like(ms)], dim=1)
            sol = torch.linalg.lstsq(A, bs.unsqueeze(1)).solution
            with torch.no_grad():
                self.log_B_slope.fill_(float(sol[0]))
                self.log_B_intercept.fill_(float(sol[1]))

        # Average per-partial params over all MIDI notes
        tau1_by_k  = [[] for _ in range(self.K)]
        tau2_by_k  = [[] for _ in range(self.K)]
        a1_by_k    = [[] for _ in range(self.K)]
        beat_by_k  = [[] for _ in range(self.K)]
        A0_by_k    = [[] for _ in range(self.K)]

        for midi, p in priors.items():
            for k_idx in range(self.K):
                if k_idx < len(p['tau1']) and p['tau1'][k_idx] > 0:
                    tau1_by_k[k_idx].append(p['tau1'][k_idx])
                if k_idx < len(p['tau2']) and p['tau2'][k_idx] > 0:
                    tau2_by_k[k_idx].append(p['tau2'][k_idx])
                if k_idx < len(p['a1']) and 0 < p['a1'][k_idx] < 1:
                    a1_by_k[k_idx].append(p['a1'][k_idx])
                if k_idx < len(p['beat_hz']) and p['beat_hz'][k_idx] > 0:
                    beat_by_k[k_idx].append(p['beat_hz'][k_idx])
                if k_idx < len(p['A0']) and p['A0'][k_idx] > 0:
                    A0_by_k[k_idx].append(p['A0'][k_idx])

        with torch.no_grad():
            for k_idx in range(self.K):
                if tau1_by_k[k_idx]:
                    v = _safe_mean(tau1_by_k[k_idx], 0.3)
                    self.log_tau1[k_idx] = math.log(max(v, 0.01))
                if tau2_by_k[k_idx]:
                    v = _safe_mean(tau2_by_k[k_idx], 3.0)
                    self.log_tau2[k_idx] = math.log(max(v, 0.05))
                if a1_by_k[k_idx]:
                    v = _safe_mean(a1_by_k[k_idx], 0.25)
                    v = max(0.02, min(0.98, v))
                    self.logit_a1[k_idx] = math.log(v / (1 - v))
                if beat_by_k[k_idx]:
                    v = _safe_mean(beat_by_k[k_idx], 0.5)
                    self.log_beat_hz[k_idx] = math.log(max(v, 0.05))
                # A0 spectrum from relative amplitudes (normalize to k=1)
                if A0_by_k[0] and A0_by_k[k_idx]:
                    A0_ref = _safe_mean(A0_by_k[0], 1.0)
                    A0_k   = _safe_mean(A0_by_k[k_idx], max(1.0 / (k_idx + 1), 1e-4))
                    ratio  = A0_k / max(A0_ref, 1e-10)
                    if ratio > 0:
                        self.log_A0[k_idx] = math.log(max(ratio, 1e-6))

        print(f"[PhysicsBank] Initialized from {params_path} ({len(priors)} MIDI notes)")

    # ── Physical parameter accessors ──────────────────────────────────────────

    def get_B(self, midi_norm: torch.Tensor) -> torch.Tensor:
        """B(midi_norm) = exp(slope*x + intercept), clamped to [1e-6, 1e-2]."""
        log_B = self.log_B_slope * midi_norm + self.log_B_intercept
        return torch.exp(log_B.clamp(-14.0, -4.6))

    def get_inharmonic_freqs(self, f0: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        f_k = k · f0 · √(1 + B·k²)
        Returns (B_batch, K)
        """
        K  = self.K
        ks = torch.arange(1, K + 1, device=f0.device, dtype=f0.dtype)
        f0_e = f0.unsqueeze(1)    # (B, 1)
        B_e  = B.unsqueeze(1)     # (B, 1)
        ks_e = ks.unsqueeze(0)    # (1, K)
        return f0_e * ks_e * torch.sqrt(1.0 + B_e * ks_e ** 2)  # (B, K)

    # ── Forward (vectorized) ──────────────────────────────────────────────────

    def forward(
        self,
        f0:       torch.Tensor,   # (B,)  Hz
        vel_norm: torch.Tensor,   # (B,)  [0, 1]
        n_frames: int,
        ctrl:     dict = None,    # optional per-frame controller corrections
    ) -> torch.Tensor:
        """
        Synthesize audio (fully vectorized, differentiable).

        Returns: (B, 1, T*S)  mono audio, S = frame_size
        """
        B_batch = f0.shape[0]
        device  = f0.device
        K       = self.K
        S       = self.frame_size
        T       = n_frames

        # ── Physical parameters ──────────────────────────────────────
        midi_norm = ((torch.log(f0.clamp(F0_LO, F0_HI)) - math.log(F0_LO)) /
                     (math.log(F0_HI) - math.log(F0_LO)))   # (B,)
        B_val = self.get_B(midi_norm)                         # (B,)

        # Inharmonic frequencies + per-partial detune
        f_k = self.get_inharmonic_freqs(f0, B_val)            # (B, K)
        detune = torch.tanh(self.harm_detune) * 0.005
        f_k = f_k * (1.0 + detune)                            # (B, K)

        # Decay
        tau1 = torch.exp(self.log_tau1).clamp(0.01, 5.0)      # (K,)
        tau2 = torch.exp(self.log_tau2).clamp(0.05, 60.0)     # (K,)
        a1   = torch.sigmoid(self.logit_a1)                    # (K,)
        a2   = 1.0 - a1

        # Beating offset
        beat_hz = torch.exp(self.log_beat_hz).clamp(0.05, 10.0)  # (K,)

        # Amplitude spectrum (normalized, velocity-dependent brightness)
        A0_raw = F.softmax(self.log_A0, dim=0) * K             # (K,)
        vel_   = vel_norm.unsqueeze(1)                          # (B, 1)
        k_norm = torch.arange(1, K + 1, device=device, dtype=f0.dtype) / K
        vel_bright = vel_ * k_norm                              # (B, K): emphasize high k at high vel
        A0 = A0_raw * (0.5 + 0.5 * (1.0 + vel_bright))         # (B, K)

        # ── Build time axis for ALL samples at once ──────────────────
        # t[i] = time of sample i  (0..T*S-1) / sr
        t_all = torch.arange(T * S, device=device, dtype=f0.dtype) / self.sr  # (T*S,)

        # ── Apply controller corrections (if provided) ────────────────
        # ctrl['delta_log_tau1']: (B, T, K) — per-frame tau1 correction
        # ctrl['delta_exc']:      (B, T, K) — per-frame amplitude injection
        # For simplicity: apply time-averaged correction from ctrl
        tau1_eff = tau1  # (K,)
        tau2_eff = tau2
        exc_injection = None
        if ctrl is not None:
            if 'delta_log_tau1' in ctrl:
                # (B, T, K) → average over T for now
                tau1_eff = tau1 * torch.exp(ctrl['delta_log_tau1'].mean(dim=1))  # (B, K)
            if 'delta_exc' in ctrl:
                exc_injection = F.softplus(ctrl['delta_exc'])  # (B, T, K)

        # ── Bi-exponential amplitude envelopes ───────────────────────
        # A(t) = a1 * exp(-t/tau1) + a2 * exp(-t/tau2)
        # Shape: (B, K, T*S) — one envelope per partial per batch

        # tau1_eff shape: (K,) or (B, K)
        if tau1_eff.dim() == 1:
            # (K,) → (1, K, 1) for broadcasting with t_all (T*S,)
            e_fast = torch.exp(-t_all.view(1, 1, -1) / tau1_eff.view(1, K, 1))  # (1, K, T*S)
            e_slow = torch.exp(-t_all.view(1, 1, -1) / tau2_eff.view(1, K, 1))  # (1, K, T*S)
            env = (a1.view(1, K, 1) * e_fast + a2.view(1, K, 1) * e_slow)       # (1, K, T*S)
            env = env * A0.unsqueeze(2)                                            # (B, K, T*S)
        else:
            # (B, K) tau1_eff
            e_fast = torch.exp(-t_all.view(1, 1, -1) / tau1_eff.unsqueeze(2))
            e_slow = torch.exp(-t_all.view(1, 1, -1) / tau2_eff.view(1, K, 1))
            env = a1.view(1, K, 1) * e_fast + a2.view(1, K, 1) * e_slow
            env = env * A0.unsqueeze(2)

        # Add excitation injection from controller (attack transient boost)
        if exc_injection is not None:
            # exc_injection: (B, T, K). Spread over frame samples with attack shape.
            # Attack window per frame: decays within the frame
            s_vec = torch.arange(S, device=device, dtype=f0.dtype) / S
            atk_win = torch.exp(-s_vec * 4.0)  # fast within-frame decay
            exc_frames = exc_injection.permute(0, 2, 1)  # (B, K, T)
            # Expand: (B, K, T*S)
            exc_expanded = (exc_frames.unsqueeze(-1) * atk_win.view(1, 1, 1, S)
                            ).reshape(B_batch, K, T * S)
            env = env + exc_expanded

        # ── Multi-string oscillators ──────────────────────────────────
        # String offsets: 1 string = [0], 2 strings = [-½, +½], 3 strings = [-1, 0, +1]
        # Offsets are in units of beat_hz — approximation (real pianos have asymmetric tuning)

        # We synthesize using 3 oscillators always; for monochord notes,
        # beat_hz is set to near 0 so all 3 collapse to same frequency.
        # This avoids branching and is differentiable.

        # String offsets (3 strings): -beat_hz, 0, +beat_hz
        beat_ = beat_hz.view(1, K, 1)  # (1, K, 1)

        # Phase: random per partial per batch
        # To make differentiable and batch-stable: use fixed phase offset per partial
        # (phase is not a trainable parameter — it's set at note onset)
        torch.manual_seed(42)  # reproducible phases for evaluation
        phi_a = torch.rand(B_batch, K, device=device) * 2 * math.pi
        phi_b = torch.rand(B_batch, K, device=device) * 2 * math.pi
        phi_c = torch.rand(B_batch, K, device=device) * 2 * math.pi

        t_ = t_all.view(1, 1, -1)          # (1, 1, T*S)
        fk_ = f_k.unsqueeze(2)             # (B, K, 1)

        osc_a = torch.cos(2 * math.pi * (fk_ - beat_) * t_ + phi_a.unsqueeze(2))
        osc_b = torch.cos(2 * math.pi * fk_ * t_              + phi_b.unsqueeze(2))
        osc_c = torch.cos(2 * math.pi * (fk_ + beat_) * t_ + phi_c.unsqueeze(2))
        osc   = (osc_a + osc_b + osc_c) / 3.0  # (B, K, T*S)

        # ── Sum partials ──────────────────────────────────────────────
        harmonic = (env * osc).sum(dim=1)   # (B, T*S)

        # ── Noise ─────────────────────────────────────────────────────
        noise_level = torch.exp(self.log_noise_level)
        tau_noise   = torch.exp(self.log_tau_noise).clamp(0.005, 1.0)
        noise_env   = torch.exp(-t_all / tau_noise)  # (T*S,)

        noise_raw = torch.randn(B_batch, T * S, device=device)
        # Simple IIR shaping (first-order, pole = noise_pole)
        pole = torch.sigmoid(self.noise_pole) * 0.99  # keep stable
        noise_shaped = self._apply_iir(noise_raw, pole)
        noise = noise_shaped * noise_env.unsqueeze(0) * noise_level

        audio = (harmonic + noise).unsqueeze(1)  # (B, 1, T*S)

        # ── Soundboard ────────────────────────────────────────────────
        strength = torch.sigmoid(self.soundboard_raw)  # (scalar)
        if strength > 0.01:
            audio = self._apply_soundboard(audio, strength)

        return audio   # (B, 1, T*S)

    def _apply_iir(self, x: torch.Tensor, pole: torch.Tensor) -> torch.Tensor:
        """
        Apply first-order IIR low-pass (approximate noise shaping) via recurrence.
        y[n] = (1-pole)*x[n] + pole*y[n-1]
        Differentiable through pole, but not through x samples (recurrence graph).
        """
        alpha = 1.0 - pole
        out = torch.zeros_like(x)
        prev = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(x.shape[1]):
            prev = alpha * x[:, i] + pole * prev
            out[:, i] = prev
        return out

    def _apply_soundboard(self, audio: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        """
        Apply parametric soundboard FIR convolution.
        audio: (B, 1, L)
        strength: scalar ∈ (0, 1)
        """
        B_batch, C, L = audio.shape
        ir = self.soundboard_ir  # (n_sb,)

        # Causal FIR convolution using F.conv1d
        # Pad left to maintain length (causal)
        pad = self.n_sb - 1
        audio_padded = F.pad(audio, (pad, 0))
        ir_weight = ir.view(1, 1, -1)  # (1, 1, n_sb)
        wet = F.conv1d(audio_padded, ir_weight, groups=1)  # (B, 1, L)

        return audio + strength * (wet - audio)

    # ── Convenience: synthesize single note ──────────────────────────────────

    @torch.no_grad()
    def synthesize(self, midi: int, vel_idx: int, duration: float = 4.0) -> torch.Tensor:
        """Synthesize a single note. Returns (1, n_samples) CPU float32."""
        self.eval()
        f0 = torch.tensor([440.0 * 2.0 ** ((midi - 69) / 12.0)])
        vel = torch.tensor([vel_idx / 7.0])
        n_frames = math.ceil(duration * self.sr / self.frame_size)
        audio, *_ = self.forward(f0, vel, n_frames) if False else (self.forward(f0, vel, n_frames),)
        return audio.squeeze(0).cpu()  # (1, n_samples)

    def synthesize_and_save(self, midi: int, vel_idx: int,
                            out_path: str, duration: float = 4.0):
        import soundfile as sf
        audio = self.synthesize(midi, vel_idx, duration)
        audio_np = audio.numpy().T  # (n_samples, 1)
        sf.write(out_path, audio_np, self.sr, subtype='PCM_16')
        print(f"[PhysicsBank] Saved {out_path}")
