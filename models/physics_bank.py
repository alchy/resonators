"""
models/physics_bank.py
──────────────────────
Physics-Informed Resonator Bank — Phase 1 redesign.

Architecture: additive synthesis with hard-coded physical structure,
learned physical parameters. No neural controller in this module.

Physical model:
  For each harmonic k = 1..K:
    f_k = k · f0 · √(1 + B · k²)                [inharmonicity]

    Two string oscillators per harmonic (beating pair):
      osc_a: f_k + Δf_k/2
      osc_b: f_k - Δf_k/2
    → amplitude modulation at Δf_k Hz (beating)

    Amplitude envelope (bi-exponential):
      A_k(t) = a1_k · exp(-t/τ1_k) + (1-a1_k) · exp(-t/τ2_k)
    → τ1 = fast (hammer/early), τ2 = slow (string natural)

  Noise:
    attack burst: N(0,1) filtered through 16-band shaping → exp(-t/τ_noise) envelope
    floor:        very low sustained noise

  Synthesizer state (per note, reset each clip):
    amp_fast[K]:   fast decay amplitudes
    amp_slow[K]:   slow decay amplitudes
    phase_a[K]:    phase of string-a oscillator
    phase_b[K]:    phase of string-b oscillator

Differentiability:
  All physical parameters (B, τ1, τ2, a1, Δf, A0_k) are nn.Parameters.
  Can be initialized from analytical extraction (params.json).
  Frame-by-frame synthesis is differentiable w.r.t. all parameters.

Can be used standalone (physics-only) or driven by a neural controller
that provides per-frame corrections to decay rates and excitation.
"""

import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

MIDI_LO  = 21    # A0
MIDI_HI  = 108   # C8
F0_LO    = 27.5  # Hz
F0_HI    = 4186. # Hz


# ── Helpers ───────────────────────────────────────────────────────────────────

def midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def inharmonic_freqs(f0: torch.Tensor, B: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute inharmonic partial frequencies.
    f0: (B_batch,), B_inharmon: (B_batch,)
    Returns: (B_batch, K)  — frequencies in Hz for k=1..K
    """
    ks = torch.arange(1, K + 1, device=f0.device, dtype=f0.dtype)  # (K,)
    f0_  = f0.unsqueeze(1)   # (B_batch, 1)
    B_   = B.unsqueeze(1)    # (B_batch, 1)
    ks_  = ks.unsqueeze(0)   # (1, K)
    return f0_ * ks_ * torch.sqrt(1.0 + B_ * ks_ ** 2)  # (B_batch, K)


# ── Parameter initialization from params.json ────────────────────────────────

def _load_piano_priors(params_path: str, n_harmonic: int, sr: int,
                       midi_lo: int = MIDI_LO, midi_hi: int = MIDI_HI
                       ) -> dict:
    """
    Load analytical priors from params.json.
    Returns dict of tensors indexed by MIDI note: {21: {...}, 22: {...}, ...}
    These become the initialization values for learnable parameters.
    """
    if not Path(params_path).exists():
        return {}

    with open(params_path) as f:
        data = json.load(f)

    samples = data.get('samples', {})
    priors = {}

    for midi in range(midi_lo, midi_hi + 1):
        # Average over all velocity layers for this MIDI note
        B_vals, tau1_k, tau2_k, a1_k, beat_hz_k, beat_depth_k = [], [], [], [], [], []
        A0_k_list = []

        found_any = False
        for vel in range(8):
            key = f"m{midi:03d}_vel{vel}"
            if key not in samples:
                continue
            s = samples[key]
            found_any = True
            if s['B'] > 0:
                B_vals.append(s['B'])

            partials = s.get('partials', [])
            for k_idx in range(n_harmonic):
                k = k_idx + 1
                # Find partial with matching k
                p = next((p for p in partials if p['k'] == k), None)

                # Initialize per-k lists if first velocity
                while len(tau1_k) <= k_idx:
                    tau1_k.append([])
                    tau2_k.append([])
                    a1_k.append([])
                    beat_hz_k.append([])
                    beat_depth_k.append([])
                    A0_k_list.append([])

                if p is not None:
                    if p.get('tau1') is not None:
                        tau1_k[k_idx].append(p['tau1'])
                    if p.get('tau2') is not None:
                        tau2_k[k_idx].append(p['tau2'])
                    if p.get('a1') is not None:
                        a1_k[k_idx].append(p['a1'])
                    if p.get('beat_hz') is not None and p['beat_hz'] > 0.05:
                        beat_hz_k[k_idx].append(p['beat_hz'])
                    if p.get('beat_depth') is not None:
                        beat_depth_k[k_idx].append(p['beat_depth'])
                    if p.get('A0') is not None and p['A0'] > 0:
                        A0_k_list[k_idx].append(p['A0'])

        if not found_any:
            continue

        # Average values
        B_mean = float(_safe_mean(B_vals, 1e-4))

        # Per-partial averages
        tau1_means  = [float(_safe_mean(v, 0.5)) for v in tau1_k]
        tau2_means  = [float(_safe_mean(v, 5.0)) for v in tau2_k]
        a1_means    = [float(_safe_mean(v, 0.3)) for v in a1_k]
        beat_means  = [float(_safe_mean(v, 0.5)) for v in beat_hz_k]
        depth_means = [float(_safe_mean(v, 0.1)) for v in beat_depth_k]
        A0_means    = [float(_safe_mean(v, 0.1 / (k + 1))) for k, v in enumerate(A0_k_list)]

        priors[midi] = {
            'B': B_mean,
            'tau1': tau1_means,
            'tau2': tau2_means,
            'a1': a1_means,
            'beat_hz': beat_means,
            'beat_depth': depth_means,
            'A0': A0_means,
        }

    return priors


def _safe_mean(vals: list, default: float) -> float:
    vals = [v for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)]
    return sum(vals) / len(vals) if vals else default


# ── Physics Bank ──────────────────────────────────────────────────────────────

class PhysicsResonatorBank(nn.Module):
    """
    Differentiable physics-informed piano synthesizer.

    Parameters are either:
      a) Initialized from params.json analytical extraction, then fine-tuned by gradient
      b) Initialized from physics priors and learned from scratch

    Forward signature (single-instrument mode):
      audio = bank.forward(f0, vel_norm, n_frames, state=None)

    Forward with controller corrections:
      audio = bank.forward(f0, vel_norm, n_frames, ctrl_dict, state=None)
      ctrl_dict keys: 'delta_log_tau1', 'delta_log_tau2', 'delta_exc', 'delta_beat'
    """

    def __init__(self, cfg: dict, params_path: str = None):
        super().__init__()

        pc = cfg.get('physics', {})
        self.K          = int(pc.get('n_harmonic', 80))        # harmonic count
        self.sr         = int(cfg.get('sample_rate', 44100))
        self.frame_size = int(cfg.get('frame_size', 256))
        self.dt         = self.frame_size / self.sr

        # ── Inharmonicity ─────────────────────────────────────────────
        # B(midi) = exp(log_B_slope * midi_norm + log_B_intercept)
        # Initialized from data or from physics prior (bass high, treble low)
        self.log_B_slope     = nn.Parameter(torch.tensor(-2.0))   # B decreases with midi
        self.log_B_intercept = nn.Parameter(torch.tensor(-7.5))   # log(B) at MIDI 21
        # B_lo (bass A0): exp(-7.5) ≈ 0.00055, B_hi (treble C8): exp(-7.5 + (-2)*(87/87)) ≈ 0.000075

        # ── Per-partial parameters (averaged over keyboard) ───────────
        # These are learned globally but can be conditioned on MIDI via the controller.

        # log τ1: fast decay (hammer/early), init ~0.3s
        self.log_tau1 = nn.Parameter(
            torch.linspace(math.log(0.4), math.log(0.05), self.K)
        )
        # log τ2: slow decay (string), init ~4s decreasing with harmonic number
        self.log_tau2 = nn.Parameter(
            torch.linspace(math.log(5.0), math.log(0.3), self.K)
        )
        # a1: mixing ratio (fast component), init 0.2 for low k, 0.5 for high k
        self.logit_a1 = nn.Parameter(
            torch.linspace(-1.4, 0.0, self.K)  # sigmoid → 0.20..0.50
        )

        # ── Beating (string coupling) ─────────────────────────────────
        # log Δf: beating frequency in Hz, init ~0.3 Hz for bass, ~1 Hz for treble
        self.log_beat_hz = nn.Parameter(
            torch.linspace(math.log(0.3), math.log(1.5), self.K)
        )
        # beat_depth: modulation index, init 0.15
        self.logit_beat_depth = nn.Parameter(
            torch.full((self.K,), -1.7)  # sigmoid → ~0.15
        )

        # ── Amplitude spectrum ────────────────────────────────────────
        # log A0[k]: relative amplitude of harmonic k (normalized so k=1 ≈ 1.0)
        # Init: roughly 1/k spectrum (bright piano)
        self.log_A0 = nn.Parameter(
            -torch.log(torch.arange(1, self.K + 1).float()) * 0.7
        )

        # ── Per-partial systematic tuning deviation ───────────────────
        # Small per-partial frequency correction (±0.5%)
        self.harm_detune = nn.Parameter(torch.zeros(self.K))

        # ── Noise model ───────────────────────────────────────────────
        self.n_noise_bands = int(pc.get('n_noise_bands', 16))
        # Log-spaced band frequencies (Hz)
        self.register_buffer(
            'noise_freqs',
            torch.exp(torch.linspace(math.log(200), math.log(20000), self.n_noise_bands))
        )
        # Per-band gain (learnable)
        self.noise_band_gain = nn.Parameter(torch.zeros(self.n_noise_bands))
        # Attack noise decay time (log τ_noise)
        self.log_tau_noise = nn.Parameter(torch.tensor(math.log(0.05)))
        # Noise level (relative to harmonic)
        self.log_noise_level = nn.Parameter(torch.tensor(math.log(0.08)))

        # ── Stereo panning ────────────────────────────────────────────
        # Pan per partial: log(stereo_width), simple linear spread
        pan_norm = (torch.arange(self.K).float() / max(self.K - 1, 1)) * 2 - 1  # -1..+1
        self.register_buffer('pan_init', pan_norm * 0.1)  # subtle spread
        self.pan_scale = nn.Parameter(torch.tensor(0.0))  # learned width

        # ── Load priors from params.json ──────────────────────────────
        if params_path is not None and Path(params_path).exists():
            self._init_from_params(params_path)

    def _init_from_params(self, params_path: str):
        """Initialize learnable parameters from analytically extracted priors."""
        priors = _load_piano_priors(params_path, self.K, self.sr)
        if not priors:
            return

        # Fit B(midi) log-linear model
        midis = sorted(priors.keys())
        B_vals = [priors[m]['B'] for m in midis]
        B_valid = [(m, b) for m, b in zip(midis, B_vals) if b > 0]
        if len(B_valid) >= 4:
            ms = torch.tensor([m for m, _ in B_valid], dtype=torch.float32)
            bs = torch.tensor([math.log(b) for _, b in B_valid], dtype=torch.float32)
            ms_norm = (ms - MIDI_LO) / (MIDI_HI - MIDI_LO)
            # log(B) = slope * midi_norm + intercept
            A = torch.stack([ms_norm, torch.ones_like(ms_norm)], dim=1)
            # Least squares
            coef, _ = torch.lstsq(bs.unsqueeze(1), A) if hasattr(torch, 'lstsq') else (None, None)
            if coef is not None:
                with torch.no_grad():
                    self.log_B_slope.fill_(float(coef[0]))
                    self.log_B_intercept.fill_(float(coef[1]))

        # Average per-partial params over all MIDI notes (weighted by amp)
        tau1_agg  = [[] for _ in range(self.K)]
        tau2_agg  = [[] for _ in range(self.K)]
        a1_agg    = [[] for _ in range(self.K)]
        beat_agg  = [[] for _ in range(self.K)]
        depth_agg = [[] for _ in range(self.K)]
        A0_agg    = [[] for _ in range(self.K)]

        for midi, p in priors.items():
            for k_idx in range(min(self.K, len(p['tau1']))):
                if p['tau1'][k_idx] > 0:
                    tau1_agg[k_idx].append(p['tau1'][k_idx])
                if len(p['tau2']) > k_idx and p['tau2'][k_idx] > 0:
                    tau2_agg[k_idx].append(p['tau2'][k_idx])
                if len(p['a1']) > k_idx and 0 < p['a1'][k_idx] < 1:
                    a1_agg[k_idx].append(p['a1'][k_idx])
                if len(p['beat_hz']) > k_idx and p['beat_hz'][k_idx] > 0:
                    beat_agg[k_idx].append(p['beat_hz'][k_idx])
                if len(p['beat_depth']) > k_idx and p['beat_depth'][k_idx] > 0:
                    depth_agg[k_idx].append(p['beat_depth'][k_idx])
                if len(p['A0']) > k_idx and p['A0'][k_idx] > 0:
                    A0_agg[k_idx].append(p['A0'][k_idx])

        with torch.no_grad():
            for k_idx in range(self.K):
                if tau1_agg[k_idx]:
                    v = _safe_mean(tau1_agg[k_idx], 0.3)
                    self.log_tau1[k_idx] = math.log(max(v, 0.01))
                if tau2_agg[k_idx]:
                    v = _safe_mean(tau2_agg[k_idx], 3.0)
                    self.log_tau2[k_idx] = math.log(max(v, 0.05))
                if a1_agg[k_idx]:
                    v = _safe_mean(a1_agg[k_idx], 0.25)
                    v = max(0.01, min(0.99, v))
                    self.logit_a1[k_idx] = math.log(v / (1 - v))
                if beat_agg[k_idx]:
                    v = _safe_mean(beat_agg[k_idx], 0.5)
                    self.log_beat_hz[k_idx] = math.log(max(v, 0.05))
                if depth_agg[k_idx]:
                    v = _safe_mean(depth_agg[k_idx], 0.15)
                    v = max(0.01, min(0.99, v))
                    self.logit_beat_depth[k_idx] = math.log(v / (1 - v))

        print(f"[PhysicsBank] Initialized from {params_path} ({len(priors)} MIDI notes)")

    # ── Inharmonicity ────────────────────────────────────────────────────────

    def get_B(self, midi_norm: torch.Tensor) -> torch.Tensor:
        """B(midi_norm) = exp(slope * midi_norm + intercept). midi_norm in [0,1]."""
        log_B = self.log_B_slope * midi_norm + self.log_B_intercept
        return torch.exp(log_B.clamp(-14, -4))  # B in [1e-6, 1e-2]

    # ── Synthesis ────────────────────────────────────────────────────────────

    def forward(
        self,
        f0:        torch.Tensor,  # (B,) fundamental Hz
        vel_norm:  torch.Tensor,  # (B,) velocity in [0, 1]
        n_frames:  int,
        ctrl:      dict = None,   # optional controller corrections
        state:     dict = None,   # optional state dict for stateful synthesis
    ) -> tuple[torch.Tensor, dict]:
        """
        Synthesize audio.

        Returns:
            audio: (B, 2, n_frames * frame_size)  stereo
            state: dict with final resonator state (for continuation)
        """
        B_batch = f0.shape[0]
        device  = f0.device
        T       = n_frames
        S       = self.frame_size
        K       = self.K

        # ── Physical parameters ──────────────────────────────────────
        midi_norm = (torch.log(f0.clamp(F0_LO, F0_HI)) - math.log(F0_LO)) / \
                    (math.log(F0_HI) - math.log(F0_LO))   # (B,)

        B_val = self.get_B(midi_norm)                      # (B,)
        f_k   = inharmonic_freqs(f0, B_val, K)             # (B, K)

        # Apply per-partial systematic tuning (±0.5%)
        detune = torch.tanh(self.harm_detune) * 0.005      # (K,)
        f_k    = f_k * (1.0 + detune.unsqueeze(0))         # (B, K)

        # Decay parameters
        tau1  = torch.exp(self.log_tau1).clamp(0.01, 5.0)    # (K,)
        tau2  = torch.exp(self.log_tau2).clamp(0.05, 60.0)   # (K,)
        a1    = torch.sigmoid(self.logit_a1)                  # (K,)
        a2    = 1.0 - a1                                      # (K,)

        # Beating
        beat_hz    = torch.exp(self.log_beat_hz).clamp(0.05, 10.0)  # (K,)
        beat_depth = torch.sigmoid(self.logit_beat_depth)            # (K,)

        # Amplitude spectrum
        A0 = torch.softmax(self.log_A0, dim=0) * K       # (K,) normalized, sum=K
        # Velocity scaling: louder → brighter (emphasize high k)
        vel_bright = vel_norm.unsqueeze(1) * torch.arange(1, K + 1, device=device, dtype=torch.float32).unsqueeze(0) / K
        A0 = A0.unsqueeze(0) * (0.3 + 0.7 * (1.0 + vel_bright))  # (B, K)

        # ── Initialize state ─────────────────────────────────────────
        if state is None:
            # amp_fast[k] = a1[k] * A0[k]  (fast decay component)
            amp_fast  = a1.unsqueeze(0) * A0                  # (B, K)
            amp_slow  = a2.unsqueeze(0) * A0                  # (B, K)
            # Phase: random initial phase per partial per string
            phase_a   = torch.rand(B_batch, K, device=device) * 2 * math.pi
            phase_b   = torch.rand(B_batch, K, device=device) * 2 * math.pi
        else:
            amp_fast = state['amp_fast']
            amp_slow = state['amp_slow']
            phase_a  = state['phase_a']
            phase_b  = state['phase_b']

        # ── Decay rates per dt ────────────────────────────────────────
        decay_fast = torch.exp(-self.dt / tau1)  # (K,) frame decay factor
        decay_slow = torch.exp(-self.dt / tau2)  # (K,) frame decay factor

        # ── Frequencies for both string oscillators ───────────────────
        # String a: f_k + beat_hz/2
        # String b: f_k - beat_hz/2
        beat_offset = beat_hz.unsqueeze(0) / 2.0   # (1, K)
        f_a = f_k + beat_offset                     # (B, K) — string a
        f_b = f_k - beat_offset                     # (B, K) — string b

        # Phase advance per sample
        dphi_a = 2.0 * math.pi * f_a / self.sr    # (B, K)
        dphi_b = 2.0 * math.pi * f_b / self.sr    # (B, K)

        # ── Panning ──────────────────────────────────────────────────
        pan = self.pan_init + self.pan_scale * self.pan_init  # (K,)
        pan = pan.clamp(-1, 1)
        gain_L = torch.sqrt((1 - pan) / 2.0).unsqueeze(0)  # (1, K)
        gain_R = torch.sqrt((1 + pan) / 2.0).unsqueeze(0)  # (1, K)

        # ── Frame-by-frame synthesis ──────────────────────────────────
        # Precompute all frames at once using vector operations
        # (more efficient than Python loop)

        # Build sample time array for all frames
        # t_rel[f, s] = relative time of sample s in frame f
        t_samples = torch.arange(S, device=device, dtype=torch.float32)  # (S,)

        audio_L = torch.zeros(B_batch, T * S, device=device)
        audio_R = torch.zeros(B_batch, T * S, device=device)

        for f in range(T):
            # ── Apply controller corrections (if any) ────────────────
            if ctrl is not None:
                # ctrl['delta_log_tau1']: (B, K) additive correction to log_tau1
                # ctrl['delta_exc']:      (B, K) excitation amplitude injection
                if 'delta_log_tau1' in ctrl:
                    tau1_eff = tau1 * torch.exp(ctrl['delta_log_tau1'][:, f] if ctrl['delta_log_tau1'].dim() == 3 else ctrl['delta_log_tau1'])
                    decay_fast_f = torch.exp(-self.dt / tau1_eff.clamp(0.005, 5.0))
                else:
                    decay_fast_f = decay_fast  # (K,)

                if 'delta_exc' in ctrl:
                    exc = ctrl['delta_exc'][:, f] if ctrl['delta_exc'].dim() == 3 else ctrl['delta_exc']
                    amp_fast = amp_fast + F.softplus(exc) * a1.unsqueeze(0)
                    amp_slow = amp_slow + F.softplus(exc) * a2.unsqueeze(0)
            else:
                decay_fast_f = decay_fast

            # ── Synthesize current frame ─────────────────────────────
            # Total amplitude per partial
            amp = amp_fast + amp_slow  # (B, K)

            # Phase at start of frame for string a and b
            # phase_a, phase_b: (B, K)
            # Phase trajectory across frame samples:
            # phi_a(s) = phase_a + dphi_a * s  for s in 0..S-1
            phi_a = phase_a.unsqueeze(2) + dphi_a.unsqueeze(2) * t_samples.unsqueeze(0).unsqueeze(0)  # (B, K, S)
            phi_b = phase_b.unsqueeze(2) + dphi_b.unsqueeze(2) * t_samples.unsqueeze(0).unsqueeze(0)  # (B, K, S)

            # Oscillator signals
            osc_a = torch.cos(phi_a)  # (B, K, S)
            osc_b = torch.cos(phi_b)  # (B, K, S)
            osc   = (osc_a + osc_b) * 0.5  # (B, K, S) — average (beating naturally emerges)

            # Apply amplitude and panning
            amp_ = amp.unsqueeze(2)       # (B, K, 1)
            frame_L = (amp_ * gain_L.unsqueeze(2) * osc).sum(dim=1)  # (B, S)
            frame_R = (amp_ * gain_R.unsqueeze(2) * osc).sum(dim=1)  # (B, S)

            audio_L[:, f * S: (f + 1) * S] = frame_L
            audio_R[:, f * S: (f + 1) * S] = frame_R

            # ── Update state ─────────────────────────────────────────
            amp_fast = amp_fast * decay_fast_f  # (B, K)
            amp_slow = amp_slow * decay_slow    # (B, K)
            phase_a  = (phase_a + dphi_a * S) % (2 * math.pi)
            phase_b  = (phase_b + dphi_b * S) % (2 * math.pi)

        # ── Add noise ────────────────────────────────────────────────
        noise_audio = self._synthesize_noise(B_batch, T * S, device)
        audio_L = audio_L + noise_audio
        audio_R = audio_R + noise_audio

        # ── Stack stereo ─────────────────────────────────────────────
        audio = torch.stack([audio_L, audio_R], dim=1)  # (B, 2, T*S)

        new_state = {
            'amp_fast': amp_fast,
            'amp_slow': amp_slow,
            'phase_a':  phase_a,
            'phase_b':  phase_b,
        }
        return audio, new_state

    def _synthesize_noise(self, B_batch: int, n_samples: int, device) -> torch.Tensor:
        """
        Shaped attack noise: broadband burst with exponential decay.
        Returns (B_batch, n_samples) noise signal.
        """
        t = torch.arange(n_samples, device=device, dtype=torch.float32) / self.sr

        # Attack decay envelope
        tau_noise = torch.exp(self.log_tau_noise).clamp(0.005, 1.0)
        noise_env = torch.exp(-t / tau_noise)  # (n_samples,)

        # White noise
        noise_raw = torch.randn(B_batch, n_samples, device=device)

        # Simple IIR shaping per band (approximated as weighted sum of low-passes)
        # Apply single first-order IIR as primary shaping for efficiency
        # Real noise shaping would use a filterbank, but this is good enough
        noise_level = torch.exp(self.log_noise_level)
        noise = noise_raw * noise_env.unsqueeze(0) * noise_level

        return noise

    # ── Synthesis with full audio (for evaluation) ────────────────────────────

    def synthesize(self, midi: int, vel_idx: int,
                   duration: float = 4.0) -> torch.Tensor:
        """
        Convenience method: synthesize a single note and return CPU audio tensor.
        Returns (2, n_samples) stereo float32.
        """
        f0 = torch.tensor([midi_to_hz(torch.tensor(float(midi)))]).float()
        vel_norm = torch.tensor([vel_idx / 7.0]).float()
        n_frames = math.ceil(duration * self.sr / self.frame_size)

        self.eval()
        with torch.no_grad():
            audio, _ = self.forward(f0, vel_norm, n_frames)
        return audio.squeeze(0)  # (2, n_samples)
