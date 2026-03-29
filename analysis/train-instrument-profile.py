"""
analysis/train-instrument-profile.py
──────────────────────────────────────
Phase 3 (pipeline step 3): learn a smooth instrument profile from raw extracted params.

A small factorised neural network learns the piano's physical parameter
landscape from all available (midi, vel) samples — handling noisy extractions,
missing notes, and inter-velocity inconsistencies.

Architecture (physically motivated factorisation):
  B_net    : MLP(midi)           → B          (inharmonicity, vel-independent)
  tau_net  : MLP(midi, k)        → tau1, tau2  (string decay, vel-independent)
  A0_net   : MLP(midi, k, vel)   → A0_ratio    (spectral shape, vel-dependent)
  df_net   : MLP(midi)           → df          (beating, vel-independent)
  dur_net  : MLP(midi)           → duration_s
  eq_net   : MLP(midi, freq)     → gain_db     (body EQ, vel-independent)

All positive outputs trained in log-space (MSE on log values → geometric error).

Output files:
  --out   analysis/params-nn-profile-{bank}.json   full 88×8 params for GUI/synthesiser
  --model analysis/profile.pt                      model weights + metadata (reusable)

Log:  runtime-logs/train-profile-log.txt  (auto-created, tee of stdout)

Usage:
    python -u analysis/train-instrument-profile.py \\
        --in    analysis/params-ks-grand.json \\
        --out   analysis/params-nn-profile-ks-grand.json \\
        --model analysis/profile.pt \\
        --epochs 800 --hidden 64 --lr 0.003 \\
        [--midi-from 21] [--midi-to 108] [--sr 44100] \\
        [--no-preserve-orig] [--plot]

Arguments:
  --in              Input extracted params JSON
  --out             Output smoothed params JSON (for GUI / generate-samples.py)
  --model           Output PyTorch model weights (default: analysis/profile.pt)
  --epochs          Training epochs (default: 800)
  --hidden          MLP hidden size (default: 64)
  --lr              Learning rate (default: 0.003)
  --midi-from/to    MIDI range for profile output (default: 21–108)
  --sr              Sample rate written into output JSON (default: 44100)
  --no-preserve-orig  Overwrite extracted values with NN predictions (default: keep)
  --plot            Show training loss + parameter curves
"""

import argparse
import json
import math
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Runtime logging (tee stdout → runtime-logs/train-profile-log.txt) ────────

def _setup_log() -> None:
    log_dir = Path("runtime-logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "train-profile-log.txt"

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()

    sys.stdout = _Tee(sys.__stdout__, open(log_path, "w", encoding="utf-8", buffering=1))


# ── helpers ──────────────────────────────────────────────────────────────────

def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def midi_feat(midi: float) -> torch.Tensor:
    """Normalised MIDI feature + sinusoidal embedding for register awareness."""
    m = (midi - 21) / 87.0          # 0..1
    # Sinusoidal at multiple scales captures register transitions
    return torch.tensor([
        m,
        math.sin(math.pi * m),
        math.sin(2 * math.pi * m),
        math.sin(4 * math.pi * m),
        math.cos(math.pi * m),
        math.cos(2 * math.pi * m),
    ], dtype=torch.float32)


def vel_feat(vel: int) -> torch.Tensor:
    v = vel / 7.0
    return torch.tensor([v, v ** 0.5, v ** 2.0], dtype=torch.float32)


def k_feat(k: int, k_max: int = 90) -> torch.Tensor:
    kn = (k - 1) / (k_max - 1)
    return torch.tensor([
        kn,
        math.log(k) / math.log(k_max),
        1.0 / k,
    ], dtype=torch.float32)


def freq_feat(freq_hz: float, sr: int = 44100) -> torch.Tensor:
    fn = math.log(max(freq_hz, 10.0)) / math.log(sr / 2)
    return torch.tensor([fn, fn ** 2], dtype=torch.float32)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.SiLU())
    return nn.Sequential(*mods)


# ── Network ──────────────────────────────────────────────────────────────────

MIDI_DIM = 6   # midi_feat output dim
VEL_DIM  = 3
K_DIM    = 3
FREQ_DIM = 2


class InstrumentProfile(nn.Module):
    """
    Factorised network: separate sub-networks for vel-independent
    and vel-dependent parameters.

    tau1_k1_net: fundamental (k=1) decay time — dedicated to avoid bias from
      the many short-tau high-k partials.
    noise_net: attack noise model (amplitude, decay, brightness) — vel-dependent
      because hammer energy and impact character change with velocity.
    biexp_net: bi-exponential decay coefficients (a1, tau2/tau1) — per-partial,
      vel-dependent because soft vs. loud strokes have different initial transient.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.B_net         = mlp(MIDI_DIM, hidden, 1)                        # log(B)
        self.dur_net       = mlp(MIDI_DIM, hidden, 1)                        # log(dur)
        self.tau1_k1_net   = mlp(MIDI_DIM + VEL_DIM, hidden, 1)             # log(tau1) for k=1
        self.tau_ratio_net = mlp(MIDI_DIM + K_DIM, hidden, 1)               # log(tau_k / tau_k1)
        self.A0_net        = mlp(MIDI_DIM + K_DIM + VEL_DIM, hidden, 1)     # log(A0_ratio)
        self.df_net        = mlp(MIDI_DIM + K_DIM, hidden, 1)               # log(df)
        self.eq_net        = mlp(MIDI_DIM + FREQ_DIM, hidden, 1)            # gain_db
        self.wf_net        = mlp(MIDI_DIM, hidden, 1)                       # log(width_factor)
        # noise_net → [log(attack_tau_s), log(centroid_hz), log(A_noise)]
        self.noise_net     = mlp(MIDI_DIM + VEL_DIM, hidden, 3)
        # biexp_net → [logit(a1), log(tau2/tau1)]
        self.biexp_net     = mlp(MIDI_DIM + K_DIM + VEL_DIM, hidden, 2)

        # Bias B_net final layer to log(1e-4) ≈ -9.2 so initial B is in piano range.
        nn.init.constant_(self.B_net[-1].bias, -9.2)

        # Noise net biases: attack_tau ≈ 0.05s → log=-3.0; centroid ≈ 3000 Hz → log=8.0;
        # A_noise ≈ 0.06 → log=-2.8
        nn.init.constant_(self.noise_net[-1].bias[0], -3.0)
        nn.init.constant_(self.noise_net[-1].bias[1],  8.0)
        nn.init.constant_(self.noise_net[-1].bias[2], -2.8)

        # biexp_net biases: a1 → logit(0.85) ≈ 1.73 (mostly fast decay);
        # log(tau2/tau1) → log(3) ≈ 1.1 (tau2 ≈ 3×tau1)
        nn.init.constant_(self.biexp_net[-1].bias[0],  1.73)
        nn.init.constant_(self.biexp_net[-1].bias[1],  1.10)

    def forward_B(self, mf):                  return self.B_net(mf)
    def forward_dur(self, mf):                return self.dur_net(mf)
    def forward_tau1_k1(self, mf, vf):        return self.tau1_k1_net(torch.cat([mf, vf], -1))
    def forward_tau_ratio(self, mf, kf):      return self.tau_ratio_net(torch.cat([mf, kf], -1))
    def forward_A0(self, mf, kf, vf):         return self.A0_net(torch.cat([mf, kf, vf], -1))
    def forward_df(self, mf, kf):             return self.df_net(torch.cat([mf, kf], -1))
    def forward_eq(self, mf, ff):             return self.eq_net(torch.cat([mf, ff], -1))
    def forward_wf(self, mf):                 return self.wf_net(mf)
    def forward_noise(self, mf, vf):          return self.noise_net(torch.cat([mf, vf], -1))
    def forward_biexp(self, mf, kf, vf):      return self.biexp_net(torch.cat([mf, kf, vf], -1))


# ── Data extraction ───────────────────────────────────────────────────────────

def build_dataset(samples: dict) -> dict:
    """Extract training tensors from raw params dict."""
    B_data, dur_data, wf_data = [], [], []
    tau_data, tau1_k1_data, A0_data, df_data, eq_data = [], [], [], [], []
    noise_data, biexp_data = [], []

    # Common EQ freq grid
    eq_freqs = None
    for s in samples.values():
        eq = s.get("spectral_eq") or {}
        if eq.get("freqs_hz"):
            eq_freqs = np.array(eq["freqs_hz"])
            break

    for key, s in samples.items():
        if s.get("_interpolated"):
            continue   # skip previously interpolated entries
        midi = s.get("midi")
        vel  = s.get("vel")
        if midi is None or vel is None:
            continue

        mf = midi_feat(midi)
        vf = vel_feat(vel)

        # B
        B = s.get("B") or 0
        if B > 1e-7:
            B_data.append((mf, math.log(B)))

        # duration
        dur = s.get("duration_s") or 0
        if dur > 0.1:
            dur_data.append((mf, math.log(dur)))

        # stereo width
        eq = s.get("spectral_eq") or {}
        wf = eq.get("stereo_width_factor") or 0
        if wf > 0.1:
            wf_data.append((mf, math.log(wf)))

        # spectral EQ
        if eq_freqs is not None and eq.get("gains_db"):
            gd = np.array(eq["gains_db"])
            if len(gd) == len(eq_freqs):
                for fi, (fhz, g) in enumerate(zip(eq_freqs, gd)):
                    ff = freq_feat(fhz)
                    eq_data.append((mf, ff, float(g)))

        # noise model: attack_tau_s, centroid_hz, A_noise
        noise = s.get("noise") or {}
        atk_tau = noise.get("attack_tau_s") or 0
        centroid = noise.get("centroid_hz") or 0
        A_noise  = noise.get("A_noise") or 0
        if atk_tau > 0.001 and centroid > 50:
            if A_noise < 0.001:
                A_noise = 0.06  # fall back to physical default if not stored
            noise_data.append((mf, vf,
                                math.log(max(atk_tau, 1e-4)),
                                math.log(max(centroid, 10.0)),
                                math.log(max(A_noise, 1e-4))))

        # per-partial
        parts = {p["k"]: p for p in s.get("partials", []) if "k" in p}
        a0_k1 = parts.get(1, {}).get("A0") or 0

        # k=1 tau1 needed for ratio computation
        tau1_k1_val = parts.get(1, {}).get("tau1") or 0

        for k, p in parts.items():
            kf = k_feat(k)

            t1 = p.get("tau1") or 0

            # k=1 sustain: dedicated dataset for tau1_k1_net
            if k == 1 and t1 > 0.005:
                tau1_k1_data.append((mf, vf, math.log(t1)))

            # tau ratio for k=2..10: log(tau_k / tau_k1)
            # vel-independent ratio captures physical decay structure
            if 2 <= k <= 10 and t1 > 0.005 and tau1_k1_val > 0.005:
                ratio = t1 / tau1_k1_val
                if 1e-4 < ratio < 100:
                    # Cap target to 0.5 (exp(0.5)=1.65x max) — extreme ratios are artifacts
                    log_r = min(0.5, math.log(ratio))
                    tau_data.append((mf, kf, log_r))

            # A0 ratio (k >= 1, normalised to k=1)
            a0 = p.get("A0") or 0
            if a0 > 0 and a0_k1 > 0:
                ratio = a0 / a0_k1
                if ratio > 1e-5:
                    A0_data.append((mf, kf, vf, math.log(ratio)))

            # beating (stored as beat_hz in params.json)
            df = p.get("beat_hz") or p.get("df") or 0
            if df > 0.001:
                df_data.append((mf, kf, math.log(df)))

            # bi-exponential decay: a1 and tau2/tau1 ratio
            a1_val  = p.get("a1")
            tau2_val = p.get("tau2")
            if (a1_val is not None and tau2_val is not None
                    and 0.01 < a1_val < 0.99 and t1 > 0.005
                    and tau2_val > t1 * 1.1):
                logit_a1 = math.log(a1_val / (1.0 - a1_val))
                log_ratio = math.log(tau2_val / t1)
                biexp_data.append((mf, kf, vf, logit_a1, log_ratio))

    # ── Outlier filtering before batching ─────────────────────────────────────
    def iqr_filter_list(items, val_idx, k_iqr=3.0):
        """Remove items where value at val_idx is an outlier (IQR method)."""
        vals = np.array([x[val_idx] for x in items], dtype=float)
        if len(vals) < 4:
            return items
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q75 - q25
        if iqr < 1e-12:
            return items
        med = np.median(vals)
        return [x for x, v in zip(items, vals) if abs(v - med) <= k_iqr * iqr]

    B_data   = iqr_filter_list(B_data,   1)
    dur_data = iqr_filter_list(dur_data, 1)
    wf_data  = iqr_filter_list(wf_data,  1)
    # tau ratios: use tight IQR (1.5x) — extreme ratio outliers are extraction artifacts
    tau_data      = iqr_filter_list(tau_data,      2, k_iqr=1.5)
    tau1_k1_data  = iqr_filter_list(tau1_k1_data,  2)
    A0_data       = iqr_filter_list(A0_data,        3)
    df_data       = iqr_filter_list(df_data,        2)
    noise_data    = iqr_filter_list(noise_data,     2)  # filter on log(attack_tau)
    biexp_data    = iqr_filter_list(biexp_data,     3)  # filter on logit(a1)

    # Pre-stack into batch tensors for fast training
    batches = {}

    # B: [N, midi_dim], [N]
    if B_data:
        batches["B_mf"] = torch.stack([d[0] for d in B_data])
        batches["B_y"]  = torch.tensor([d[1] for d in B_data], dtype=torch.float32)

    if dur_data:
        batches["dur_mf"] = torch.stack([d[0] for d in dur_data])
        batches["dur_y"]  = torch.tensor([d[1] for d in dur_data], dtype=torch.float32)

    if wf_data:
        batches["wf_mf"] = torch.stack([d[0] for d in wf_data])
        batches["wf_y"]  = torch.tensor([d[1] for d in wf_data], dtype=torch.float32)

    if tau1_k1_data:
        batches["tk1_mf"] = torch.stack([d[0] for d in tau1_k1_data])
        batches["tk1_vf"] = torch.stack([d[1] for d in tau1_k1_data])
        batches["tk1_y"]  = torch.tensor([d[2] for d in tau1_k1_data], dtype=torch.float32)

    if tau_data:
        batches["tau_mf"] = torch.stack([d[0] for d in tau_data])
        batches["tau_kf"] = torch.stack([d[1] for d in tau_data])
        batches["tau_y"]  = torch.tensor([d[2] for d in tau_data], dtype=torch.float32)
        # weight: down-weight noisier high-k partials slightly
        batches["tau_w"]  = torch.tensor(
            [1.0 / (1.0 + float(d[1][0]) * 2) for d in tau_data], dtype=torch.float32)

    if A0_data:
        batches["a0_mf"] = torch.stack([d[0] for d in A0_data])
        batches["a0_kf"] = torch.stack([d[1] for d in A0_data])
        batches["a0_vf"] = torch.stack([d[2] for d in A0_data])
        batches["a0_y"]  = torch.tensor([d[3] for d in A0_data], dtype=torch.float32)

    if df_data:
        batches["df_mf"] = torch.stack([d[0] for d in df_data])
        batches["df_kf"] = torch.stack([d[1] for d in df_data])
        batches["df_y"]  = torch.tensor([d[2] for d in df_data], dtype=torch.float32)

    # EQ: subsample
    eq_sub = eq_data[::4] if len(eq_data) > 400 else eq_data
    if eq_sub:
        batches["eq_mf"] = torch.stack([d[0] for d in eq_sub])
        batches["eq_ff"] = torch.stack([d[1] for d in eq_sub])
        batches["eq_y"]  = torch.tensor([d[2] for d in eq_sub], dtype=torch.float32)

    if noise_data:
        batches["noise_mf"] = torch.stack([d[0] for d in noise_data])
        batches["noise_vf"] = torch.stack([d[1] for d in noise_data])
        # targets: [log_tau, log_centroid, log_A_noise]
        batches["noise_y"] = torch.tensor(
            [[d[2], d[3], d[4]] for d in noise_data], dtype=torch.float32)

    if biexp_data:
        batches["biexp_mf"] = torch.stack([d[0] for d in biexp_data])
        batches["biexp_kf"] = torch.stack([d[1] for d in biexp_data])
        batches["biexp_vf"] = torch.stack([d[2] for d in biexp_data])
        # targets: [logit_a1, log_ratio]
        batches["biexp_y"] = torch.tensor(
            [[d[3], d[4]] for d in biexp_data], dtype=torch.float32)

    return dict(
        batches=batches, eq_freqs=eq_freqs,
        n_B=len(B_data), n_tau=len(tau_data), n_tau1_k1=len(tau1_k1_data),
        n_A0=len(A0_data), n_df=len(df_data), n_eq=len(eq_sub) if eq_sub else 0,
        n_noise=len(noise_data), n_biexp=len(biexp_data),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: InstrumentProfile, ds: dict, epochs: int = 800,
          lr: float = 3e-3, verbose: bool = True) -> list[float]:
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    b = ds["batches"]
    losses = []

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        terms = []

        if "B_mf" in b:
            pred = model.forward_B(b["B_mf"]).squeeze(-1)
            terms.append(nn.functional.mse_loss(pred, b["B_y"]))

        if "dur_mf" in b:
            pred = model.forward_dur(b["dur_mf"]).squeeze(-1)
            # Weight longer notes more — bass sustain (20-30s) must not be averaged down
            dur_w = torch.exp(b["dur_y"] * 0.1)
            dur_w = dur_w / dur_w.mean()
            terms.append((dur_w * (pred - b["dur_y"]) ** 2).mean())

        if "wf_mf" in b:
            pred = model.forward_wf(b["wf_mf"]).squeeze(-1)
            terms.append(nn.functional.mse_loss(pred, b["wf_y"]))

        # k=1 sustain — dedicated network, high weight (2x)
        if "tk1_mf" in b:
            pred = model.forward_tau1_k1(b["tk1_mf"], b["tk1_vf"]).squeeze(-1)
            terms.append(2.0 * nn.functional.mse_loss(pred, b["tk1_y"]))

        if "tau_mf" in b:
            pred = model.forward_tau_ratio(b["tau_mf"], b["tau_kf"]).squeeze(-1)
            # Huber loss: robust to remaining outliers, regresses toward median-like behavior
            terms.append((b["tau_w"] * nn.functional.huber_loss(pred, b["tau_y"], delta=0.3, reduction='none')).mean())

        if "a0_mf" in b:
            pred = model.forward_A0(b["a0_mf"], b["a0_kf"], b["a0_vf"]).squeeze(-1)
            terms.append(nn.functional.mse_loss(pred, b["a0_y"]))

        if "df_mf" in b:
            pred = model.forward_df(b["df_mf"], b["df_kf"]).squeeze(-1)
            terms.append(nn.functional.mse_loss(pred, b["df_y"]))

        if "eq_mf" in b:
            pred = model.forward_eq(b["eq_mf"], b["eq_ff"]).squeeze(-1)
            terms.append(0.1 * nn.functional.mse_loss(pred, b["eq_y"]))

        if "noise_mf" in b:
            pred = model.forward_noise(b["noise_mf"], b["noise_vf"])  # [N, 3]
            terms.append(nn.functional.mse_loss(pred, b["noise_y"]))

        if "biexp_mf" in b:
            pred = model.forward_biexp(b["biexp_mf"], b["biexp_kf"], b["biexp_vf"])  # [N, 2]
            terms.append(nn.functional.mse_loss(pred, b["biexp_y"]))

        # Smoothness penalty: consecutive MIDI values should give similar outputs
        # Evaluated on a fixed grid, independent of training data
        if epoch % 5 == 0:
            midi_grid = torch.arange(21, 108, dtype=torch.float32)
            mf_grid = torch.stack([midi_feat(float(m)) for m in midi_grid])
            kf_ref = k_feat(1)
            kf_batch = kf_ref.unsqueeze(0).expand(len(midi_grid), -1)
            vf_ref = vel_feat(4)
            vf_batch = vf_ref.unsqueeze(0).expand(len(midi_grid), -1)

            B_grid     = model.forward_B(mf_grid).squeeze(-1)
            tau1_grid  = model.forward_tau1_k1(mf_grid, vf_batch).squeeze(-1)
            a0_grid    = model.forward_A0(mf_grid, kf_batch, vf_batch).squeeze(-1)
            noise_grid = model.forward_noise(mf_grid, vf_batch)  # [N, 3]

            smooth = (
                (B_grid[1:]    - B_grid[:-1]).pow(2).mean() +
                (tau1_grid[1:] - tau1_grid[:-1]).pow(2).mean() +
                (a0_grid[1:]   - a0_grid[:-1]).pow(2).mean() +
                (noise_grid[1:] - noise_grid[:-1]).pow(2).mean()
            )
            terms.append(0.3 * smooth)

        loss = sum(terms) / len(terms) if terms else torch.tensor(0.0)
        loss.backward()
        opt.step()
        sched.step()

        losses.append(float(loss.detach()))
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}/{epochs}  loss={loss.item():.6f}  lr={sched.get_last_lr()[0]:.2e}")

    return losses


# ── Profile generation ────────────────────────────────────────────────────────

def generate_profile(
    model: InstrumentProfile,
    ds: dict,
    midi_from: int = 21,
    midi_to: int = 108,
    sr: int = 44100,
    orig_samples: dict | None = None,
) -> dict:
    """
    Evaluate trained model at all (midi, vel) positions.
    Returns samples dict compatible with params.json format.
    """
    model.eval()
    samples_out = {}

    eq_freqs = ds.get("eq_freqs")

    with torch.no_grad():
        for midi in range(midi_from, midi_to + 1):
            mf = midi_feat(midi)
            f0 = midi_to_hz(midi)

            B     = float(torch.exp(model.forward_B(mf)).item())
            B     = max(B, 1e-8)
            dur   = float(torch.exp(model.forward_dur(mf)).item())
            dur   = max(dur, 0.3)
            wf    = float(torch.exp(model.forward_wf(mf)).item())
            wf    = float(np.clip(wf, 0.1, 10.0))

            # EQ profile
            spectral_eq: dict = {}
            if eq_freqs is not None:
                gains = []
                for fhz in eq_freqs:
                    ff = freq_feat(float(fhz))
                    g = float(model.forward_eq(mf, ff).item())
                    gains.append(float(np.clip(g, -30, 20)))
                spectral_eq = {
                    "freqs_hz": [round(float(f), 2) for f in eq_freqs],
                    "gains_db": [round(g, 4) for g in gains],
                    "stereo_width_factor": round(wf, 4),
                }

            # Number of audible partials
            n_partials = max(1, int((sr / 2) / f0))

            for vel in range(8):
                vf = vel_feat(vel)
                key = f"m{midi:03d}_vel{vel}"

                # Noise model prediction for this (midi, vel)
                noise_pred = model.forward_noise(mf, vf).squeeze(0)  # [3]
                attack_tau = float(torch.exp(noise_pred[0]).item())
                attack_tau = float(np.clip(attack_tau, 0.002, 1.0))
                centroid   = float(torch.exp(noise_pred[1]).item())
                centroid   = float(np.clip(centroid, 100.0, 20000.0))
                A_noise    = float(torch.exp(noise_pred[2]).item())
                A_noise    = float(np.clip(A_noise, 0.001, 0.5))
                noise_out  = {
                    "attack_tau_s": round(attack_tau, 5),
                    "centroid_hz":  round(centroid, 1),
                    "A_noise":      round(A_noise, 5),
                }

                # Per-partial predictions
                partials = []
                for k in range(1, n_partials + 1):
                    kf = k_feat(k)
                    f_k = k * f0 * math.sqrt(1.0 + B * k ** 2)
                    if f_k >= sr / 2:
                        break

                    # tau1: k=1 from dedicated net; k>1 from tau1_k1 * exp(ratio)
                    tau1_k1 = float(torch.exp(model.forward_tau1_k1(mf, vf)).item())
                    if k == 1:
                        tau1 = tau1_k1
                    else:
                        log_ratio = float(model.forward_tau_ratio(mf, kf).item())
                        log_k_bias = -0.3 * math.log(k)
                        log_ratio = max(log_k_bias - 2.0, min(0.0, log_ratio))
                        tau1 = tau1_k1 * math.exp(log_ratio)
                    tau1 = max(tau1, 0.005)

                    # Bi-exponential decay parameters
                    biexp_pred = model.forward_biexp(mf, kf, vf).squeeze(0)  # [2]
                    a1_raw   = float(torch.sigmoid(biexp_pred[0]).item())
                    tau2_ratio = float(torch.exp(biexp_pred[1]).item())
                    a1_val   = float(np.clip(a1_raw, 0.05, 0.99))
                    tau2_val = tau1 * max(tau2_ratio, 1.1)  # tau2 always > tau1
                    # Only emit tau2/a1 when meaningfully biexponential (a1 < 0.92)
                    emit_biexp = a1_val < 0.92

                    a0_pred  = model.forward_A0(mf, kf, vf)
                    a0_ratio = float(torch.exp(a0_pred).item())
                    a0_ratio = max(a0_ratio, 1e-6)
                    A0 = a0_ratio

                    df_pred = model.forward_df(mf, kf)
                    df = float(torch.exp(df_pred).item())
                    df = max(df, 0.0)

                    entry = {
                        "k":      k,
                        "f_hz":   round(f_k, 4),
                        "A0":     round(float(A0), 6),
                        "tau1":   round(float(tau1), 6),
                        "a1":     round(a1_val, 4),
                        "beat_hz": round(float(df), 6),
                    }
                    if emit_biexp:
                        entry["tau2"] = round(tau2_val, 6)
                    partials.append(entry)

                sample = {
                    "midi":       midi,
                    "vel":        vel,
                    "B":          round(float(B), 8),
                    "duration_s": round(float(dur), 3),
                    "partials":   partials,
                    "noise":      noise_out,
                    "_from_profile": True,
                }
                if spectral_eq:
                    sample["spectral_eq"] = spectral_eq

                # Preserve original if available (trust measured data)
                if orig_samples and key in orig_samples and not orig_samples[key].get("_interpolated"):
                    sample = copy.deepcopy(orig_samples[key])
                    sample["_from_profile"] = False

                samples_out[key] = sample

    return samples_out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    _setup_log()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",     dest="inp",    default="analysis/params.json")
    ap.add_argument("--out",    dest="out",    default="analysis/params-nn-profile.json")
    ap.add_argument("--model",  dest="model",  default="analysis/profile.pt")
    ap.add_argument("--epochs", type=int,      default=800)
    ap.add_argument("--hidden", type=int,      default=64)
    ap.add_argument("--lr",     type=float,    default=3e-3)
    ap.add_argument("--midi-from", type=int,   default=21)
    ap.add_argument("--midi-to",   type=int,   default=108)
    ap.add_argument("--sr",        type=int,   default=44100)
    ap.add_argument("--no-preserve-orig", action="store_true",
                    help="Replace all samples with NN output, even available ones")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print(f"Reading {args.inp} ...")
    raw = json.loads(Path(args.inp).read_text())
    samples = raw["samples"]

    n_avail = sum(1 for s in samples.values() if not s.get("_interpolated"))
    print(f"Available measured samples: {n_avail}")

    print("Building dataset ...")
    ds = build_dataset(samples)
    print(f"  B={ds['n_B']}  tau={ds['n_tau']}  tau1_k1={ds['n_tau1_k1']}  "
          f"A0={ds['n_A0']}  df={ds['n_df']}  eq={ds['n_eq']}  "
          f"noise={ds['n_noise']}  biexp={ds['n_biexp']}")

    model = InstrumentProfile(hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"Training {args.epochs} epochs ...")
    train(model, ds, epochs=args.epochs, lr=args.lr)

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": args.hidden,
        "eq_freqs": ds["eq_freqs"].tolist() if ds["eq_freqs"] is not None else None,
    }, args.model)
    print(f"Model saved -> {args.model}")

    # Generate full profile
    print("Generating full parameter profile ...")
    orig = None if args.no_preserve_orig else samples
    profile_samples = generate_profile(
        model, ds,
        midi_from=args.midi_from, midi_to=args.midi_to,
        sr=args.sr, orig_samples=orig,
    )

    n_nn    = sum(1 for s in profile_samples.values() if s.get("_from_profile"))
    n_orig  = sum(1 for s in profile_samples.values() if not s.get("_from_profile"))
    print(f"  NN-generated: {n_nn}  |  Preserved originals: {n_orig}")

    out_data = dict(raw, samples=profile_samples)
    Path(args.out).write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"Written -> {args.out}")

    if args.plot:
        _plot(profile_samples, samples)


def _plot(profile: dict, orig: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for vel, ax_row in zip([4, 7], axes):
        for ax, param in zip(ax_row, ["B", "tau1_k1"]):
            orig_m, orig_v, nn_m, nn_v = [], [], [], []
            for midi in range(21, 109):
                key = f"m{midi:03d}_vel{vel}"
                s_o = orig.get(key)
                s_p = profile.get(key)
                if s_o and not s_o.get("_interpolated"):
                    if param == "B":
                        v = s_o.get("B") or 0
                        if v > 0:
                            orig_m.append(midi); orig_v.append(v)
                    else:
                        parts = {p["k"]: p for p in s_o.get("partials", []) if "k" in p}
                        v = parts.get(1, {}).get("tau1") or 0
                        if v > 0:
                            orig_m.append(midi); orig_v.append(v)
                if s_p:
                    if param == "B":
                        v = s_p.get("B") or 0
                        if v > 0:
                            nn_m.append(midi); nn_v.append(v)
                    else:
                        parts = {p["k"]: p for p in s_p.get("partials", []) if "k" in p}
                        v = parts.get(1, {}).get("tau1") or 0
                        if v > 0:
                            nn_m.append(midi); nn_v.append(v)

            ax.scatter(orig_m, orig_v, s=30, c="lime", label="measured", zorder=3)
            ax.plot(nn_m, nn_v, "r-", lw=1.5, label="NN profile", alpha=0.8)
            ax.set_title(f"vel{vel} {param}")
            ax.legend(fontsize=8)
            ax.set_xlabel("MIDI")

    plt.tight_layout()
    out = "analysis/profile_diagnostics.png"
    plt.savefig(out, dpi=120)
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()
