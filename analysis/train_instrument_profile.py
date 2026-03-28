"""
analysis/train_instrument_profile.py
──────────────────────────────────────
Learn a smooth instrument profile from raw extracted parameters.

A small factorised neural network learns the piano's physical parameter
landscape from all available (midi, vel) samples — handling noisy extractions,
missing notes, and inter-velocity inconsistencies in one step.

Architecture (physically motivated factorisation):
  B_net    : MLP(midi)           → B          (inharmonicity, vel-independent)
  tau_net  : MLP(midi, k)        → tau1, tau2  (string decay, vel-independent)
  A0_net   : MLP(midi, k, vel)   → A0_ratio    (spectral shape, vel-dependent)
  df_net   : MLP(midi)           → df          (beating, vel-independent)
  dur_net  : MLP(midi)           → duration_s
  eq_net   : MLP(midi, freq)     → gain_db     (body EQ, vel-independent)

All positive outputs trained in log-space (MSE on log values → geometric error).
Width factor trained in log-space separately.

Output: profile.pt  (model weights + metadata)
        params_profile.json  (full 88×8 params, compatible with GUI/synthesiser)

Usage:
  python analysis/train_instrument_profile.py
         --in  analysis/params.json       # raw extraction (may be sparse/noisy)
         --out analysis/params_profile.json
         --model analysis/profile.pt
         [--epochs 800]
         [--midi-from 21 --midi-to 108]
         [--plot]
"""

import argparse
import json
import math
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.B_net   = mlp(MIDI_DIM, hidden, 1)           # log(B)
        self.dur_net = mlp(MIDI_DIM, hidden, 1)           # log(dur)
        self.tau_net = mlp(MIDI_DIM + K_DIM, hidden, 2)  # log(tau1), log(tau2)
        self.A0_net  = mlp(MIDI_DIM + K_DIM + VEL_DIM, hidden, 1)  # log(A0_ratio)
        self.df_net  = mlp(MIDI_DIM + K_DIM, hidden, 1)  # log(df+1)
        self.eq_net  = mlp(MIDI_DIM + FREQ_DIM, hidden, 1)  # gain_db
        self.wf_net  = mlp(MIDI_DIM, hidden, 1)           # log(width_factor)

    def forward_B(self, mf):         return self.B_net(mf)
    def forward_dur(self, mf):       return self.dur_net(mf)
    def forward_tau(self, mf, kf):   return self.tau_net(torch.cat([mf, kf], -1))
    def forward_A0(self, mf, kf, vf): return self.A0_net(torch.cat([mf, kf, vf], -1))
    def forward_df(self, mf, kf):    return self.df_net(torch.cat([mf, kf], -1))
    def forward_eq(self, mf, ff):    return self.eq_net(torch.cat([mf, ff], -1))
    def forward_wf(self, mf):        return self.wf_net(mf)


# ── Data extraction ───────────────────────────────────────────────────────────

def build_dataset(samples: dict) -> dict:
    """Extract training tensors from raw params dict."""
    B_data, dur_data, wf_data = [], [], []
    tau_data, A0_data, df_data, eq_data = [], [], [], []

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

        # per-partial
        parts = {p["k"]: p for p in s.get("partials", []) if "k" in p}
        a0_k1 = parts.get(1, {}).get("A0") or 0

        for k, p in parts.items():
            kf = k_feat(k)

            # tau
            t1 = p.get("tau1") or 0
            t2 = p.get("tau2") or 0
            if t1 > 0.005:
                tau_data.append((mf, kf, math.log(t1),
                                 math.log(max(t2, 0.005)) if t2 > 0.005 else None))

            # A0 ratio (k >= 1, normalised to k=1)
            a0 = p.get("A0") or 0
            if a0 > 0 and a0_k1 > 0:
                ratio = a0 / a0_k1
                if ratio > 1e-5:
                    A0_data.append((mf, kf, vf, math.log(ratio)))

            # df (beat Hz)
            df = p.get("df") or 0
            if df > 0.001:
                df_data.append((mf, kf, math.log(df)))

    return dict(
        B=B_data, dur=dur_data, wf=wf_data,
        tau=tau_data, A0=A0_data, df=df_data, eq=eq_data,
        eq_freqs=eq_freqs,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: InstrumentProfile, ds: dict, epochs: int = 800,
          lr: float = 3e-3, verbose: bool = True) -> list[float]:
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    losses = []

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        n = 0

        # B
        for (mf, log_b) in ds["B"]:
            pred = model.forward_B(mf).squeeze()
            loss = loss + (pred - log_b) ** 2
            n += 1

        # duration
        for (mf, log_d) in ds["dur"]:
            pred = model.forward_dur(mf).squeeze()
            loss = loss + (pred - log_d) ** 2
            n += 1

        # width factor
        for (mf, log_w) in ds["wf"]:
            pred = model.forward_wf(mf).squeeze()
            loss = loss + (pred - log_w) ** 2
            n += 1

        # tau (weight higher-k less — noisy)
        for (mf, kf, log_t1, log_t2) in ds["tau"]:
            k_weight = 1.0 / (1.0 + float(kf[0]) * 3)
            pred = model.forward_tau(mf, kf)
            loss = loss + k_weight * (pred[0] - log_t1) ** 2
            if log_t2 is not None:
                loss = loss + k_weight * 0.5 * (pred[1] - log_t2) ** 2
            n += 1

        # A0 ratio
        for (mf, kf, vf, log_r) in ds["A0"]:
            pred = model.forward_A0(mf, kf, vf).squeeze()
            loss = loss + (pred - log_r) ** 2
            n += 1

        # df
        for (mf, kf, log_df) in ds["df"]:
            pred = model.forward_df(mf, kf).squeeze()
            loss = loss + (pred - log_df) ** 2
            n += 1

        # EQ (subsample to limit compute)
        eq_batch = ds["eq"][::4] if len(ds["eq"]) > 200 else ds["eq"]
        for (mf, ff, g) in eq_batch:
            pred = model.forward_eq(mf, ff).squeeze()
            loss = loss + 0.1 * (pred - g) ** 2
            n += 1

        if n > 0:
            loss = loss / n
        loss.backward()
        opt.step()
        sched.step()

        losses.append(float(loss))
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

                # Per-partial predictions
                partials = []
                for k in range(1, n_partials + 1):
                    kf = k_feat(k)
                    f_k = k * f0 * math.sqrt(1.0 + B * k ** 2)
                    if f_k >= sr / 2:
                        break

                    tau_pred = model.forward_tau(mf, kf)
                    tau1 = float(torch.exp(tau_pred[0]).item())
                    tau2 = float(torch.exp(tau_pred[1]).item())
                    tau1 = max(tau1, 0.005)
                    tau2 = max(tau2, 0.005)
                    if tau2 >= tau1:
                        tau2 = None  # only keep tau2 if shorter than tau1

                    a0_pred = model.forward_A0(mf, kf, vf)
                    a0_ratio = float(torch.exp(a0_pred).item())
                    a0_ratio = max(a0_ratio, 1e-6)
                    A0 = a0_ratio  # k=1 A0_ratio = 1.0 by definition

                    df_pred = model.forward_df(mf, kf)
                    df = float(torch.exp(df_pred).item()) - 1.0
                    df = max(df, 0.0)

                    entry = {
                        "k":    k,
                        "f_hz": round(f_k, 4),
                        "A0":   round(float(A0), 6),
                        "tau1": round(float(tau1), 6),
                        "df":   round(float(df), 6),
                    }
                    if tau2 is not None:
                        entry["tau2"] = round(float(tau2), 6)
                    partials.append(entry)

                sample = {
                    "midi":       midi,
                    "vel":        vel,
                    "B":          round(float(B), 8),
                    "duration_s": round(float(dur), 3),
                    "partials":   partials,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",     dest="inp",    default="analysis/params.json")
    ap.add_argument("--out",    dest="out",    default="analysis/params_profile.json")
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
    print(f"  B={len(ds['B'])}  dur={len(ds['dur'])}  tau={len(ds['tau'])}  "
          f"A0={len(ds['A0'])}  df={len(ds['df'])}  eq={len(ds['eq'])}")

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
