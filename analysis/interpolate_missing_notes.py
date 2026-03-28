"""
analysis/interpolate_missing_notes.py
──────────────────────────────────────
Interpolate partial parameters for MIDI notes missing from params.json.

Use case: sparse sample bank (e.g. every 4th note, or 20 strategic notes)
→ interpolate full 88-note params.json from available measurements.

Strategy per parameter:
  B (inharmonicity)     : log-space spline over MIDI axis (physics: B ∝ 1/L²)
  tau1_k, tau2_k        : log-space spline per partial k; outlier-filtered
  A0 ratios             : spline per partial k (normalised within each note)
  df (beat Hz)          : spline per partial k; 0 for single-string notes
  n_partials            : derived from sr/2 / f_k (physical, not interpolated)
  duration_s            : spline over MIDI axis
  spectral_eq           : per-frequency spline over MIDI axis
  stereo_width_factor   : spline over MIDI axis

Outlier filtering: for each (parameter, k), values more than N×IQR from the
median are removed before fitting. Remaining values are fitted with a
smoothing spline (UnivariateSpline with s > 0).

Usage:
  python analysis/interpolate_missing_notes.py
         --in  analysis/params.json
         --out analysis/params_interpolated.json
         [--vel 4]           # reference velocity for shape/timbre params
         [--all-vel]         # interpolate all 8 velocity layers (default: all)
         [--midi-from 21 --midi-to 108]
         [--plot]            # show diagnostic plots
"""

import argparse
import copy
import json
import math
import numpy as np
from pathlib import Path
from scipy.interpolate import UnivariateSpline, interp1d


# ── helpers ──────────────────────────────────────────────────────────────────

def midi_to_hz(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def max_partials(midi: int, sr: int = 44100) -> int:
    """Number of harmonic partials below Nyquist for this note."""
    f0 = midi_to_hz(midi)
    return max(1, int((sr / 2) / f0))


def iqr_filter(xs, ys, k_iqr: float = 3.0):
    """Remove outliers > k_iqr * IQR from median. Returns filtered (xs, ys)."""
    if len(ys) < 4:
        return xs, ys
    ys = np.array(ys)
    q25, q75 = np.percentile(ys, 25), np.percentile(ys, 75)
    iqr = q75 - q25
    if iqr < 1e-12:
        return xs, ys
    med = np.median(ys)
    mask = np.abs(ys - med) <= k_iqr * iqr
    return [x for x, m in zip(xs, mask) if m], ys[mask].tolist()


def fit_spline(midis, values, log_space: bool = False,
               s_factor: float = 1.0, k_iqr: float = 3.0):
    """
    Fit a smoothing spline to (midis, values).
    log_space=True: fit in log(value) space (for strictly positive params).
    Returns a callable f(midi) → value.
    """
    midis, values = iqr_filter(midis, values, k_iqr)
    if len(midis) < 2:
        v = float(np.median(values)) if values else 0.0
        return lambda m: v

    midis = np.array(midis, dtype=float)
    values = np.array(values, dtype=float)

    # Sort
    order = np.argsort(midis)
    midis, values = midis[order], values[order]

    if log_space:
        valid = values > 0
        if valid.sum() < 2:
            v = float(np.median(values))
            return lambda m: v
        midis, values = midis[valid], values[valid]
        y_fit = np.log(values)
    else:
        y_fit = values

    # Smoothing spline: s controls smoothness (larger = smoother)
    n = len(midis)
    s = s_factor * n  # default scipy s=n gives moderate smoothing

    try:
        if n >= 4:
            sp = UnivariateSpline(midis, y_fit, s=s, k=min(3, n - 1), ext=3)
        else:
            sp = interp1d(midis, y_fit, kind='linear', fill_value='extrapolate')
    except Exception:
        sp = interp1d(midis, y_fit, kind='linear', fill_value='extrapolate')

    if log_space:
        return lambda m, _sp=sp: float(np.exp(np.clip(_sp(m), -10, 10)))
    else:
        return lambda m, _sp=sp: float(_sp(m))


# ── main interpolation ────────────────────────────────────────────────────────

def interpolate_params(
    data: dict,
    vel_layers: list[int] | None = None,
    midi_from: int = 21,
    midi_to: int = 108,
    sr: int = 44100,
    verbose: bool = True,
) -> dict:
    """
    Fill in missing MIDI notes by interpolating from available data.
    Modifies data in-place and returns it.
    """
    samples = data["samples"]
    if vel_layers is None:
        vel_layers = list(range(8))

    target_midis = list(range(midi_from, midi_to + 1))

    for vel in vel_layers:
        if verbose:
            print(f"\n--- Velocity {vel} ---")

        # Collect available notes for this velocity
        avail: dict[int, dict] = {}
        for midi in range(21, 109):
            key = f"m{midi:03d}_vel{vel}"
            if key in samples:
                avail[midi] = samples[key]

        if len(avail) < 2:
            if verbose:
                print(f"  Only {len(avail)} notes available, skipping interpolation")
            continue

        avail_midis = sorted(avail.keys())
        missing = [m for m in target_midis if f"m{m:03d}_vel{vel}" not in samples]

        if verbose:
            print(f"  Available: {len(avail_midis)} notes  |  Missing: {len(missing)}")
        if not missing:
            if verbose:
                print("  Nothing to interpolate.")
            continue

        # ── 1. Inharmonicity B ─────────────────────────────────────────────
        B_midis  = avail_midis[:]
        B_values = [avail[m].get("B", 0.0) or 0.0 for m in B_midis]
        # B should be > 0; zero values are extraction failures
        B_midis_clean = [m for m, b in zip(B_midis, B_values) if b > 1e-7]
        B_values_clean = [b for b in B_values if b > 1e-7]
        fit_B = fit_spline(B_midis_clean, B_values_clean, log_space=True, s_factor=2.0)

        # ── 2. duration_s ──────────────────────────────────────────────────
        dur_midis  = avail_midis[:]
        dur_values = [avail[m].get("duration_s") or 3.0 for m in dur_midis]
        fit_dur = fit_spline(dur_midis, dur_values, log_space=True, s_factor=1.5)

        # ── 3. Per-partial params: tau1, tau2, A0 ratio, df ───────────────
        # Determine max k across all available notes
        all_k = set()
        for s in avail.values():
            for p in s.get("partials", []):
                if "k" in p:
                    all_k.add(p["k"])
        all_k = sorted(all_k)

        # Collect per-k data
        tau1_fits:    dict[int, callable] = {}
        tau2_fits:    dict[int, callable] = {}
        a0_ratio_fits: dict[int, callable] = {}  # A0_k / A0_k1
        df_fits:      dict[int, callable] = {}

        for k in all_k:
            t1_m, t1_v = [], []
            t2_m, t2_v = [], []
            a0r_m, a0r_v = [], []
            df_m, df_v = [], []

            for midi in avail_midis:
                s = avail[midi]
                parts = {p["k"]: p for p in s.get("partials", []) if "k" in p}
                p = parts.get(k)
                p1 = parts.get(1)
                if p is None or p1 is None:
                    continue

                tau1 = p.get("tau1") or 0
                if tau1 > 0:
                    t1_m.append(midi); t1_v.append(tau1)

                tau2 = p.get("tau2") or 0
                if tau2 > 0:
                    t2_m.append(midi); t2_v.append(tau2)

                a0 = p.get("A0") or 0
                a0_1 = p1.get("A0") or 0
                if a0 > 0 and a0_1 > 0:
                    a0r_m.append(midi); a0r_v.append(a0 / a0_1)

                df = p.get("df") or 0
                if df > 0:
                    df_m.append(midi); df_v.append(df)

            tau1_fits[k]     = fit_spline(t1_m, t1_v, log_space=True, s_factor=1.5) if t1_m else None
            tau2_fits[k]     = fit_spline(t2_m, t2_v, log_space=True, s_factor=1.5) if t2_m else None
            a0_ratio_fits[k] = fit_spline(a0r_m, a0r_v, log_space=True, s_factor=1.5) if a0r_m else None
            df_fits[k]       = fit_spline(df_m, df_v, log_space=True, s_factor=2.0)  if df_m  else None

        # A0 absolute for k=1 (used to scale all others)
        a0_k1_midis  = avail_midis[:]
        a0_k1_values = []
        for midi in a0_k1_midis:
            s = avail[midi]
            parts = {p["k"]: p for p in s.get("partials", []) if "k" in p}
            a0_k1_values.append(parts.get(1, {}).get("A0") or 1.0)
        fit_a0_k1 = fit_spline(a0_k1_midis, a0_k1_values, log_space=True, s_factor=1.5)

        # ── 4. Spectral EQ ────────────────────────────────────────────────
        # Collect common freq grid from first available sample with eq
        eq_freqs = None
        eq_gains_by_midi: dict[int, np.ndarray] = {}
        for midi in avail_midis:
            eq = avail[midi].get("spectral_eq") or {}
            if eq.get("freqs_hz") and eq.get("gains_db"):
                fq = np.array(eq["freqs_hz"])
                gd = np.array(eq["gains_db"])
                if eq_freqs is None:
                    eq_freqs = fq
                if len(gd) == len(eq_freqs):
                    eq_gains_by_midi[midi] = gd

        eq_freq_fits: list[callable] = []
        if eq_freqs is not None and len(eq_gains_by_midi) >= 2:
            eq_midis = sorted(eq_gains_by_midi.keys())
            for fi in range(len(eq_freqs)):
                gvals = [eq_gains_by_midi[m][fi] for m in eq_midis]
                eq_freq_fits.append(fit_spline(eq_midis, gvals, log_space=False, s_factor=2.0))

        # Stereo width factor
        wf_midis, wf_values = [], []
        for midi in avail_midis:
            eq = avail[midi].get("spectral_eq") or {}
            wf = eq.get("stereo_width_factor")
            if wf and wf > 0:
                wf_midis.append(midi); wf_values.append(wf)
        fit_wf = fit_spline(wf_midis, wf_values, log_space=True, s_factor=2.0) if wf_midis else None

        # ── 5. Build interpolated samples ─────────────────────────────────
        n_created = 0
        for midi in missing:
            f0 = midi_to_hz(midi)
            n_p = max_partials(midi, sr)
            B = max(fit_B(midi), 1e-7)
            dur = max(fit_dur(midi), 0.5)
            a0_k1 = max(fit_a0_k1(midi), 1e-6)

            partials = []
            for k in range(1, n_p + 1):
                if k not in tau1_fits or tau1_fits[k] is None:
                    continue

                # Inharmonic frequency
                f_k = k * f0 * math.sqrt(1.0 + B * k ** 2)
                if f_k >= sr / 2:
                    break

                tau1 = max(tau1_fits[k](midi), 0.01)
                tau2 = None
                if k in tau2_fits and tau2_fits[k] is not None:
                    tau2_val = tau2_fits[k](midi)
                    if tau2_val > 0.01 and tau2_val < tau1:
                        tau2 = round(tau2_val, 6)

                a0_ratio = 1.0 if k == 1 else (
                    max(a0_ratio_fits[k](midi), 1e-6)
                    if k in a0_ratio_fits and a0_ratio_fits[k] is not None
                    else 0.1
                )
                A0 = a0_k1 * a0_ratio

                df = 0.0
                if k in df_fits and df_fits[k] is not None:
                    df = max(df_fits[k](midi), 0.0)

                entry = {
                    "k":    k,
                    "f_hz": round(f_k, 4),
                    "A0":   round(float(A0), 6),
                    "tau1": round(float(tau1), 6),
                    "df":   round(float(df), 6),
                }
                if tau2 is not None:
                    entry["tau2"] = tau2
                partials.append(entry)

            # Spectral EQ
            spectral_eq: dict = {}
            if eq_freqs is not None and eq_freq_fits:
                gains = [float(np.clip(f(midi), -30, 20)) for f in eq_freq_fits]
                spectral_eq["freqs_hz"] = [round(float(f), 2) for f in eq_freqs]
                spectral_eq["gains_db"] = [round(g, 4) for g in gains]
            if fit_wf is not None:
                spectral_eq["stereo_width_factor"] = round(float(np.clip(fit_wf(midi), 0.2, 8.0)), 4)

            key = f"m{midi:03d}_vel{vel}"
            sample = {
                "midi":       midi,
                "vel":        vel,
                "B":          round(float(B), 8),
                "duration_s": round(float(dur), 3),
                "partials":   partials,
                "_interpolated": True,
            }
            if spectral_eq:
                sample["spectral_eq"] = spectral_eq

            samples[key] = sample
            n_created += 1

        if verbose:
            print(f"  Created {n_created} interpolated notes")

    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="inp", default="analysis/params.json")
    ap.add_argument("--out",  dest="out", default="analysis/params_interpolated.json")
    ap.add_argument("--vel",  type=int,   default=None,
                    help="Single velocity layer to process (default: all)")
    ap.add_argument("--midi-from", type=int, default=21)
    ap.add_argument("--midi-to",   type=int, default=108)
    ap.add_argument("--sr",        type=int, default=44100)
    ap.add_argument("--plot",      action="store_true")
    args = ap.parse_args()

    vel_layers = [args.vel] if args.vel is not None else list(range(8))

    inp_path = Path(args.inp)
    print(f"Reading {inp_path} ...")
    data = json.loads(inp_path.read_text())

    orig_count = len(data["samples"])
    interpolate_params(data, vel_layers=vel_layers,
                       midi_from=args.midi_from, midi_to=args.midi_to,
                       sr=args.sr, verbose=True)
    new_count = len(data["samples"])

    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nDone: {orig_count} -> {new_count} samples (+{new_count - orig_count})")
    print(f"Written -> {out_path}")

    if args.plot:
        _plot_diagnostics(data, vel_layers[0])


def _plot_diagnostics(data: dict, vel: int):
    """Quick plot of B and tau1_k1 across MIDI range."""
    import matplotlib.pyplot as plt
    samples = data["samples"]
    orig_midis, orig_B, orig_tau = [], [], []
    interp_midis, interp_B, interp_tau = [], [], []

    for midi in range(21, 109):
        key = f"m{midi:03d}_vel{vel}"
        s = samples.get(key)
        if not s:
            continue
        B = s.get("B", 0)
        parts = {p["k"]: p for p in s.get("partials", []) if "k" in p}
        tau = parts.get(1, {}).get("tau1", 0)
        if s.get("_interpolated"):
            interp_midis.append(midi); interp_B.append(B); interp_tau.append(tau)
        else:
            orig_midis.append(midi); orig_B.append(B); orig_tau.append(tau)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.scatter(orig_midis, orig_B, s=20, label="original", color="lime")
    ax1.scatter(interp_midis, interp_B, s=10, marker="x", label="interpolated", color="red")
    ax1.set_ylabel("B (inharmonicity)"); ax1.set_title(f"Vel {vel}: Inharmonicity B"); ax1.legend()
    ax2.scatter(orig_midis, orig_tau, s=20, label="original", color="lime")
    ax2.scatter(interp_midis, interp_tau, s=10, marker="x", label="interpolated", color="red")
    ax2.set_ylabel("tau1 k=1 (s)"); ax2.set_xlabel("MIDI"); ax2.set_title("tau1 k=1"); ax2.legend()
    plt.tight_layout()
    plt.savefig("analysis/interpolation_diagnostics.png", dpi=120)
    print("Plot saved -> analysis/interpolation_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    main()
