"""
analysis/smooth_tau_cross_velocity.py
──────────────────────────────────────
Post-process params.json: for each (midi, partial-k), replace tau1 and tau2
with an amplitude-weighted median across all 8 velocity layers.

Rationale:
  String decay times are determined by material properties (string, bridge, damper),
  NOT by hammer velocity. Per-velocity extraction is noisy for weak partials at low
  velocity layers (low SNR → unreliable curve fit). Cross-velocity smoothing preserves
  the physics while eliminating extraction artifacts that make each velocity sound like
  a different instrument.

  Amplitudes A0 / A0_2 are kept per-velocity (real physics: harder strike → brighter,
  different hammer contact time → different spectral balance).

Usage:
  python analysis/smooth_tau_cross_velocity.py [--in params.json] [--out params_smoothed.json]
  python analysis/smooth_tau_cross_velocity.py --inplace   # overwrite params.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def weighted_median(values, weights):
    """Compute weighted median of values with given weights."""
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    # Remove NaN/zero weights
    mask = np.isfinite(values) & (weights > 0) & (values > 0)
    if not mask.any():
        return np.median(values) if len(values) > 0 else 0.0
    v, w = values[mask], weights[mask]
    if len(v) == 1:
        return float(v[0])
    # Sort by value, cumulative weight
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    w_cum = np.cumsum(w_sorted) / w_sorted.sum()
    # Find first index where cumulative weight >= 0.5
    idx = np.searchsorted(w_cum, 0.5)
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def smooth_params(data: dict, snr_threshold: float = 0.25, verbose: bool = True) -> dict:
    """
    For each MIDI note, for each partial k, blend tau1/tau2 between the original
    per-velocity extraction and a high-SNR reference tau, based on signal reliability.

    Strategy:
      - "Reference tau" = amplitude-weighted median across all velocity layers
        (dominated by high-velocity / high-A0 layers = high SNR)
      - Per-velocity reliability = A0 / A0_max  (0..1)
      - If reliability >= snr_threshold: keep original tau (extraction is trustworthy)
      - If reliability < snr_threshold: blend toward reference tau
        blend_weight = 1 - (reliability / snr_threshold)   [0..1, 1=full reference]
        tau_final = (1 - blend_weight) * tau_orig + blend_weight * tau_ref

    This preserves accurate high-velocity extractions while correcting noisy
    low-velocity extractions, which cause each velocity to sound like a different
    instrument.

    snr_threshold: A0 fraction of maximum below which blending starts (default 0.25).
    """
    samples = data["samples"]

    # Group sample keys by midi note
    midi_to_keys: dict[int, list[str]] = defaultdict(list)
    for key, s in samples.items():
        midi = s.get("midi")
        if midi is not None:
            midi_to_keys[midi].append(key)

    stats = {"midi_notes": 0, "partials_smoothed": 0, "high_variance": 0, "blended": 0}

    for midi in sorted(midi_to_keys.keys()):
        keys = midi_to_keys[midi]
        stats["midi_notes"] += 1

        # Collect all partial data across velocity layers for this midi
        # {k: [(tau1, A0, tau2, A0_2, sample_key), ...]}
        partial_data: dict[int, list] = defaultdict(list)

        for key in keys:
            s = samples[key]
            for p in s.get("partials", []):
                k = p.get("k")
                if k is None:
                    continue
                tau1 = p.get("tau1", 0) or 0
                A0   = p.get("A0",   0) or 0
                tau2 = p.get("tau2")    or None
                A0_2 = p.get("A0_2")   or None
                partial_data[k].append((tau1, A0, tau2, A0_2, key))

        # Compute reference tau per partial (amplitude-weighted median across all vels)
        ref_tau1: dict[int, float] = {}
        ref_tau2: dict[int, float] = {}
        max_a0:   dict[int, float] = {}
        max_a0_2: dict[int, float] = {}

        for k, entries in partial_data.items():
            tau1_vals = [e[0] for e in entries]
            a0_vals   = [e[1] for e in entries]
            tau2_vals = [e[2] for e in entries if e[2] is not None]
            a0_2_vals = [e[3] for e in entries if e[3] is not None]

            ref_tau1[k] = weighted_median(tau1_vals, a0_vals)
            max_a0[k]   = max(a0_vals) if a0_vals else 1.0

            if tau2_vals and a0_2_vals and len(tau2_vals) == len(a0_2_vals):
                ref_tau2[k]   = weighted_median(tau2_vals, a0_2_vals)
                max_a0_2[k]   = max(a0_2_vals) if a0_2_vals else 1.0

            # Count high-variance partials (for stats)
            valid = [t for t in tau1_vals if t > 0]
            if len(valid) >= 3 and max(valid) / min(valid) > 3.0:
                stats["high_variance"] += 1

            stats["partials_smoothed"] += 1

        # Apply blended tau back into each velocity layer
        for key in keys:
            s = samples[key]
            for p in s.get("partials", []):
                k = p.get("k")
                if k is None or k not in ref_tau1:
                    continue

                A0 = p.get("A0", 0) or 0
                reliability = A0 / max_a0[k] if max_a0.get(k, 0) > 0 else 1.0
                reliability = min(reliability, 1.0)

                if reliability >= snr_threshold:
                    # Signal is strong enough — trust the original extraction
                    pass
                else:
                    # Blend toward reference: more blending for weaker signal
                    blend = 1.0 - reliability / snr_threshold   # 0..1
                    tau_orig = p.get("tau1", 0) or 0
                    tau_ref  = ref_tau1[k]
                    if tau_orig > 0 and tau_ref > 0:
                        p["tau1"] = round((1.0 - blend) * tau_orig + blend * tau_ref, 6)
                        stats["blended"] += 1

                # Same for tau2
                A0_2 = p.get("A0_2") or 0
                if k in ref_tau2 and A0_2 > 0:
                    rel2 = A0_2 / max_a0_2.get(k, 1.0)
                    rel2 = min(rel2, 1.0)
                    if rel2 < snr_threshold:
                        blend2 = 1.0 - rel2 / snr_threshold
                        tau2_orig = p.get("tau2") or 0
                        if tau2_orig > 0:
                            p["tau2"] = round(
                                (1.0 - blend2) * tau2_orig + blend2 * ref_tau2[k], 6
                            )

    if verbose:
        print(f"Smoothed {stats['midi_notes']} MIDI notes, "
              f"{stats['partials_smoothed']} unique (midi,k) pairs.")
        print(f"High-variance partials detected: {stats['high_variance']}")
        print(f"Blended (low-SNR) tau values:    {stats['blended']}")

    return data


def blend_color_toward_reference(
    data: dict,
    ref_vel: int = 4,
    blend: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Blend per-velocity spectral color (A0 amplitude ratios) toward a reference
    velocity layer, while preserving each velocity's total energy.

    Rationale:
      The mid-velocity layer (vel4 by default) tends to have the most reliable
      partial amplitude extraction: strong enough signal for good SNR, but not
      so loud that the hammer contact time changes spectral balance dramatically.
      Blending other velocities toward this "sweet spot" produces more consistent
      timbre across the dynamic range.

    blend=0.0: no change (per-velocity spectral shapes unchanged)
    blend=1.0: all velocities get ref_vel's spectral shape (only loudness differs)
    blend=0.5: halfway blend (recommended starting point)

    Total energy (RMS of A0 vector) per velocity is preserved exactly.
    """
    samples = data["samples"]

    midi_to_keys: dict[int, list[str]] = defaultdict(list)
    for key, s in samples.items():
        midi = s.get("midi")
        if midi is not None:
            midi_to_keys[midi].append(key)

    n_blended = 0

    for midi in sorted(midi_to_keys.keys()):
        # Find reference velocity sample
        ref_key = f"m{midi:03d}_vel{ref_vel}"
        ref_sample = samples.get(ref_key)
        if not ref_sample:
            continue

        ref_partials = {p["k"]: p for p in ref_sample.get("partials", []) if "k" in p}
        all_k = list(ref_partials.keys())
        if not all_k:
            continue

        # Reference normalized spectral shape (unit-energy)
        ref_a0 = np.array([ref_partials[k].get("A0", 0.0) or 0.0 for k in all_k])
        ref_norm = np.linalg.norm(ref_a0)
        if ref_norm < 1e-9:
            continue
        ref_shape = ref_a0 / ref_norm   # unit vector

        for key in midi_to_keys[midi]:
            s = samples[key]
            vel = s.get("vel")
            if vel == ref_vel:
                continue   # reference velocity unchanged

            part_map = {p["k"]: p for p in s.get("partials", []) if "k" in p}

            # Build A0 vector aligned to ref_k order (0 for missing partials)
            a0_vec = np.array([
                (part_map[k].get("A0", 0.0) or 0.0) if k in part_map else 0.0
                for k in all_k
            ])
            total_energy = np.linalg.norm(a0_vec)
            if total_energy < 1e-9:
                continue

            # Normalized shape for this velocity
            vel_shape = a0_vec / total_energy

            # Blend shapes, re-normalise, scale back to original total energy
            blended_shape = (1.0 - blend) * vel_shape + blend * ref_shape
            blended_norm = np.linalg.norm(blended_shape)
            if blended_norm < 1e-9:
                continue
            new_a0_vec = blended_shape / blended_norm * total_energy

            # Write back
            for i, k in enumerate(all_k):
                if k in part_map and new_a0_vec[i] > 0:
                    part_map[k]["A0"] = round(float(new_a0_vec[i]), 6)
                    n_blended += 1

    if verbose:
        print(f"Color blend={blend:.2f} (ref vel{ref_vel}): {n_blended} A0 values updated")

    return data


def print_before_after(data_orig: dict, data_new: dict, midi: int):
    """Print tau1 comparison for a specific MIDI note."""
    print(f"\n=== MIDI {midi} --- tau1 before -> after smoothing ===")
    for vel in range(8):
        key = f"m{midi:03d}_vel{vel}"
        s_orig = data_orig["samples"].get(key)
        s_new  = data_new["samples"].get(key)
        if not s_orig:
            continue
        print(f"  vel{vel}:")
        for k_t in range(1, 9):
            p_orig = next((p for p in s_orig.get("partials", []) if p.get("k") == k_t), None)
            p_new  = next((p for p in s_new.get("partials", [])  if p.get("k") == k_t), None)
            if p_orig and p_new:
                t_old = p_orig.get("tau1", 0)
                t_new = p_new.get("tau1", 0)
                changed = " <--" if abs(t_old - t_new) > 0.005 else ""
                print(f"    k={k_t}: {t_old:.3f}s -> {t_new:.3f}s{changed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",   dest="inp",  default="analysis/params.json")
    ap.add_argument("--out",  dest="out",  default="analysis/params_smoothed.json")
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite input file instead of writing separate output")
    ap.add_argument("--check", type=int, metavar="MIDI",
                    help="Print before/after for this MIDI note")
    ap.add_argument("--snr-threshold", type=float, default=0.25,
                    help="A0 reliability threshold for tau blending (default 0.25)")
    ap.add_argument("--color-blend", type=float, default=0.0,
                    help="Blend spectral color toward ref velocity (0=off, 1=full). Default 0.")
    ap.add_argument("--color-ref", type=int, default=4,
                    help="Reference velocity for color blending (default 4)")
    args = ap.parse_args()

    inp_path = Path(args.inp)
    print(f"Reading {inp_path} ...")
    data_orig = json.loads(inp_path.read_text())

    import copy
    data_new = copy.deepcopy(data_orig)

    smooth_params(data_new, snr_threshold=args.snr_threshold, verbose=True)

    if args.color_blend > 0.0:
        blend_color_toward_reference(
            data_new, ref_vel=args.color_ref, blend=args.color_blend, verbose=True
        )

    if args.check:
        print_before_after(data_orig, data_new, args.check)

    out_path = inp_path if args.inplace else Path(args.out)
    out_path.write_text(json.dumps(data_new, indent=2, ensure_ascii=False))
    print(f"\nWritten -> {out_path}")


if __name__ == "__main__":
    main()
