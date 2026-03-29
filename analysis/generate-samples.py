"""
analysis/generate-samples.py
─────────────────────────────
CLI batch generator: synthesizes a range of MIDI notes × velocity layers
using the same physics engine as the GUI.

Exact CLI mirror of gui/routers/generate.py::_run_job() — use this to
reproduce or re-run a generation batch outside the GUI.

Synthesis chain:
  analysis.physics_synth.synthesize_note()
    ← resolve_note_params(session_config, midi)   global + per-note deltas
    ← velocity RMS profile from session config    (or gamma fallback)
    ← spectral color blend (vel_color_blend / vel_color_ref)
    ← tau1 k=1 scale (per-note override)
  → {out-dir}/m{midi:03d}-vel{v}-f{sr_code}.wav
  → {out-dir}/instrument-definition.json

Usage:
    python -u analysis/generate-samples.py \\
        --params  analysis/params-ks-grand.json \\
        --session gui/sessions/ks-grand/config.json \\
        --out-dir gui/sessions/ks-grand/generated \\
        --from    21  --to 108 \\
        --vel     0 1 2 3 4 5 6 7

Arguments:
  --params   Params JSON (output of train-instrument-profile or extract-params)
  --session  Session config.json (render/timbre/stereo/vel_profile settings)
             If omitted, factory defaults from config_schema are used.
  --out-dir  Output directory for WAV files
             Default: {session_dir}/generated, or ./generated if no session
  --from     First MIDI note to generate (default: 21 = A0)
  --to       Last MIDI note to generate (default: 108 = C8)
  --vel      Velocity layers to generate, space-separated (default: 0–7)
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow importing from project root (analysis.*, gui.*)
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.config_schema import resolve_note_params


def _apply_color_blend(sample: dict, ref_sample: dict, blend: float) -> dict:
    if blend <= 0.0 or ref_sample is None:
        return sample
    part_map = {p["k"]: p for p in sample.get("partials", []) if "k" in p}
    ref_map  = {p["k"]: p for p in ref_sample.get("partials", []) if "k" in p}
    all_k = sorted(set(part_map) & set(ref_map))
    if not all_k:
        return sample
    a0_vec  = np.array([part_map[k].get("A0", 0.0) or 0.0 for k in all_k])
    ref_vec = np.array([ref_map[k].get("A0",  0.0) or 0.0 for k in all_k])
    total_e  = np.linalg.norm(a0_vec)
    ref_norm = np.linalg.norm(ref_vec)
    if total_e < 1e-9 or ref_norm < 1e-9:
        return sample
    vel_shape = a0_vec / total_e
    ref_shape = ref_vec / ref_norm
    blended   = (1.0 - blend) * vel_shape + blend * ref_shape
    blended  /= max(np.linalg.norm(blended), 1e-9)
    new_a0    = blended * total_e
    sample = copy.deepcopy(sample)
    pm = {p["k"]: p for p in sample.get("partials", []) if "k" in p}
    for i, k in enumerate(all_k):
        if k in pm and new_a0[i] > 0:
            pm[k]["A0"] = float(new_a0[i])
    return sample


def main():
    parser = argparse.ArgumentParser(
        description="Batch-synthesize WAV samples from physics params.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params",  default="analysis/params.json",
                        help="Extracted/profiled params JSON (params_out from pipeline)")
    parser.add_argument("--session", default=None,
                        help="Session config.json (render/timbre/stereo/vel_profile). "
                             "If omitted, factory defaults are used.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for WAV files. "
                             "Defaults to same dir as --session + /generated, "
                             "or ./generated if no session given.")
    parser.add_argument("--from",    dest="midi_from", type=int, default=21,
                        metavar="MIDI", help="First MIDI note (inclusive)")
    parser.add_argument("--to",      dest="midi_to",   type=int, default=108,
                        metavar="MIDI", help="Last MIDI note (inclusive)")
    parser.add_argument("--vel",     dest="vel_layers", type=int, nargs="+",
                        default=list(range(8)), metavar="V",
                        help="Velocity layers to generate (0-7)")
    args = parser.parse_args()

    # ── Load params ────────────────────────────────────────────────────────────
    params_path = Path(args.params)
    if not params_path.exists():
        sys.exit(f"ERROR: params file not found: {params_path}")
    params_data = json.loads(params_path.read_text())

    # ── Load session config ────────────────────────────────────────────────────
    if args.session:
        cfg_path = Path(args.session)
        if not cfg_path.exists():
            sys.exit(f"ERROR: session config not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
    else:
        from gui.config_schema import default_config
        cfg = default_config(str(params_path))
        print("No --session given; using factory defaults.", flush=True)

    # ── Output directory ───────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.session:
        out_dir = Path(args.session).parent / "generated"
    else:
        out_dir = Path("generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Synthesis ──────────────────────────────────────────────────────────────
    from analysis.physics_synth import synthesize_note

    midi_range = list(range(args.midi_from, args.midi_to + 1))
    vel_layers = [v for v in args.vel_layers if 0 <= v <= 7]
    total      = len(midi_range) * len(vel_layers)
    done       = 0
    errors     = []

    print(f"Generating {len(midi_range)} notes × {len(vel_layers)} vel = {total} samples", flush=True)
    print(f"Output → {out_dir}", flush=True)

    for midi in midi_range:
        for vel in vel_layers:
            key    = f"m{midi:03d}_vel{vel}"
            sample = params_data["samples"].get(key)
            if sample is None:
                done += 1
                continue
            try:
                kwargs      = resolve_note_params(cfg, midi)
                tau_scale   = kwargs.pop("_tau1_k1_scale", 1.0)
                color_blend = kwargs.pop("vel_color_blend", 0.0) or 0.0
                color_ref   = int(kwargs.pop("vel_color_ref", 4) or 4)

                if color_blend > 0.0:
                    ref_key    = f"m{midi:03d}_vel{color_ref}"
                    ref_sample = params_data["samples"].get(ref_key)
                    sample     = _apply_color_blend(sample, ref_sample, color_blend)

                if tau_scale != 1.0:
                    sample = copy.deepcopy(sample)
                    for p in sample.get("partials", []):
                        if p.get("k") == 1:
                            p["tau1"] = (p.get("tau1") or 3.0) * tau_scale

                gamma        = kwargs.pop("velocity_curve_gamma", 0.7)
                base_rms     = kwargs.get("target_rms", 0.06) or 0.06
                vel_profile  = cfg.get("velocity_rms_profile", {})
                if vel_profile and str(vel) in vel_profile:
                    vel_ratio = float(vel_profile[str(vel)])
                else:
                    vel_ratio = ((vel + 1) / 8.0) ** gamma
                kwargs["target_rms"] = float(base_rms * vel_ratio)

                allowed = {
                    "duration", "sr", "soundboard_strength", "beat_scale",
                    "pan_spread", "eq_strength", "eq_freq_min", "stereo_boost",
                    "harmonic_brightness", "fade_out", "target_rms",
                    "noise_level", "stereo_decorr", "onset_ms",
                }
                filtered = {k: v for k, v in kwargs.items() if k in allowed}
                filtered["rng_seed"] = midi * 100 + vel

                audio  = synthesize_note(sample, **filtered)
                sr_val = int(filtered.get("sr", 44100))
                sr_code = 48 if sr_val >= 48000 else 44
                fname  = f"m{midi:03d}-vel{vel}-f{sr_code}.wav"
                sf.write(str(out_dir / fname), audio, sr_val)

            except Exception as e:
                errors.append(f"{key}: {e}")
                print(f"  ERROR {key}: {e}", flush=True)

            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)

    # ── instrument-definition.json ─────────────────────────────────────────────
    n_files = len(list(out_dir.glob("*.wav")))
    meta    = cfg.get("instrument_meta", {})
    session_name = Path(args.session).parent.name if args.session else out_dir.name
    instrument_def = {
        "instrumentName":    meta.get("instrumentName", session_name),
        "velocityMaps":      str(len(vel_layers)),
        "instrumentVersion": meta.get("instrumentVersion", "1"),
        "author":            meta.get("author", "Unknown"),
        "description":       meta.get("description", "N/A"),
        "category":          meta.get("category", "Piano"),
        "sampleCount":       n_files,
    }
    (out_dir / "instrument-definition.json").write_text(
        json.dumps(instrument_def, indent=2, ensure_ascii=False)
    )

    print(f"\nDone: {done} processed, {len(errors)} errors", flush=True)
    print(f"instrument-definition.json written ({n_files} samples)", flush=True)
    if errors:
        print("\nErrors:", flush=True)
        for e in errors:
            print(f"  {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
