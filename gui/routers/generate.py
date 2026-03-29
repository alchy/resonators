"""
gui/routers/generate.py
────────────────────────
Generation endpoints: trigger synthesis for a range of notes/velocities.
Jobs run in a background thread; status polled via GET .../generate/status.

Synthesis chain (all in-process, no subprocess):
  analysis.physics_synth.synthesize_note()
    ← resolve_note_params(cfg, midi)   (global + per-note deltas)
    ← velocity RMS profile             (from cfg or gamma fallback)
    ← spectral color blend             (vel_color_blend / vel_color_ref)
    ← tau1 k=1 scale                   (per-note override)
  → gui/sessions/{name}/generated/m{midi:03d}-vel{v}-f{sr_code}.wav
  → gui/sessions/{name}/generated/instrument-definition.json

Equivalent CLI: python -u analysis/generate-samples.py --params ... --session ...

Endpoints:
  POST /api/sessions/{name}/generate          start generation job
  GET  /api/sessions/{name}/generate/status   poll progress
  POST /api/sessions/{name}/generate/cancel   cancel running job
  GET  /api/sessions/{name}/files             list generated WAV files
"""

import copy
import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.config_schema import resolve_note_params, midi_to_name
from gui.logger import get_logger
from gui.routers.sessions import load_config, session_dir

log = get_logger("gui.generate")

router = APIRouter()

# In-memory job registry (one job per session at a time)
_jobs: dict[str, dict] = {}


def _apply_color_blend(sample: dict, ref_sample: dict, blend: float) -> dict:
    """
    Blend sample's A0 amplitude ratios toward ref_sample's, preserving total energy.
    Returns a (possibly new) sample dict.
    """
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


def _load_params(cfg: dict, name: str, params_file: str = "") -> dict:
    """Load params JSON. params_file overrides session source_params if given."""
    if params_file:
        p = Path(params_file)
    else:
        p = Path(cfg.get("source_params", ""))
    if not p.exists():
        p = session_dir(name) / "params.json"
    return json.loads(p.read_text())


def _run_job(session_name: str, midi_range: list[int], vel_layers: list[int], cfg: dict,
             params_file: str = ""):
    from analysis.physics_synth import synthesize_note

    job = _jobs[session_name]
    job["status"] = "running"
    job["done"] = 0
    job["total"] = len(midi_range) * len(vel_layers)
    job["errors"] = []
    log.info(f"[{session_name}] Job started: {len(midi_range)} notes × {len(vel_layers)} vel = {job['total']} samples")

    params_data = _load_params(cfg, session_name, params_file)
    out_dir = session_dir(session_name) / "generated"
    out_dir.mkdir(exist_ok=True)

    for midi in midi_range:
        for vel in vel_layers:
            if job.get("cancelled"):
                job["status"] = "cancelled"
                return
            key = f"m{midi:03d}_vel{vel}"
            sample = params_data["samples"].get(key)
            if sample is None:
                job["done"] += 1
                continue
            try:
                kwargs = resolve_note_params(cfg, midi)
                tau_scale = kwargs.pop("_tau1_k1_scale", 1.0)
                color_blend = kwargs.pop("vel_color_blend", 0.0) or 0.0
                color_ref   = int(kwargs.pop("vel_color_ref", 4) or 4)

                # Apply spectral color blending toward reference velocity
                if color_blend > 0.0:
                    ref_key    = f"m{midi:03d}_vel{color_ref}"
                    ref_sample = params_data["samples"].get(ref_key)
                    sample = _apply_color_blend(sample, ref_sample, color_blend)

                # Apply tau1_k1_scale to k=1 partial
                if tau_scale != 1.0:
                    sample = copy.deepcopy(sample)
                    for p in sample.get("partials", []):
                        if p.get("k") == 1:
                            p["tau1"] = (p.get("tau1") or 3.0) * tau_scale

                # Velocity scaling: use sample-derived profile if available,
                # otherwise fall back to gamma power curve.
                gamma = kwargs.pop("velocity_curve_gamma", 0.7)
                base_rms = kwargs.get("target_rms", 0.06) or 0.06
                vel_profile = cfg.get("velocity_rms_profile", {})
                if vel_profile and str(vel) in vel_profile:
                    vel_ratio = float(vel_profile[str(vel)])
                else:
                    vel_fraction = (vel + 1) / 8.0
                    vel_ratio = vel_fraction ** gamma
                kwargs["target_rms"] = float(base_rms * vel_ratio)

                # Remove keys not accepted by synthesize_note
                allowed = {
                    "duration", "sr", "soundboard_strength", "beat_scale",
                    "pan_spread", "eq_strength", "eq_freq_min", "stereo_boost",
                    "harmonic_brightness", "fade_out", "target_rms",
                    "noise_level", "stereo_decorr", "onset_ms",
                }
                filtered = {k: v for k, v in kwargs.items() if k in allowed}

                # Deterministic seed: same note/vel always produces identical stereo field
                filtered["rng_seed"] = midi * 100 + vel
                audio = synthesize_note(sample, **filtered)
                sr_val = int(filtered.get("sr", 44100))
                sr_code = 48 if sr_val >= 48000 else 44
                fname = f"m{midi:03d}-vel{vel}-f{sr_code}.wav"
                sf.write(str(out_dir / fname), audio, sr_val)
                job["last_file"] = fname
            except Exception as e:
                log.error(f"[{session_name}] {key}: {e}")
                job["errors"].append(f"{key}: {e}")
            job["done"] += 1

    job["status"] = "done"
    job["finished_at"] = time.time()
    log.info(f"[{session_name}] Job done: {job['done']} files, {len(job['errors'])} errors")

    # Write instrument-definition.json into the generated directory
    n_files = len(list(out_dir.glob("*.wav")))
    meta = cfg.get("instrument_meta", {})
    instrument_def = {
        "instrumentName":    meta.get("instrumentName", session_name),
        "velocityMaps":      str(len(job.get("vel_layers", [3]))),
        "instrumentVersion": meta.get("instrumentVersion", "1"),
        "author":            meta.get("author", "Unknown"),
        "description":       meta.get("description", "N/A"),
        "category":          meta.get("category", "Piano"),
        "sampleCount":       n_files,
    }
    (out_dir / "instrument-definition.json").write_text(
        json.dumps(instrument_def, indent=2, ensure_ascii=False)
    )
    log.info(f"[{session_name}] instrument-definition.json written ({n_files} samples)")


# ── Endpoints ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    midi_from:   int = 21
    midi_to:     int = 108
    vel_layers:  list[int] = [3]
    params_file: str = ""   # override session source_params; "" = use session default


@router.post("/{name}/generate")
def start_generate(name: str, body: GenerateRequest):
    cfg = load_config(name)
    if name in _jobs and _jobs[name].get("status") == "running":
        raise HTTPException(409, "Generation already running for this session")

    midi_range = list(range(body.midi_from, body.midi_to + 1))
    vel_layers = [v for v in body.vel_layers if 0 <= v <= 7]

    # Resolve effective params path for display in status
    effective_params = body.params_file or cfg.get("source_params", "")

    _jobs[name] = {
        "status": "starting",
        "done": 0,
        "total": len(midi_range) * len(vel_layers),
        "errors": [],
        "last_file": None,
        "started_at": time.time(),
        "finished_at": None,
        "cancelled": False,
        "midi_from": body.midi_from,
        "midi_to": body.midi_to,
        "vel_layers": vel_layers,
        "params_file": effective_params,
    }

    t = threading.Thread(
        target=_run_job,
        args=(name, midi_range, vel_layers, cfg, body.params_file),
        daemon=True,
    )
    t.start()
    return _jobs[name]


@router.get("/{name}/generate/status")
def get_generate_status(name: str):
    if name not in _jobs:
        return {"status": "idle"}
    j = _jobs[name]
    pct = round(100 * j["done"] / j["total"]) if j["total"] > 0 else 0
    return {**j, "progress_pct": pct}


@router.post("/{name}/generate/cancel")
def cancel_generate(name: str):
    if name not in _jobs or _jobs[name].get("status") != "running":
        raise HTTPException(400, "No running job to cancel")
    _jobs[name]["cancelled"] = True
    return {"cancelled": True}


@router.get("/{name}/files")
def list_files(name: str):
    gen_dir = session_dir(name) / "generated"
    if not gen_dir.exists():
        return {"files": []}
    files = []
    for f in sorted(gen_dir.glob("*.wav")):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "url": f"/audio/{name}/generated/{f.name}",
        })
    return {"files": files}
