"""
gui/routers/generate.py
────────────────────────
Generation endpoints: trigger synthesis for a range of notes/velocities.
Jobs run in a background thread; status polled via GET .../generate/status.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

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


def _load_params(cfg: dict, name: str) -> dict:
    p = Path(cfg.get("source_params", ""))
    if not p.exists():
        p = session_dir(name) / "params.json"
    return json.loads(p.read_text())


def _run_job(session_name: str, midi_range: list[int], vel_layers: list[int], cfg: dict):
    from analysis.physics_synth import synthesize_note

    job = _jobs[session_name]
    job["status"] = "running"
    job["done"] = 0
    job["total"] = len(midi_range) * len(vel_layers)
    job["errors"] = []
    log.info(f"[{session_name}] Job started: {len(midi_range)} notes × {len(vel_layers)} vel = {job['total']} samples")

    params_data = _load_params(cfg, session_name)
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

                # Apply tau1_k1_scale to k=1 partial in a copy of sample
                if tau_scale != 1.0:
                    import copy
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
                    "pan_spread", "eq_strength", "stereo_boost",
                    "harmonic_brightness", "fade_out", "target_rms",
                }
                filtered = {k: v for k, v in kwargs.items() if k in allowed}

                audio = synthesize_note(sample, **filtered)
                fname = f"{midi_to_name(midi).replace('#','s')}_vel{vel}.wav"
                sf.write(str(out_dir / fname), audio, int(filtered.get("sr", 44100)))
                job["last_file"] = fname
            except Exception as e:
                log.error(f"[{session_name}] {key}: {e}")
                job["errors"].append(f"{key}: {e}")
            job["done"] += 1

    job["status"] = "done"
    job["finished_at"] = time.time()
    log.info(f"[{session_name}] Job done: {job['done']} files, {len(job['errors'])} errors")


# ── Endpoints ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    midi_from: int = 21
    midi_to: int = 108
    vel_layers: list[int] = [3]


@router.post("/{name}/generate")
def start_generate(name: str, body: GenerateRequest):
    cfg = load_config(name)
    if name in _jobs and _jobs[name].get("status") == "running":
        raise HTTPException(409, "Generation already running for this session")

    midi_range = list(range(body.midi_from, body.midi_to + 1))
    vel_layers = [v for v in body.vel_layers if 0 <= v <= 7]

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
    }

    t = threading.Thread(
        target=_run_job,
        args=(name, midi_range, vel_layers, cfg),
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
