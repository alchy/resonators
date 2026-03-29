"""
gui/routers/sessions.py
────────────────────────
Session CRUD: create, list, delete, read/write config, per-note overrides.

Session layout:
  gui/sessions/{name}/
    config.json      render/timbre/stereo/per_note/velocity_rms_profile
    params.json      copy of source_params made at session creation
    generated/
      m{midi:03d}-vel{v}-f{sr_code}.wav   synthesized samples
      instrument-definition.json          metadata for player

Endpoints:
  GET    /api/sessions/banks               list bank subdirectories in a parent dir
  GET    /api/sessions                     list sessions
  POST   /api/sessions                     create session
  DELETE /api/sessions/{name}              delete session + generated files
  GET    /api/sessions/{name}/config       get config + param metadata
  PUT    /api/sessions/{name}/config       update render/timbre/stereo/vel_profile
  GET    /api/sessions/{name}/note/{midi}  get per-note overrides + resolved params
  PUT    /api/sessions/{name}/note/{midi}  set per-note overrides
  DELETE /api/sessions/{name}/note/{midi}  clear per-note overrides
  GET    /api/sessions/{name}/params       list all notes in session params.json
"""

import json
import shutil  # used by delete_session
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.config_schema import (
    PARAM_META, PER_NOTE_DELTA_META, default_config,
    resolve_note_params, midi_to_name,
)

router = APIRouter()

SESSIONS_DIR = Path("gui/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── Bank browser ─────────────────────────────────────────────────────────────

@router.get("/banks")
def list_banks(dir: str = "C:/SoundBanks/IthacaPlayer"):
    """List subdirectories of `dir` as candidate sample banks.

    For each subdirectory returns:
      name, path, wav_count (capped at 10 for speed), has_definition,
      definition (parsed instrument-definition.json or null).
    """
    p = Path(dir)
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"Directory not found: {dir}")
    banks = []
    for subdir in sorted(p.iterdir()):
        if not subdir.is_dir():
            continue
        # Count WAVs quickly — stop at 10 to keep it fast
        wav_count = 0
        for f in subdir.iterdir():
            if f.suffix.lower() == ".wav":
                wav_count += 1
                if wav_count >= 10:
                    break
        def_path = subdir / "instrument-definition.json"
        definition = None
        if def_path.exists():
            try:
                definition = json.loads(def_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        banks.append({
            "name": subdir.name,
            "path": str(subdir).replace("\\", "/"),
            "wav_count": wav_count,
            "has_definition": def_path.exists(),
            "definition": definition,
        })
    return {"dir": str(p).replace("\\", "/"), "banks": banks}


def session_dir(name: str) -> Path:
    return SESSIONS_DIR / name


def load_config(name: str) -> dict:
    path = session_dir(name) / "config.json"
    if not path.exists():
        raise HTTPException(404, f"Session '{name}' not found")
    return json.loads(path.read_text())


def save_config(name: str, config: dict):
    path = session_dir(name) / "config.json"
    path.write_text(json.dumps(config, indent=2))


# ── List / create ────────────────────────────────────────────────────────────

@router.get("")
def list_sessions():
    sessions = []
    for d in sorted(SESSIONS_DIR.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
            gen_dir = d / "generated"
            n_files = len(list(gen_dir.glob("*.wav"))) if gen_dir.exists() else 0
            sessions.append({
                "name": d.name,
                "source_params": cfg.get("source_params", ""),
                "n_generated": n_files,
            })
    return sessions


class CreateSession(BaseModel):
    name: str                            # bank name, e.g. "ks-grand"
    instrument_meta: Optional[dict] = None  # pre-filled from instrument-definition.json
    wav_dir: str = ""                    # source WAV bank path, stored for pipeline restore


@router.post("")
def create_session(body: CreateSession):
    name = body.name.strip().replace(" ", "_").lower()
    d = session_dir(name)
    if d.exists():
        raise HTTPException(400, f"Session '{name}' already exists")

    d.mkdir(parents=True)
    (d / "generated").mkdir()

    # source_params points directly at the trained profile; may not exist yet
    # (pipeline has not been run yet) — will be updated by apply_pipeline
    source_params = f"analysis/params-nn-profile-{name}.json"

    cfg = default_config(source_params)

    # Use caller-supplied metadata (from instrument-definition.json) if provided
    if body.instrument_meta:
        meta = {
            "instrumentName":    body.instrument_meta.get("instrumentName", name),
            "author":            body.instrument_meta.get("author", "n/a"),
            "category":          body.instrument_meta.get("category", "Piano"),
            "instrumentVersion": str(body.instrument_meta.get("instrumentVersion", "1")),
            "description":       body.instrument_meta.get("description", "n/a"),
            "velocityMaps":      str(body.instrument_meta.get("velocityMaps", "8")),
            "sampleCount":       0,
        }
    else:
        meta = {
            "instrumentName":    name,
            "author":            "n/a",
            "category":          "Piano",
            "instrumentVersion": "1",
            "description":       "n/a",
            "velocityMaps":      "8",
            "sampleCount":       0,
        }
    cfg["instrument_meta"] = meta
    if body.wav_dir:
        cfg["wav_dir"] = body.wav_dir
    save_config(name, cfg)
    return {"name": name, "created": True}


@router.delete("/{name}")
def delete_session(name: str):
    d = session_dir(name)
    if not d.exists():
        raise HTTPException(404, f"Session '{name}' not found")
    shutil.rmtree(d)
    return {"deleted": name}


# ── Config get / update ───────────────────────────────────────────────────────

@router.get("/{name}/config")
def get_config(name: str):
    cfg = load_config(name)
    return {
        "config": cfg,
        "param_meta": PARAM_META,
        "per_note_delta_meta": PER_NOTE_DELTA_META,
    }


@router.put("/{name}/config")
def update_config(name: str, body: dict):
    cfg = load_config(name)
    # Merge incoming changes into existing config sections
    for section in ("render", "timbre", "stereo"):
        if section in body:
            cfg[section].update(body[section])
    if "per_note" in body:
        cfg.setdefault("per_note", {}).update(body["per_note"])
    if "velocity_rms_profile" in body:
        cfg["velocity_rms_profile"] = body["velocity_rms_profile"]
    save_config(name, cfg)
    return cfg


# ── Per-note overrides ────────────────────────────────────────────────────────

@router.get("/{name}/note/{midi}")
def get_note_config(name: str, midi: int):
    cfg = load_config(name)
    note_overrides = cfg.get("per_note", {}).get(str(midi), {})
    resolved = resolve_note_params(cfg, midi)
    return {
        "midi": midi,
        "note_name": midi_to_name(midi),
        "overrides": note_overrides,
        "resolved": resolved,
        "per_note_delta_meta": PER_NOTE_DELTA_META,
    }


class NoteOverride(BaseModel):
    overrides: dict


@router.put("/{name}/note/{midi}")
def set_note_override(name: str, midi: int, body: NoteOverride):
    cfg = load_config(name)
    cfg.setdefault("per_note", {})[str(midi)] = body.overrides
    save_config(name, cfg)
    return {"midi": midi, "overrides": body.overrides}


@router.delete("/{name}/note/{midi}")
def clear_note_override(name: str, midi: int):
    cfg = load_config(name)
    cfg.get("per_note", {}).pop(str(midi), None)
    save_config(name, cfg)
    return {"midi": midi, "cleared": True}


# ── Params inspection ────────────────────────────────────────────────────────

@router.get("/{name}/params")
def get_params_summary(name: str):
    """Return list of all notes available in this session's params.json."""
    cfg = load_config(name)
    params_path = Path(cfg.get("source_params", session_dir(name) / "params.json"))
    if not params_path.exists():
        params_path = session_dir(name) / "params.json"
    data = json.loads(params_path.read_text())
    notes = []
    for key, s in data["samples"].items():
        notes.append({
            "key": key,
            "midi": s["midi"],
            "vel": s["vel"],
            "note_name": midi_to_name(s["midi"]),
            "n_partials": len(s.get("partials", [])),
            "has_eq": "spectral_eq" in s,
            "duration_s": s.get("duration_s"),
        })
    return {"notes": notes}


