"""
gui/routers/sessions.py
────────────────────────
Session CRUD endpoints.
"""

import json
import shutil
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
    name: str
    source_params: str = "analysis/params.json"


@router.post("")
def create_session(body: CreateSession):
    name = body.name.strip().replace(" ", "_").lower()
    d = session_dir(name)
    if d.exists():
        raise HTTPException(400, f"Session '{name}' already exists")

    d.mkdir(parents=True)
    (d / "generated").mkdir()

    # Copy source params into session
    src = Path(body.source_params)
    if not src.exists():
        raise HTTPException(400, f"source_params not found: {src}")
    shutil.copy(src, d / "params.json")

    cfg = default_config(str(d / "params.json"))
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
