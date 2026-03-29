"""
gui/server.py
─────────────
FastAPI backend for the Resonator Synthesizer GUI.

Run:
    python gui/server.py
    uvicorn gui.server:app --reload --port 8989

Routers mounted:
  /api/sessions/*   sessions.py  — session CRUD, config, per-note overrides
  /api/sessions/*   generate.py  — batch synthesis jobs
  /api/sessions/*   audio.py     — audio file serving / spectrum analysis
  /api/             profile.py   — param profile listing, EGRB model status
  /api/             pipeline.py  — analysis pipeline (extract/eq/train), SSE logs
  /audio/*          static       — generated WAV files (gui/sessions/*/generated/)
  /                 static       — frontend (gui/static/)

Logging:
  gui/logs/server.log  — FastAPI + uvicorn internal logs (plain FileHandler,
                          RotatingFileHandler avoided: WinError 32 on Windows
                          when reloader + worker both hold the file open)
  runtime-logs/*.txt   — stdout tee from analysis subprocesses (auto-created)
"""

import logging
import sys
from pathlib import Path

# Allow import of analysis.* from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from gui.logger import get_logger

log = get_logger("gui.server")

# Redirect uvicorn logs into same file (plain FileHandler — RotatingFileHandler
# causes WinError 32 on Windows because the reloader + worker both hold the file open)
_log_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", "%Y-%m-%d %H:%M:%S")
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _uv = logging.getLogger(_name)
    _fh = logging.FileHandler(Path("gui/logs/server.log"), encoding="utf-8")
    _fh.setFormatter(_log_fmt)
    _uv.addHandler(_fh)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gui.routers import sessions, generate, audio, profile, pipeline

app = FastAPI(title="Resonator Synth GUI", version="1.0.0")
log.info("Resonator Synth GUI starting on port 8989")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(generate.router, prefix="/api/sessions", tags=["generate"])
app.include_router(audio.router,    prefix="/api/sessions", tags=["audio"])
app.include_router(profile.router,  prefix="/api",          tags=["profile"])
app.include_router(pipeline.router, prefix="/api",          tags=["pipeline"])

# Serve generated audio from all sessions
app.mount("/audio", StaticFiles(directory="gui/sessions"), name="audio")

# Serve frontend last
app.mount("/", StaticFiles(directory="gui/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gui.server:app", host="0.0.0.0", port=8989, reload=True)
