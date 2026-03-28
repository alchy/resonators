"""
gui/server.py
─────────────
FastAPI backend for the Resonator Synthesizer GUI.
Run: python gui/server.py   (or: uvicorn gui.server:app --reload --port 8989)
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

# Redirect uvicorn logs into same rotating file
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _uv = logging.getLogger(_name)
    from logging.handlers import RotatingFileHandler
    from pathlib import Path as _P
    _fh = RotatingFileHandler(_P("gui/logs/server.log"), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    _uv.addHandler(_fh)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gui.routers import sessions, generate, audio, profile

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

# Serve generated audio from all sessions
app.mount("/audio", StaticFiles(directory="gui/sessions"), name="audio")

# Serve frontend last
app.mount("/", StaticFiles(directory="gui/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gui.server:app", host="0.0.0.0", port=8989, reload=True)
