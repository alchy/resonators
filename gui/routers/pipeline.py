"""
gui/routers/pipeline.py
────────────────────────
Full analysis pipeline: extract params → spectral EQ → train profile.
Each step runs as a subprocess for isolation and real-time stdout streaming.

POST /api/pipeline/run    — start pipeline (all steps or from a specific step)
GET  /api/pipeline/status — poll step progress
POST /api/pipeline/cancel — interrupt current step + subprocess
"""

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.logger import get_logger

log = get_logger("gui.pipeline")
router = APIRouter()

PYTHON = sys.executable
STEP_NAMES = ["extract", "eq", "train"]

# Single global pipeline job
_job: dict = {"status": "idle"}


class PipelineRequest(BaseModel):
    wav_dir:   str  = "C:/SoundBanks/IthacaPlayer/ks-grand"
    out:       str  = "analysis/params_profile.json"
    epochs:    int  = 300
    kmax:      int  = 16
    workers:   int  = 4
    from_step: str  = "extract"   # "extract" | "eq" | "train"


def _make_fresh_job() -> dict:
    return {
        "status":      "idle",
        "step":        None,
        "step_idx":    -1,
        "cancel":      False,
        "error":       None,
        "started_at":  None,
        "finished_at": None,
        "steps": {
            "extract": {"status": "idle", "log_lines": [], "progress_pct": 0, "rc": None},
            "eq":      {"status": "idle", "log_lines": [], "progress_pct": 0, "rc": None},
            "train":   {"status": "idle", "log_lines": [], "progress_pct": 0, "rc": None,
                        "epoch": 0, "total": 0, "loss": None},
        },
    }


def _parse_train_line(line: str, step_state: dict) -> None:
    """Extract epoch/loss from train_instrument_profile.py output."""
    m = re.search(r'[Ee]poch\s+(\d+)\s*/\s*(\d+)', line)
    if m:
        step_state["epoch"] = int(m.group(1))
        step_state["total"] = int(m.group(2))
        if step_state["total"] > 0:
            step_state["progress_pct"] = round(100 * step_state["epoch"] / step_state["total"])
    m2 = re.search(r'loss[=:\s]+([\d.]+(?:e[+\-]?\d+)?)', line, re.IGNORECASE)
    if m2:
        try:
            step_state["loss"] = float(m2.group(1))
        except ValueError:
            pass


def _parse_extract_line(line: str, step_state: dict) -> None:
    """Extract progress from extract_params.py output: 'nnn/704' pattern."""
    m = re.search(r'(\d+)\s*/\s*(\d+)', line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            step_state["progress_pct"] = round(100 * done / total)


def _stream_subprocess(cmd: list, step_state: dict, job: dict) -> None:
    """Run subprocess, stream stdout+stderr into step_state log_lines."""
    step_state["status"] = "running"
    step_state["log_lines"] = []
    is_train = step_state is job["steps"]["train"]
    is_extract = step_state is job["steps"]["extract"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            cwd=str(Path(__file__).parent.parent.parent),
        )
        job["_proc"] = proc

        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            step_state["log_lines"].append(line)
            if len(step_state["log_lines"]) > 80:
                step_state["log_lines"] = step_state["log_lines"][-80:]
            if is_train:
                _parse_train_line(line, step_state)
            elif is_extract:
                _parse_extract_line(line, step_state)

        proc.wait()
        step_state["rc"] = proc.returncode
        step_state["status"] = "done" if proc.returncode == 0 else "error"
        if proc.returncode == 0:
            step_state["progress_pct"] = 100

    except Exception as e:
        step_state["status"] = "error"
        step_state["log_lines"].append(f"Exception: {e}")
    finally:
        job.pop("_proc", None)


def _run_pipeline(req: PipelineRequest) -> None:
    job = _job
    job["started_at"] = time.time()
    job["status"] = "running"

    from_idx = STEP_NAMES.index(req.from_step) if req.from_step in STEP_NAMES else 0

    for idx, step in enumerate(STEP_NAMES):
        if idx < from_idx:
            job["steps"][step]["status"] = "skipped"

    for idx, step in enumerate(STEP_NAMES[from_idx:], start=from_idx):
        if job.get("cancel"):
            job["status"] = "cancelled"
            return

        job["step"] = step
        job["step_idx"] = idx
        step_state = job["steps"][step]

        if step == "extract":
            cmd = [PYTHON, "-u", "analysis/extract_params.py",
                   "--bank", req.wav_dir,
                   "--out", "analysis/params.json",
                   "--workers", str(req.workers)]
        elif step == "eq":
            cmd = [PYTHON, "-u", "analysis/compute_spectral_eq.py",
                   "--params", "analysis/params.json",
                   "--bank", req.wav_dir,
                   "--workers", str(req.workers)]
        else:  # train
            cmd = [PYTHON, "-u", "analysis/train_instrument_profile.py",
                   "--in", "analysis/params.json",
                   "--out", req.out,
                   "--epochs", str(req.epochs)]

        log.info(f"Pipeline [{step}]: {' '.join(cmd)}")
        _stream_subprocess(cmd, step_state, job)

        if step_state["status"] == "error":
            job["status"] = "error"
            job["error"] = f"Step '{step}' failed (rc={step_state.get('rc')})"
            log.error(job["error"])
            return

    job["status"] = "done"
    job["step"] = None
    job["finished_at"] = time.time()
    log.info("Pipeline complete")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/pipeline/run")
def start_pipeline(body: PipelineRequest):
    if _job.get("status") == "running":
        raise HTTPException(409, "Pipeline already running")
    _job.clear()
    _job.update(_make_fresh_job())
    t = threading.Thread(target=_run_pipeline, args=(body,), daemon=True)
    t.start()
    return {"started": True}


@router.get("/pipeline/status")
def get_pipeline_status():
    j = dict(_job)
    j["steps"] = {k: dict(v) for k, v in _job.get("steps", {}).items()}
    return j


@router.post("/pipeline/cancel")
def cancel_pipeline():
    if _job.get("status") != "running":
        raise HTTPException(400, "No active pipeline")
    _job["cancel"] = True
    proc = _job.get("_proc")
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
    return {"cancelling": True}
