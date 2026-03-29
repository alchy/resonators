"""
gui/routers/pipeline.py
────────────────────────
Full analysis pipeline: extract params → spectral EQ → e2e train (SetterNN).
Each step runs as a subprocess for isolation and real-time stdout streaming.

Output paths derived from WAV bank directory name (bank suffix):
  wav_dir     C:/SoundBanks/IthacaPlayer/salamander
  params_out  →  analysis/params-salamander.json
  checkpoint  →  checkpoints/e2e/best.pt

Endpoints:
  POST /api/pipeline/run          start pipeline (all steps or from a specific step)
  GET  /api/pipeline/status       poll step progress + log tail
  POST /api/pipeline/cancel       interrupt current step + subprocess
  GET  /api/pipeline/log/{step}   last N lines from runtime-logs/{step}-log.txt
  GET  /api/pipeline/log-stream/{step}  SSE tail of log file (EventSource)
  GET  /api/pipeline/egrb_status  E2E training state from runtime-logs/train-e2e-log.txt
  GET  /api/pipeline/vel_profile  velocity RMS ratios derived from A0 energies
  POST /api/pipeline/apply/{session}  point session at trained checkpoint

Logging:
  Analysis scripts tee stdout to runtime-logs/{step}-log.txt (created on first run).
  GUI backend internal logs go to gui/logs/server.log via gui.logger.
"""

import asyncio
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gui.logger import get_logger

log = get_logger("gui.pipeline")
router = APIRouter()

PYTHON = sys.executable
STEP_NAMES = ["extract", "eq", "train"]

# Single global pipeline job
_job: dict = {"status": "idle"}


class PipelineRequest(BaseModel):
    wav_dir:      str   = "C:/SoundBanks/IthacaPlayer/ks-grand"
    params_out:   str   = "analysis/params.json"
    e2e_out:      str   = "checkpoints/e2e"
    e2e_config:   str   = "config_e2e.json"
    workers:      int   = 4
    from_step:    str   = "extract"   # "extract" | "eq" | "train"


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
                        "epoch": 0, "total": 0, "loss": None, "phase_label": ""},
        },
    }


def _parse_train_line(line: str, step_state: dict) -> None:
    """
    Parse train_e2e.py output lines. Handles:
      Phase 0a:  "  0a epoch   5/20  B_loss=0.123456"
      Phase 0b:  "  0b epoch  40/80  total=87.1891  B=21.43 ..."
      Phase 1/2: "  epoch  200/300  train=0.1234  val=0.1456  lr=1.00e-04  10s"
    """
    # Detect phase label (0a / 0b / phase1 / phase2)
    m_phase = re.match(r'\s*(0[ab])\s+epoch', line)
    if m_phase:
        step_state["phase_label"] = m_phase.group(1)
    elif re.match(r'\s*epoch\s+\d+', line):
        # Determine from context: phase1 has "train=", phase2 also
        if step_state.get("phase_label") not in ("0a", "0b"):
            step_state["phase_label"] = "e2e"

    # Epoch progress
    m = re.search(r'epoch\s+(\d+)\s*/\s*(\d+)', line, re.IGNORECASE)
    if m:
        step_state["epoch"] = int(m.group(1))
        step_state["total"] = int(m.group(2))
        if step_state["total"] > 0:
            step_state["progress_pct"] = round(100 * step_state["epoch"] / step_state["total"])

    # Loss: try "val=", "train=", "total=", "loss="
    for pattern in (r'val=([\d.]+)', r'train=([\d.]+)', r'total=([\d.]+)', r'loss[=:\s]+([\d.]+)'):
        m2 = re.search(pattern, line)
        if m2:
            try:
                step_state["loss"] = float(m2.group(1))
            except ValueError:
                pass
            break


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
    is_train   = step_state is job["steps"]["train"]
    is_extract = step_state is job["steps"]["extract"]
    is_eq      = step_state is job["steps"]["eq"]

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
            elif is_extract or is_eq:
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
    job["params_out"] = req.params_out

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
            cmd = [PYTHON, "-u", "analysis/extract-params.py",
                   "--bank", req.wav_dir,
                   "--out", req.params_out,
                   "--workers", str(req.workers)]
        elif step == "eq":
            cmd = [PYTHON, "-u", "analysis/compute-spectral-eq.py",
                   "--params", req.params_out,
                   "--bank", req.wav_dir,
                   "--workers", str(req.workers)]
        else:  # train — e2e SetterNN training
            cmd = [PYTHON, "-u", "train_e2e.py",
                   "--config",  req.e2e_config,
                   "--params",  req.params_out,
                   "--bank",    req.wav_dir,
                   "--out",     req.e2e_out]
            job["e2e_out"] = req.e2e_out

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
        # Auto-compute velocity RMS profile from extracted A0 energies
    params_path = Path(req.params_out)
    if params_path.exists():
        vel_profile = _compute_vel_profile(params_path)
        job["vel_profile"] = vel_profile
        log.info(f"Velocity profile from A0 energy: {vel_profile}")

    job["status"] = "done"
    job["step"] = None
    job["finished_at"] = time.time()
    log.info("Pipeline complete")


def _compute_vel_profile(params_path: Path) -> dict:
    """
    Derive per-velocity RMS ratios from extracted A0 energies in params.json.
    energy(vel) = mean over MIDI of sqrt(sum(A0_k^2))
    Returns {str(vel): ratio} normalised so vel7 = 1.0.
    """
    import json as _json
    import math as _math

    samples = _json.loads(params_path.read_text()).get("samples", {})
    energy_sum = {v: 0.0 for v in range(8)}
    energy_cnt = {v: 0   for v in range(8)}

    for key, sdata in samples.items():
        # key format: m021_vel3
        try:
            vel = int(key.split("_vel")[1])
        except (IndexError, ValueError):
            continue
        a0_values = [p.get("A0", 0.0) or 0.0 for p in sdata.get("partials", [])]
        if not a0_values:
            continue
        energy = _math.sqrt(sum(x * x for x in a0_values))
        energy_sum[vel] += energy
        energy_cnt[vel] += 1

    mean_e = {}
    for v in range(8):
        mean_e[v] = energy_sum[v] / energy_cnt[v] if energy_cnt[v] > 0 else 0.0

    ref = mean_e.get(7, 0.0)
    if ref < 1e-9:
        # Fallback to gamma curve if no data at all
        gamma = 0.7
        return {str(v): round(((v + 1) / 8.0) ** gamma, 4) for v in range(8)}

    # Normalise
    raw = {v: mean_e[v] / ref for v in range(8)}

    # Interpolate missing velocity layers linearly from neighbours
    known = {v: raw[v] for v in range(8) if mean_e[v] > 1e-9}
    if len(known) < 2:
        gamma = 0.7
        return {str(v): round(((v + 1) / 8.0) ** gamma, 4) for v in range(8)}

    keys = sorted(known)
    for v in range(8):
        if v not in known:
            # find nearest lower and upper known key
            lo = max((k for k in keys if k < v), default=None)
            hi = min((k for k in keys if k > v), default=None)
            if lo is None:
                raw[v] = known[hi]
            elif hi is None:
                raw[v] = known[lo]
            else:
                t = (v - lo) / (hi - lo)
                raw[v] = known[lo] + t * (known[hi] - known[lo])

    return {str(v): round(raw[v], 4) for v in range(8)}


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
    j = {k: v for k, v in _job.items() if not k.startswith("_")}  # exclude _proc etc.
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


@router.get("/pipeline/egrb_status")
def get_egrb_status():
    """Read last epoch line of runtime-logs/train-e2e-log.txt and return E2E training state."""
    log_path = Path("runtime-logs/train-e2e-log.txt")
    if not log_path.exists():
        return {"status": "idle", "epoch": 0, "total": 0, "phase": None, "loss": None, "active": False}

    # Determine if training is active (log modified in last 120s)
    age_s = time.time() - log_path.stat().st_mtime
    active = age_s < 120

    # Find last line matching epoch pattern: "epoch  800/800  loss=..."
    last_line = ""
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if re.search(r'epoch\s+\d+\s*/\s*\d+', line, re.IGNORECASE):
                last_line = line.strip()

    if not last_line:
        return {"status": "idle", "epoch": 0, "total": 0, "phase": None, "loss": None, "active": False}

    m_ep = re.search(r'epoch\s+(\d+)\s*/\s*(\d+)', last_line, re.IGNORECASE)
    # Match val=, train=, total=, or loss= (in priority order)
    m_lo = (re.search(r'val=([\d.]+)', last_line) or
            re.search(r'train=([\d.]+)', last_line) or
            re.search(r'total=([\d.]+)', last_line) or
            re.search(r'loss=([\d.]+)', last_line))

    epoch = int(m_ep.group(1)) if m_ep else 0
    total = int(m_ep.group(2)) if m_ep else 0
    loss  = float(m_lo.group(1)) if m_lo else None

    return {
        "status": "running" if active else "stopped",
        "epoch":  epoch,
        "total":  total,
        "phase":  None,
        "loss":   loss,
        "active": active,
        "age_s":  int(age_s),
    }


_LOG_PATHS = {
    "extract": "runtime-logs/extract-params-log.txt",
    "eq":      "runtime-logs/spectral-eq-log.txt",
    "train":   "runtime-logs/train-e2e-log.txt",
}


@router.get("/pipeline/log/{step}")
def get_step_log(step: str, lines: int = 40):
    """Return last N lines from the runtime log for a pipeline step."""
    log_path = Path(_LOG_PATHS.get(step, ""))
    if not log_path.exists():
        return {"lines": [], "exists": False, "path": str(log_path)}
    content = log_path.read_text(encoding="utf-8", errors="replace")
    all_lines = content.splitlines()
    return {"lines": all_lines[-lines:], "exists": True, "path": str(log_path)}


@router.get("/pipeline/log-stream/{step}")
async def stream_step_log(step: str, request: Request):
    """SSE: tail log file and push new lines as they appear (JSON-encoded arrays)."""
    import json as _json
    log_path = Path(_LOG_PATHS.get(step, ""))

    async def event_gen():
        sent_bytes = 0
        # Send existing content first (last 80 lines), replace=True flag
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace")
            tail = content.splitlines()[-80:]
            if tail:
                payload = _json.dumps({"replace": True, "lines": tail})
                yield f"data: {payload}\n\n"
            sent_bytes = log_path.stat().st_size

        while True:
            if await request.is_disconnected():
                break
            try:
                if log_path.exists():
                    size = log_path.stat().st_size
                    if size > sent_bytes:
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(sent_bytes)
                            new_text = f.read()
                        sent_bytes = size
                        new_lines = [l for l in new_text.splitlines() if l.strip()]
                        if new_lines:
                            payload = _json.dumps({"replace": False, "lines": new_lines})
                            yield f"data: {payload}\n\n"
                    elif size < sent_bytes:
                        # File truncated (new run) — resend from beginning next tick
                        sent_bytes = 0
            except Exception:
                pass
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/pipeline/vel_profile")
def get_vel_profile():
    """Compute velocity RMS profile from current analysis/params.json."""
    p = Path("analysis/params.json")
    if not p.exists():
        raise HTTPException(404, "analysis/params.json not found — run Extract first")
    return {"vel_profile": _compute_vel_profile(p)}


@router.post("/pipeline/apply/{session_name}")
def apply_pipeline_to_session(session_name: str):
    """
    Apply completed pipeline output to a session:
    - Point source_params directly at the trained profile (no copy)
    - Write computed velocity profile into session config
    """
    from gui.routers.sessions import session_dir, load_config, save_config

    sdir = session_dir(session_name)
    if not sdir.exists():
        raise HTTPException(404, f"Session '{session_name}' not found")

    # Prefer e2e checkpoint (best.pt), fallback to raw params.json
    e2e_out = _job.get("e2e_out", "checkpoints/e2e")
    best_pt  = Path(e2e_out) / "best.pt"
    if best_pt.exists():
        src = best_pt
    else:
        params_out = _job.get("params_out", "analysis/params.json")
        src = Path(params_out)
        if not src.exists():
            raise HTTPException(400, "No e2e checkpoint and no params.json found")

    # Use forward slashes so the path is consistent on all platforms
    source_params = str(src).replace("\\", "/")

    cfg = load_config(session_name)
    cfg["source_params"] = source_params

    # Apply velocity profile from A0 energies
    params_src = Path(_job.get("params_out", "analysis/params.json"))
    if params_src.exists():
        cfg["velocity_rms_profile"] = _compute_vel_profile(params_src)

    save_config(session_name, cfg)
    log.info(f"Applied pipeline to session '{session_name}': source_params → {source_params}")
    return {"applied": True, "source_params": source_params,
            "vel_profile": cfg["velocity_rms_profile"]}


@router.post("/pipeline/snapshot/{session_name}")
def snapshot_session(session_name: str):
    """
    Save a timestamped snapshot of the current NN profile + session config.
    Writes to snapshots/{session_name}-{YYYYMMDD-HHMM}/
      params-nn-profile.json   copy of source_params
      config.json              copy of session config
    """
    import shutil
    from datetime import datetime
    from gui.routers.sessions import session_dir, load_config

    cfg = load_config(session_name)
    source = Path(cfg.get("source_params", ""))

    snap_dir = Path("snapshots") / f"{session_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    if source.exists():
        dest_params = snap_dir / source.name
        shutil.copy2(source, dest_params)
        saved["params"] = str(dest_params).replace("\\", "/")

    cfg_src = session_dir(session_name) / "config.json"
    if cfg_src.exists():
        dest_cfg = snap_dir / "config.json"
        shutil.copy2(cfg_src, dest_cfg)
        saved["config"] = str(dest_cfg).replace("\\", "/")

    log.info(f"Snapshot saved: {snap_dir}")
    return {"snapshot_dir": str(snap_dir).replace("\\", "/"), "saved": saved}
