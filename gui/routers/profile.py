"""
gui/routers/profile.py
───────────────────────
DDSP profile training endpoint.

POST /api/profile/train   — start training job in background thread
GET  /api/profile/status  — poll epoch / loss / status
POST /api/profile/cancel  — request early stop
"""

import json
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.logger import get_logger

log = get_logger("gui.profile")
router = APIRouter()

# Single global training job (only one at a time)
_job: dict = {"status": "idle"}


class TrainRequest(BaseModel):
    wav_dir:    str  = "C:/SoundBanks/IthacaPlayer/ks-grand"
    out:        str  = "analysis/params_profile.json"
    model_out:  str  = "analysis/profile_ddsp.pt"
    init:       str  = ""           # optional warm-start path
    epochs:     int  = 300
    seg:        float = 0.5
    kmax:       int  = 16
    batch:      int  = 8
    lr:         float = 1e-3
    sr:         int  = 44100


def _run_train(req: TrainRequest):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from analysis.train_ddsp import (
        load_wav_bank, train_ddsp, generate_profile, build_feature_tables,
        InstrumentProfile,
    )
    import torch

    job = _job
    job["status"]      = "loading"
    job["epoch"]       = 0
    job["total"]       = req.epochs
    job["loss"]        = None
    job["error"]       = None
    job["cancel"]      = False
    job["started_at"]  = time.time()
    job["out"]         = req.out

    try:
        seg_len = int(req.seg * req.sr)
        wav_bank = load_wav_bank(req.wav_dir, sr_target=req.sr, seg_secs=req.seg)
        if not wav_bank:
            raise RuntimeError(f"No WAV files found in {req.wav_dir}")

        log.info(f"DDSP train: {len(wav_bank)} notes, {req.epochs} epochs")

        model = InstrumentProfile(hidden=64)
        if req.init and Path(req.init).exists():
            ckpt = torch.load(req.init, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            log.info(f"Warm-started from {req.init}")

        job["status"] = "running"

        train_ddsp(
            model, wav_bank,
            epochs=req.epochs, lr=req.lr,
            sr=req.sr, seg_len=seg_len,
            k_max=req.kmax, batch_size=req.batch,
            verbose=False,
            progress=job,
        )

        if job.get("cancel"):
            job["status"] = "cancelled"
            log.info("DDSP train: cancelled by user")
            return

        # Save model
        torch.save({"state_dict": model.state_dict(), "hidden": 64, "eq_freqs": None},
                   req.model_out)
        log.info(f"Model saved -> {req.model_out}")

        # Generate profile
        job["status"] = "saving"
        orig_samples = None
        params_path = Path("analysis/params.json")
        if params_path.exists():
            orig_samples = json.loads(params_path.read_text()).get("samples")

        ds = {"eq_freqs": None, "batches": {}}
        profile_samples = generate_profile(model, ds, orig_samples=orig_samples)
        n_nn   = sum(1 for s in profile_samples.values() if s.get("_from_profile"))
        n_orig = sum(1 for s in profile_samples.values() if not s.get("_from_profile"))

        Path(req.out).write_text(
            json.dumps({"samples": profile_samples}, indent=2, ensure_ascii=False)
        )
        log.info(f"Profile written -> {req.out}  (NN:{n_nn} orig:{n_orig})")

        job["status"]   = "done"
        job["n_nn"]     = n_nn
        job["n_orig"]   = n_orig
        job["finished_at"] = time.time()

    except Exception as e:
        log.error(f"DDSP train error: {e}")
        job["status"] = "error"
        job["error"]  = str(e)


@router.post("/profile/train")
def start_train(body: TrainRequest):
    if _job.get("status") in ("running", "loading", "saving"):
        raise HTTPException(409, "Training already running")
    _job.clear()
    _job.update({"status": "starting", "epoch": 0, "total": body.epochs,
                 "loss": None, "cancel": False})
    t = threading.Thread(target=_run_train, args=(body,), daemon=True)
    t.start()
    return {"started": True, "epochs": body.epochs}


@router.get("/profile/status")
def get_train_status():
    j = dict(_job)
    ep = j.get("epoch", 0)
    tot = j.get("total", 1)
    j["progress_pct"] = round(100 * ep / tot) if tot > 0 else 0
    # ETA estimate
    if j.get("status") == "running" and ep > 5 and "started_at" in j:
        elapsed = time.time() - j["started_at"]
        eta_s = int(elapsed / ep * (tot - ep))
        j["eta_s"] = eta_s
    return j


@router.post("/profile/cancel")
def cancel_train():
    if _job.get("status") not in ("running", "loading", "saving"):
        raise HTTPException(400, "No active training job")
    _job["cancel"] = True
    return {"cancelling": True}


@router.post("/profile/apply/{session_name}")
def apply_profile_to_session(session_name: str):
    """
    Point the session's source_params at the trained profile output
    and copy it into the session directory so it survives file moves.
    """
    import shutil
    from gui.routers.sessions import session_dir, load_config, save_config

    out_path = _job.get("out")
    if not out_path or not Path(out_path).exists():
        raise HTTPException(400, f"Trained profile not found: {out_path}")

    sdir = session_dir(session_name)
    if not sdir.exists():
        raise HTTPException(404, f"Session '{session_name}' not found")

    # Copy trained profile into session directory
    dest = sdir / "params.json"
    shutil.copy2(out_path, dest)

    # Update session config to reference the session-local copy
    cfg = load_config(session_name)
    cfg["source_params"] = str(dest)
    save_config(session_name, cfg)

    log.info(f"Applied DDSP profile to session '{session_name}' -> {dest}")
    return {"applied": True, "source_params": str(dest)}
