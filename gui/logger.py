"""
gui/logger.py
─────────────
Centralized logging for the GUI backend.
Logs to gui/logs/server.log (plain append) and to stdout.
RotatingFileHandler is intentionally avoided: on Windows the reloader
subprocess and the worker process both hold the file open, making os.rename
(required by rotation) fail with WinError 32.
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path("gui/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.DEBUG)

    # Plain file handler — no rotation (avoids WinError 32 on Windows)
    fh = logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8")
    fh.setFormatter(_fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


# Root GUI logger — import and use in all routers
log = get_logger("gui")
