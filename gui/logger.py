"""
gui/logger.py
─────────────
Centralized logging for the GUI backend.
Logs to gui/logs/server.log (rotating, max 5 MB × 3 files) and to stdout.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
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

    # File handler — rotating
    fh = RotatingFileHandler(
        LOG_DIR / "server.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
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
