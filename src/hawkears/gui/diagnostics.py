"""Temporary detailed diagnostics for GUI beta testing."""

import faulthandler
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import sys
from typing import TextIO

_fault_file: TextIO | None = None


def diagnostic_directory() -> Path:
    state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return state_root / "hawkears" / "logs"


def configure_diagnostics() -> Path:
    """Configure detailed rotating logs and native-fault stack output."""
    global _fault_file

    directory = diagnostic_directory()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "hawkears-gui.log"
    handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s "
            "[%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger = logging.getLogger("hawkears")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    crash_path = directory / "hawkears-gui-crash.log"
    _fault_file = crash_path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_fault_file, all_threads=True)

    def log_unhandled_exception(exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, traceback)
        )
        sys.__excepthook__(exc_type, exc_value, traceback)

    sys.excepthook = log_unhandled_exception
    logger.info("GUI diagnostics started; pid=%d", os.getpid())
    logger.info(
        "Python=%s platform=%s", sys.version.replace("\n", " "), platform.platform()
    )
    logger.info("Crash traces: %s", crash_path)
    return log_path
