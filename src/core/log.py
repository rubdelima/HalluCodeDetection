from __future__ import annotations

import faulthandler
import logging
import sys
import traceback
from pathlib import Path

_LOGGER_NAME = "hallucode"
_fault_file = None


def setup_logging(log_file: Path | str) -> None:
    global _fault_file
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return

    target = Path(log_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    try:
        _fault_file = open(target, "a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:  # noqa: BLE001
        pass

    _install_excepthook()


def _install_excepthook() -> None:
    logger = logging.getLogger(_LOGGER_NAME)

    def _hook(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.error(
            "Uncaught exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _hook


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
