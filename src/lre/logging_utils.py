"""Logging with personal names masked (report 8, E.10: names are masked in application logs)."""
from __future__ import annotations

import logging
import re

# Any run of Devanagari letters/marks. Names on the register are written in Devanagari,
# so masking these runs keeps owner and parent names out of log files.
_DEVANAGARI_RUN = re.compile(r"[ऀ-ॣ॰-ॿ]+")


def mask(text: str) -> str:
    return _DEVANAGARI_RUN.sub("***", text)


class MaskNamesFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask(str(record.msg))
        if record.args:
            record.args = tuple(mask(a) if isinstance(a, str) else a for a in record.args)
        return True


def get_logger(name: str = "lre", level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        handler.addFilter(MaskNamesFilter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
