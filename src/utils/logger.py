import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


class StructuredLogger:
    """JSON-structured logger that writes to stdout.

    Every log line is a JSON object with timestamp, module, level, and message.
    Extra key-value pairs can be passed as kwargs.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self._name = name
        self._level = getattr(logging, level.upper(), logging.INFO)

    def _log(self, level: str, message: str, **kwargs):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": self._name,
            "level": level,
            "message": message,
        }
        record.update(kwargs)
        print(json.dumps(record), file=sys.stdout)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, exc: Optional[Exception] = None, **kwargs):
        payload = {"exception": f"{type(exc).__name__}: {exc}"} if exc else {}
        payload.update(kwargs)
        self._log("ERROR", message, **payload)
