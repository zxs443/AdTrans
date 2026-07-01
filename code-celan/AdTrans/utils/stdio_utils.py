"""Make stdout/stderr line-buffered when redirected to cluster log files."""

import os
import sys


def configure_realtime_stdio() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(line_buffering=True)
            continue
        except (AttributeError, OSError, ValueError, TypeError):
            pass
        try:
            fd = stream.fileno()
            wrapped = os.fdopen(fd, "w", buffering=1, closefd=False)
            setattr(sys, name, wrapped)
        except (OSError, ValueError, AttributeError):
            pass
