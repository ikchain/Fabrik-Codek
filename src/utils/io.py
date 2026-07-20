"""Atomic file writes (FC-92).

Write to a temp file in the same directory, fsync, then ``os.replace`` onto the
target. ``os.replace`` is atomic when source and destination share a filesystem,
so a crash or NTFS read-only fault mid-write leaves the original file intact
instead of truncated.

This applies to full-file overwrites (JSON state, full JSONL dumps). It does NOT
apply to append-only logs: there you want a plain append, where a partial
trailing line is self-healing — re-writing the whole file would be wrong.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Leave nothing behind on any failure path, then re-raise.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write ``data`` as pretty UTF-8 JSON (non-ASCII preserved)."""
    atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=False))
