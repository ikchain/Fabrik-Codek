"""Fabrik Logger - DEPRECATED.

The manual logger (utils/logger.py) has been superseded by automatic capture
via the PostToolUse hook (capture-edits.sh + extract_reasoning.py).

This module is kept for backwards compatibility with external projects that
may still import from src.flywheel.logger. No new code should use it.
"""

import warnings
import sys
from pathlib import Path

# Add project root to path to import from utils/
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.logger import (
    DATALAKE_PATH,
    QUALITY_THRESHOLDS,
    FabrikLogger,
    QualityValidationError,
    get_logger as _get_logger,
    reset_logger,
)


def get_logger(*args, **kwargs):
    """Get logger instance. DEPRECATED: use automatic capture instead."""
    warnings.warn(
        "FabrikLogger is deprecated. Data capture is now automatic via hooks.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_logger(*args, **kwargs)


__all__ = [
    "DATALAKE_PATH",
    "QUALITY_THRESHOLDS",
    "FabrikLogger",
    "QualityValidationError",
    "get_logger",
    "reset_logger",
]
