"""Dynamic config with hot-reload for runtime experimentation.

Priority chain (highest wins):
  1. FABRIK_DYNAMIC_<NAME> env var (uppercase)
  2. data/profile/dynamic_config.json (lowercase keys)
  3. default parameter (typically settings.<field>)

Prefix FABRIK_DYNAMIC_ avoids collision with Pydantic BaseSettings
which owns the FABRIK_ namespace.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_PREFIX = "FABRIK_DYNAMIC_"

# -- Lazy path resolution (avoids circular import) ---------------------------

_config_path: Path | None = None
_initialized: bool = False


def _get_config_path() -> Path:
    """Resolve JSON config path lazily from settings.data_dir."""
    global _config_path, _initialized
    if not _initialized:
        try:
            from src.config import settings

            _config_path = Path(settings.data_dir) / "profile" / "dynamic_config.json"
        except Exception:
            _config_path = Path("data/profile/dynamic_config.json")
        _initialized = True
    return _config_path  # type: ignore[return-value]


# -- mtime cache --------------------------------------------------------------

_cached_mtime: float = 0.0
_cached_data: dict[str, Any] = {}


def _read_json() -> dict[str, Any]:
    """Read JSON file with mtime cache. Re-parse only when file changes."""
    global _cached_mtime, _cached_data

    path = _get_config_path()
    if path is None or not path.exists():
        return {}

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    if mtime == _cached_mtime:
        return _cached_data

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("dynamic_config: expected dict, got %s", type(data).__name__)
            return {}
        _cached_mtime = mtime
        _cached_data = data
        return _cached_data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("dynamic_config: failed to read %s: %s", path, exc)
        return {}


# -- Cast logic ---------------------------------------------------------------


def _cast_bool(value: Any) -> bool:
    """Cast a value to bool. Strings use truthy keyword matching."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _cast_value(value: Any, cast: type) -> Any:
    """Apply type cast to a value. Returns None on failure."""
    if cast is bool:
        return _cast_bool(value)
    try:
        return cast(value)
    except (ValueError, TypeError):
        return None


# -- Public API ---------------------------------------------------------------


def get_dynamic(name: str, default: Any = None, cast: type | None = None) -> Any:
    """Read a dynamic config value with hot-reload.

    Resolution:
      1. os.environ["FABRIK_DYNAMIC_<NAME>"] (uppercase)
      2. dynamic_config.json[name] (lowercase)
      3. default

    cast: type to convert values. If None, inferred from type(default).
    If both cast and default are None, returns raw value without casting.
    """
    resolved_cast = cast
    if resolved_cast is None and default is not None:
        resolved_cast = type(default)

    # 1. Env var (highest priority)
    env_key = f"{_ENV_PREFIX}{name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val != "":
        if resolved_cast is not None:
            casted = _cast_value(env_val, resolved_cast)
            if casted is not None:
                return casted
            logger.warning("dynamic_config: cast failed for env %s=%r", env_key, env_val)
            return default
        return env_val

    # 2. JSON file
    data = _read_json()
    json_val = data.get(name)
    if json_val is not None:
        if resolved_cast is not None and not isinstance(json_val, resolved_cast):
            casted = _cast_value(json_val, resolved_cast)
            if casted is not None:
                return casted
            logger.warning("dynamic_config: cast failed for json %s=%r", name, json_val)
            return default
        return json_val

    # 3. Default
    return default


def reset_cache() -> None:
    """Reset the mtime cache. Useful for testing."""
    global _cached_mtime, _cached_data, _config_path, _initialized
    _cached_mtime = 0.0
    _cached_data = {}
    _config_path = None
    _initialized = False
