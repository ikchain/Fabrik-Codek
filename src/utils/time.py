"""UTC datetime helpers (FC-90).

Standardize on timezone-aware UTC. ``ensure_aware`` normalizes legacy naive
timestamps (assumed UTC) so datetime arithmetic never raises the
naive-vs-aware ``TypeError``.
"""

from __future__ import annotations

from datetime import UTC, datetime


def now_utc() -> datetime:
    """Current time, timezone-aware in UTC."""
    return datetime.now(UTC)


def ensure_aware(dt: datetime) -> datetime:
    """Return ``dt`` as UTC-aware; a naive input is assumed to be UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
