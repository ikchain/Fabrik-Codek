"""Context-Map determinista — bypasses pipeline for predictable queries.

Mapa estatico de patrones de query a decisiones de routing pre-computadas.
Inspirado en el context-map de Savia (pm-workspace): "no necesito buscar
porque se donde esta todo".

Consultar ANTES del Router. Si hay hit, skip RAG/graph/LLM classification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()

# All valid profile fragment names
VALID_FRAGMENTS: frozenset[str] = frozenset(
    {"identity", "tech_stack", "patterns", "projects", "decisions", "style"}
)


@dataclass
class ContextMapEntry:
    """A single mapping from query patterns to routing shortcut."""

    patterns: list[str]
    task_type: str = "general"
    use_context: bool = False
    profile_fragments: list[str] = field(default_factory=lambda: ["identity"])
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._compiled = []
        for p in self.patterns:
            try:
                self._compiled.append(re.compile(p, re.IGNORECASE))
            except re.error:
                logger.warning("context_map_invalid_pattern", pattern=p)

    def matches(self, query: str) -> bool:
        """Return True if any pattern matches the query."""
        return any(rx.search(query) for rx in self._compiled)


@dataclass
class ContextMapResult:
    """Result of a context-map lookup."""

    entry: ContextMapEntry
    matched_pattern: str


class ContextMap:
    """Deterministic query->routing map that bypasses the full pipeline.

    Loaded from a JSON file at ``data/profile/context_map.json``.
    Consulted before TaskRouter -- if hit, skips classification, RAG, and graph.
    """

    def __init__(self, entries: list[ContextMapEntry] | None = None) -> None:
        self._entries: list[ContextMapEntry] = entries or []

    @property
    def entries(self) -> list[ContextMapEntry]:
        return self._entries

    def match(self, query: str) -> ContextMapResult | None:
        """Find the first matching entry for a query.

        Returns ContextMapResult on hit, None on miss.
        """
        if not query or not query.strip():
            return None

        for entry in self._entries:
            for i, rx in enumerate(entry._compiled):
                if rx.search(query):
                    pattern_str = entry.patterns[i] if i < len(entry.patterns) else ""
                    logger.info(
                        "context_map_hit",
                        task_type=entry.task_type,
                        use_context=entry.use_context,
                        pattern=pattern_str,
                    )
                    return ContextMapResult(entry=entry, matched_pattern=pattern_str)

        logger.debug("context_map_miss")
        return None

    @classmethod
    def from_file(cls, path: Path) -> ContextMap:
        """Load context map from a JSON file.

        Returns an empty ContextMap if the file doesn't exist or is invalid.
        """
        if not path.exists():
            logger.debug("context_map_file_not_found", path=str(path))
            return cls()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("context_map_load_error", path=str(path), error=str(exc))
            return cls()

        raw_entries = data.get("entries", [])
        entries: list[ContextMapEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, dict) or "patterns" not in raw:
                continue
            fragments = raw.get("profile_fragments", ["identity"])
            # Validate fragments
            valid = [f for f in fragments if f in VALID_FRAGMENTS]
            entries.append(
                ContextMapEntry(
                    patterns=raw["patterns"],
                    task_type=raw.get("task_type", "general"),
                    use_context=raw.get("use_context", False),
                    profile_fragments=valid or ["identity"],
                )
            )

        logger.info("context_map_loaded", path=str(path), entries=len(entries))
        return cls(entries)
