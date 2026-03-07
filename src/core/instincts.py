"""Instincts Protocol — emergent behavioral patterns from repeated interactions.

Inspired by Savia's Instincts Protocol: patterns learned from user interactions
with variable confidence that decays without reinforcement.

Unlike rules (explicit and fixed), instincts are emergent and have a confidence
level that changes based on usage feedback.

Categories:
  - workflow: sequences of queries the user repeats
  - preference: format/detail preferences
  - shortcut: natural aliases the user uses
  - context: topic->strategy associations
  - timing: temporal patterns (e.g., "morning = debugging")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Confidence parameters
INITIAL_CONFIDENCE = 0.50
REINFORCE_DELTA = 0.03
PENALIZE_DELTA = 0.05
DECAY_DELTA = 0.05
CONFIDENCE_CEILING = 0.95
CONFIDENCE_FLOOR = 0.20
REVIEW_THRESHOLD = 0.30
MIN_PATTERN_COUNT = 3  # Minimum repetitions before creating an instinct
DECAY_DAYS = 30  # Days without use before decay kicks in

VALID_CATEGORIES = frozenset({"workflow", "preference", "shortcut", "context", "timing"})


@dataclass
class Instinct:
    """A single learned behavioral pattern."""

    id: str
    pattern: str
    action: str
    category: str
    confidence: float = INITIAL_CONFIDENCE
    activations: int = 0
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str | None = None
    enabled: bool = True

    def reinforce(self) -> None:
        """Increase confidence after successful use."""
        self.activations += 1
        self.confidence = min(self.confidence + REINFORCE_DELTA, CONFIDENCE_CEILING)
        self.last_used = datetime.now().isoformat()

    def penalize(self) -> None:
        """Decrease confidence after failure or negative feedback."""
        self.confidence = max(self.confidence - PENALIZE_DELTA, CONFIDENCE_FLOOR)
        self.last_used = datetime.now().isoformat()

    def apply_decay(self, now: datetime | None = None) -> bool:
        """Apply time-based decay if unused for DECAY_DAYS.

        Returns True if decay was applied.
        """
        if not self.last_used:
            return False

        if now is None:
            now = datetime.now()

        try:
            last = datetime.fromisoformat(self.last_used)
        except (ValueError, TypeError):
            return False

        days_idle = (now - last).total_seconds() / 86400.0
        if days_idle >= DECAY_DAYS:
            periods = math.floor(days_idle / DECAY_DAYS)
            for _ in range(periods):
                self.confidence = max(self.confidence - DECAY_DELTA, CONFIDENCE_FLOOR)
            return True
        return False

    @property
    def needs_review(self) -> bool:
        """Whether this instinct should be flagged for manual review."""
        return self.confidence < REVIEW_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "action": self.action,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "activations": self.activations,
            "created": self.created,
            "last_used": self.last_used,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Instinct:
        return cls(
            id=data["id"],
            pattern=data["pattern"],
            action=data["action"],
            category=data.get("category", "workflow"),
            confidence=data.get("confidence", INITIAL_CONFIDENCE),
            activations=data.get("activations", 0),
            created=data.get("created", datetime.now().isoformat()),
            last_used=data.get("last_used"),
            enabled=data.get("enabled", True),
        )


class InstinctRegistry:
    """Manages the lifecycle of learned instincts.

    Storage: ``data/profile/instincts.json``
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._instincts: dict[str, Instinct] = {}
        self._load()

    @property
    def instincts(self) -> list[Instinct]:
        return list(self._instincts.values())

    def get(self, instinct_id: str) -> Instinct | None:
        return self._instincts.get(instinct_id)

    def add(self, instinct: Instinct) -> None:
        """Add a new instinct to the registry."""
        if instinct.category not in VALID_CATEGORIES:
            logger.warning("instinct_invalid_category", category=instinct.category)
            return
        self._instincts[instinct.id] = instinct
        self._save()
        logger.info("instinct_added", id=instinct.id, pattern=instinct.pattern)

    def reinforce(self, instinct_id: str) -> bool:
        """Reinforce an instinct after successful use. Returns True if found."""
        inst = self._instincts.get(instinct_id)
        if inst is None:
            return False
        inst.reinforce()
        self._save()
        logger.debug(
            "instinct_reinforced",
            id=instinct_id,
            confidence=inst.confidence,
            activations=inst.activations,
        )
        return True

    def penalize(self, instinct_id: str) -> bool:
        """Penalize an instinct after failure. Returns True if found."""
        inst = self._instincts.get(instinct_id)
        if inst is None:
            return False
        inst.penalize()
        self._save()
        logger.debug("instinct_penalized", id=instinct_id, confidence=inst.confidence)
        return True

    def apply_decay_all(self, now: datetime | None = None) -> int:
        """Apply time-based decay to all instincts. Returns count of decayed."""
        decayed = 0
        for inst in self._instincts.values():
            if inst.apply_decay(now=now):
                decayed += 1
        if decayed > 0:
            self._save()
            logger.info("instincts_decay_applied", decayed=decayed)
        return decayed

    def match(self, query: str) -> list[Instinct]:
        """Find instincts matching a query, sorted by confidence descending.

        Only returns enabled instincts with confidence >= REVIEW_THRESHOLD.
        Matching is case-insensitive substring on the pattern field.
        """
        if not query or not query.strip():
            return []

        query_lower = query.lower()
        matches = []
        for inst in self._instincts.values():
            if not inst.enabled or inst.confidence < REVIEW_THRESHOLD:
                continue
            if inst.pattern.lower() in query_lower:
                matches.append(inst)

        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def get_review_candidates(self) -> list[Instinct]:
        """Return instincts that need manual review (low confidence)."""
        return [i for i in self._instincts.values() if i.needs_review and i.enabled]

    def remove(self, instinct_id: str) -> bool:
        """Remove an instinct. Returns True if found and removed."""
        if instinct_id in self._instincts:
            del self._instincts[instinct_id]
            self._save()
            return True
        return False

    def _load(self) -> None:
        """Load instincts from JSON file."""
        if not self._path.exists():
            self._instincts = {}
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            entries = data.get("entries", [])
            self._instincts = {}
            for entry in entries:
                try:
                    inst = Instinct.from_dict(entry)
                    self._instincts[inst.id] = inst
                except (KeyError, TypeError):
                    continue
            logger.info("instincts_loaded", count=len(self._instincts))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("instincts_load_error", error=str(exc))
            self._instincts = {}

    def _save(self) -> None:
        """Persist instincts to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "entries": [i.to_dict() for i in self._instincts.values()],
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict:
        """Return summary statistics."""
        all_inst = list(self._instincts.values())
        enabled = [i for i in all_inst if i.enabled]
        return {
            "total": len(all_inst),
            "enabled": len(enabled),
            "needs_review": len(self.get_review_candidates()),
            "avg_confidence": (
                round(sum(i.confidence for i in enabled) / len(enabled), 3) if enabled else 0.0
            ),
            "total_activations": sum(i.activations for i in all_inst),
            "by_category": {
                cat: len([i for i in all_inst if i.category == cat]) for cat in VALID_CATEGORIES
            },
        }
