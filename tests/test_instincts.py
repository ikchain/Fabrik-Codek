"""Tests for the Instincts Protocol."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.instincts import (
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    DECAY_DAYS,
    DECAY_DELTA,
    INITIAL_CONFIDENCE,
    PENALIZE_DELTA,
    REINFORCE_DELTA,
    REVIEW_THRESHOLD,
    VALID_CATEGORIES,
    Instinct,
    InstinctRegistry,
)

# ---------------------------------------------------------------------------
# Instinct dataclass
# ---------------------------------------------------------------------------


class TestInstinct:
    """Tests for the Instinct dataclass."""

    def _make(self, **kwargs) -> Instinct:
        defaults = {
            "id": "test-1",
            "pattern": "debug error",
            "action": "enable verbose logging",
            "category": "workflow",
        }
        defaults.update(kwargs)
        return Instinct(**defaults)

    def test_defaults(self):
        inst = self._make()
        assert inst.confidence == INITIAL_CONFIDENCE
        assert inst.activations == 0
        assert inst.last_used is None
        assert inst.enabled is True
        assert inst.created  # non-empty

    def test_reinforce_increases_confidence(self):
        inst = self._make()
        old = inst.confidence
        inst.reinforce()
        assert inst.confidence == old + REINFORCE_DELTA
        assert inst.activations == 1
        assert inst.last_used is not None

    def test_reinforce_respects_ceiling(self):
        inst = self._make(confidence=CONFIDENCE_CEILING)
        inst.reinforce()
        assert inst.confidence == CONFIDENCE_CEILING

    def test_penalize_decreases_confidence(self):
        inst = self._make()
        old = inst.confidence
        inst.penalize()
        assert inst.confidence == old - PENALIZE_DELTA
        assert inst.last_used is not None

    def test_penalize_respects_floor(self):
        inst = self._make(confidence=CONFIDENCE_FLOOR)
        inst.penalize()
        assert inst.confidence == CONFIDENCE_FLOOR

    def test_needs_review_below_threshold(self):
        inst = self._make(confidence=REVIEW_THRESHOLD - 0.01)
        assert inst.needs_review is True

    def test_needs_review_above_threshold(self):
        inst = self._make(confidence=REVIEW_THRESHOLD + 0.01)
        assert inst.needs_review is False

    def test_apply_decay_no_last_used(self):
        inst = self._make()
        assert inst.apply_decay() is False

    def test_apply_decay_recent(self):
        inst = self._make(last_used=datetime.now().isoformat())
        assert inst.apply_decay() is False

    def test_apply_decay_one_period(self):
        inst = self._make(confidence=0.80)
        past = datetime.now() - timedelta(days=DECAY_DAYS + 1)
        inst.last_used = past.isoformat()
        result = inst.apply_decay()
        assert result is True
        assert inst.confidence == pytest.approx(0.80 - DECAY_DELTA)

    def test_apply_decay_multiple_periods(self):
        inst = self._make(confidence=0.80)
        past = datetime.now() - timedelta(days=DECAY_DAYS * 3 + 1)
        inst.last_used = past.isoformat()
        result = inst.apply_decay()
        assert result is True
        expected = 0.80 - (DECAY_DELTA * 3)
        assert inst.confidence == pytest.approx(expected)

    def test_apply_decay_respects_floor(self):
        inst = self._make(confidence=CONFIDENCE_FLOOR + 0.01)
        past = datetime.now() - timedelta(days=DECAY_DAYS * 100)
        inst.last_used = past.isoformat()
        inst.apply_decay()
        assert inst.confidence == CONFIDENCE_FLOOR

    def test_apply_decay_invalid_last_used(self):
        inst = self._make()
        inst.last_used = "not-a-date"
        assert inst.apply_decay() is False

    def test_to_dict(self):
        inst = self._make()
        d = inst.to_dict()
        assert d["id"] == "test-1"
        assert d["pattern"] == "debug error"
        assert d["action"] == "enable verbose logging"
        assert d["category"] == "workflow"
        assert d["confidence"] == INITIAL_CONFIDENCE
        assert d["activations"] == 0
        assert d["enabled"] is True

    def test_from_dict(self):
        data = {
            "id": "x",
            "pattern": "p",
            "action": "a",
            "category": "shortcut",
            "confidence": 0.75,
            "activations": 5,
            "last_used": "2026-01-01T00:00:00",
            "enabled": False,
        }
        inst = Instinct.from_dict(data)
        assert inst.id == "x"
        assert inst.category == "shortcut"
        assert inst.confidence == 0.75
        assert inst.activations == 5
        assert inst.enabled is False

    def test_from_dict_defaults(self):
        data = {"id": "y", "pattern": "p", "action": "a"}
        inst = Instinct.from_dict(data)
        assert inst.category == "workflow"
        assert inst.confidence == INITIAL_CONFIDENCE
        assert inst.enabled is True

    def test_roundtrip(self):
        inst = self._make()
        inst.reinforce()
        d = inst.to_dict()
        inst2 = Instinct.from_dict(d)
        assert inst2.id == inst.id
        assert inst2.confidence == inst.confidence
        assert inst2.activations == inst.activations


# ---------------------------------------------------------------------------
# InstinctRegistry
# ---------------------------------------------------------------------------


class TestInstinctRegistry:
    """Tests for the InstinctRegistry."""

    @pytest.fixture
    def reg_path(self, tmp_path: Path) -> Path:
        return tmp_path / "instincts.json"

    @pytest.fixture
    def registry(self, reg_path: Path) -> InstinctRegistry:
        return InstinctRegistry(reg_path)

    def _inst(self, id: str = "i1", **kwargs) -> Instinct:
        defaults = {
            "id": id,
            "pattern": "test pattern",
            "action": "test action",
            "category": "workflow",
        }
        defaults.update(kwargs)
        return Instinct(**defaults)

    def test_empty_registry(self, registry: InstinctRegistry):
        assert registry.instincts == []
        stats = registry.stats()
        assert stats["total"] == 0

    def test_add_and_get(self, registry: InstinctRegistry):
        inst = self._inst()
        registry.add(inst)
        assert registry.get("i1") is inst
        assert len(registry.instincts) == 1

    def test_add_invalid_category(self, registry: InstinctRegistry):
        inst = self._inst(category="invalid")
        registry.add(inst)
        assert registry.get("i1") is None

    def test_add_persists(self, reg_path: Path):
        reg = InstinctRegistry(reg_path)
        reg.add(self._inst())
        # Reload from same file
        reg2 = InstinctRegistry(reg_path)
        assert reg2.get("i1") is not None
        assert reg2.get("i1").pattern == "test pattern"

    def test_reinforce(self, registry: InstinctRegistry):
        registry.add(self._inst())
        assert registry.reinforce("i1") is True
        assert registry.get("i1").activations == 1
        assert registry.get("i1").confidence == INITIAL_CONFIDENCE + REINFORCE_DELTA

    def test_reinforce_missing(self, registry: InstinctRegistry):
        assert registry.reinforce("nope") is False

    def test_penalize(self, registry: InstinctRegistry):
        registry.add(self._inst())
        assert registry.penalize("i1") is True
        assert registry.get("i1").confidence == INITIAL_CONFIDENCE - PENALIZE_DELTA

    def test_penalize_missing(self, registry: InstinctRegistry):
        assert registry.penalize("nope") is False

    def test_remove(self, registry: InstinctRegistry):
        registry.add(self._inst())
        assert registry.remove("i1") is True
        assert registry.get("i1") is None

    def test_remove_missing(self, registry: InstinctRegistry):
        assert registry.remove("nope") is False

    def test_match_basic(self, registry: InstinctRegistry):
        registry.add(self._inst(id="a", pattern="debug", confidence=0.8))
        registry.add(self._inst(id="b", pattern="test", confidence=0.6))
        matches = registry.match("how to debug this error")
        assert len(matches) == 1
        assert matches[0].id == "a"

    def test_match_empty_query(self, registry: InstinctRegistry):
        registry.add(self._inst())
        assert registry.match("") == []
        assert registry.match("   ") == []

    def test_match_sorted_by_confidence(self, registry: InstinctRegistry):
        registry.add(self._inst(id="a", pattern="code", confidence=0.5))
        registry.add(self._inst(id="b", pattern="code", confidence=0.9))
        matches = registry.match("review this code please")
        assert len(matches) == 2
        assert matches[0].id == "b"
        assert matches[1].id == "a"

    def test_match_excludes_disabled(self, registry: InstinctRegistry):
        registry.add(self._inst(id="a", pattern="code", enabled=False))
        assert registry.match("review code") == []

    def test_match_excludes_low_confidence(self, registry: InstinctRegistry):
        registry.add(self._inst(id="a", pattern="code", confidence=REVIEW_THRESHOLD - 0.01))
        assert registry.match("review code") == []

    def test_apply_decay_all(self, registry: InstinctRegistry):
        now = datetime.now()
        past = (now - timedelta(days=DECAY_DAYS + 1)).isoformat()
        registry.add(self._inst(id="a", confidence=0.8))
        registry.add(self._inst(id="b", confidence=0.7))
        registry.get("a").last_used = past
        registry.get("b").last_used = now.isoformat()
        decayed = registry.apply_decay_all(now=now)
        assert decayed == 1
        assert registry.get("a").confidence < 0.8
        assert registry.get("b").confidence == 0.7

    def test_get_review_candidates(self, registry: InstinctRegistry):
        registry.add(self._inst(id="a", confidence=REVIEW_THRESHOLD - 0.01))
        registry.add(self._inst(id="b", confidence=0.8))
        registry.add(self._inst(id="c", confidence=REVIEW_THRESHOLD - 0.05, enabled=False))
        candidates = registry.get_review_candidates()
        assert len(candidates) == 1
        assert candidates[0].id == "a"

    def test_stats(self, registry: InstinctRegistry):
        registry.add(self._inst(id="a", category="workflow", confidence=0.8))
        registry.add(self._inst(id="b", category="preference", confidence=0.6))
        registry.add(self._inst(id="c", category="workflow", confidence=0.4, enabled=False))
        stats = registry.stats()
        assert stats["total"] == 3
        assert stats["enabled"] == 2
        assert stats["avg_confidence"] == pytest.approx((0.8 + 0.6) / 2, abs=0.01)
        assert stats["by_category"]["workflow"] == 2
        assert stats["by_category"]["preference"] == 1

    def test_load_corrupt_file(self, reg_path: Path):
        reg_path.write_text("not json", encoding="utf-8")
        reg = InstinctRegistry(reg_path)
        assert reg.instincts == []

    def test_load_empty_entries(self, reg_path: Path):
        reg_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
        reg = InstinctRegistry(reg_path)
        assert reg.instincts == []

    def test_load_skip_invalid_entries(self, reg_path: Path):
        data = {
            "entries": [
                {"id": "ok", "pattern": "p", "action": "a"},
                {"bad": "entry"},  # missing required fields
            ]
        }
        reg_path.write_text(json.dumps(data), encoding="utf-8")
        reg = InstinctRegistry(reg_path)
        assert len(reg.instincts) == 1

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "instincts.json"
        reg = InstinctRegistry(path)
        reg.add(self._inst())
        assert path.exists()

    def test_all_categories_valid(self):
        expected = {"workflow", "preference", "shortcut", "context", "timing"}
        assert VALID_CATEGORIES == expected


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constant values match design spec."""

    def test_initial_confidence(self):
        assert INITIAL_CONFIDENCE == 0.50

    def test_reinforce_delta(self):
        assert REINFORCE_DELTA == 0.03

    def test_penalize_delta(self):
        assert PENALIZE_DELTA == 0.05

    def test_decay_delta(self):
        assert DECAY_DELTA == 0.05

    def test_ceiling(self):
        assert CONFIDENCE_CEILING == 0.95

    def test_floor(self):
        assert CONFIDENCE_FLOOR == 0.20

    def test_review_threshold(self):
        assert REVIEW_THRESHOLD == 0.30

    def test_decay_days(self):
        assert DECAY_DAYS == 30
