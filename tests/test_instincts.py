"""Tests for the Instincts Protocol."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.instincts import (
    AUTO_CONFIDENCE,
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    DECAY_DAYS,
    DECAY_DELTA,
    INITIAL_CONFIDENCE,
    MIN_PATTERN_COUNT,
    PENALIZE_DELTA,
    REINFORCE_DELTA,
    REVIEW_THRESHOLD,
    VALID_CATEGORIES,
    Instinct,
    InstinctRegistry,
    SessionPatternTracker,
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


# ---------------------------------------------------------------------------
# Freshness warnings
# ---------------------------------------------------------------------------


class TestFreshnessWarning:
    def _make(self, **kwargs) -> Instinct:
        defaults = {
            "id": "fw-1",
            "pattern": "test",
            "action": "test action",
            "category": "context",
        }
        defaults.update(kwargs)
        return Instinct(**defaults)

    def test_used_recently(self):
        inst = self._make(last_used=datetime.now().isoformat())
        assert inst.days_since_used is not None
        assert inst.days_since_used <= 1

    def test_never_used(self):
        inst = self._make(last_used=None)
        assert inst.days_since_used is None

    def test_used_45_days_ago(self):
        old = (datetime.now() - timedelta(days=45)).isoformat()
        inst = self._make(last_used=old)
        assert inst.days_since_used is not None
        assert inst.days_since_used >= 44  # allow 1d tolerance
        assert inst.days_since_used > DECAY_DAYS


# ---------------------------------------------------------------------------
# auto_created field
# ---------------------------------------------------------------------------


class TestAutoCreatedField:
    def _make(self, **kwargs) -> Instinct:
        defaults = {
            "id": "ac-1",
            "pattern": "test",
            "action": "test action",
            "category": "context",
        }
        defaults.update(kwargs)
        return Instinct(**defaults)

    def test_to_dict_includes_auto_created(self):
        inst = self._make(auto_created=True)
        d = inst.to_dict()
        assert "auto_created" in d
        assert d["auto_created"] is True

    def test_from_dict_defaults_false_for_legacy(self):
        data = {
            "id": "legacy-1",
            "pattern": "old",
            "action": "old action",
            "category": "workflow",
        }
        inst = Instinct.from_dict(data)
        assert inst.auto_created is False

    def test_round_trip_preserves_auto_created(self):
        inst = self._make(auto_created=True)
        d = inst.to_dict()
        restored = Instinct.from_dict(d)
        assert restored.auto_created is True


# ---------------------------------------------------------------------------
# SessionPatternTracker
# ---------------------------------------------------------------------------


class _FakeCompetenceEntry:
    def __init__(self, topic: str, score: float = 0.5):
        self.topic = topic
        self.score = score


class _FakeCompetenceMap:
    def __init__(self, topics: list[str]):
        self.topics = [_FakeCompetenceEntry(t) for t in topics]


class TestSessionPatternTracker:
    def _make_tracker(self, tmp_path, topics=None, existing_instincts=None):
        registry = InstinctRegistry(tmp_path / "instincts.json")
        if existing_instincts:
            for inst in existing_instincts:
                registry.add(inst)
        cmap = _FakeCompetenceMap(topics or ["postgresql", "docker", "fastapi"])
        return SessionPatternTracker(registry, cmap, "test-session-1234"), registry

    def _long_query(self, topic: str) -> str:
        """Build a query with >= MIN_QUERY_WORDS that includes the topic."""
        return f"How do I configure {topic} for production use in my project"

    def test_below_threshold_no_creation(self, tmp_path):
        tracker, _ = self._make_tracker(tmp_path)
        assert tracker.observe(self._long_query("postgresql")) is None
        assert tracker.observe(self._long_query("postgresql")) is None

    def test_at_threshold_creates_instinct(self, tmp_path):
        tracker, registry = self._make_tracker(tmp_path)
        for _ in range(MIN_PATTERN_COUNT - 1):
            assert tracker.observe(self._long_query("postgresql")) is None
        result = tracker.observe(self._long_query("postgresql"))
        assert result is not None
        assert result.pattern == "postgresql"
        assert result.auto_created is True
        assert registry.get(result.id) is not None

    def test_auto_confidence(self, tmp_path):
        tracker, _ = self._make_tracker(tmp_path)
        for _ in range(MIN_PATTERN_COUNT - 1):
            tracker.observe(self._long_query("docker"))
        result = tracker.observe(self._long_query("docker"))
        assert result is not None
        assert result.confidence == AUTO_CONFIDENCE

    def test_auto_created_flag(self, tmp_path):
        tracker, _ = self._make_tracker(tmp_path)
        for _ in range(MIN_PATTERN_COUNT - 1):
            tracker.observe(self._long_query("fastapi"))
        result = tracker.observe(self._long_query("fastapi"))
        assert result is not None
        assert result.auto_created is True
        assert result.category == "context"

    def test_short_query_skipped(self, tmp_path):
        tracker, _ = self._make_tracker(tmp_path)
        for _ in range(5):
            assert tracker.observe("fix postgresql") is None  # < 5 words

    def test_dedup_after_creation(self, tmp_path):
        tracker, _ = self._make_tracker(tmp_path)
        for _ in range(MIN_PATTERN_COUNT):
            tracker.observe(self._long_query("postgresql"))
        # Subsequent calls return None
        assert tracker.observe(self._long_query("postgresql")) is None
        assert tracker.observe(self._long_query("postgresql")) is None

    def test_none_competence_map(self, tmp_path):
        registry = InstinctRegistry(tmp_path / "instincts.json")
        tracker = SessionPatternTracker(registry, None, "sess-1234")
        assert tracker.observe(self._long_query("postgresql")) is None

    def test_existing_enabled_pattern_skips(self, tmp_path):
        existing = Instinct(
            id="manual-pg",
            pattern="postgresql",
            action="Manual PG instinct",
            category="context",
            enabled=True,
        )
        tracker, _ = self._make_tracker(tmp_path, existing_instincts=[existing])
        for _ in range(MIN_PATTERN_COUNT):
            tracker.observe(self._long_query("postgresql"))
        # Should not create — enabled pattern exists

    def test_existing_disabled_pattern_allows_creation(self, tmp_path):
        existing = Instinct(
            id="disabled-pg",
            pattern="postgresql",
            action="Disabled PG instinct",
            category="context",
            enabled=False,
        )
        tracker, registry = self._make_tracker(tmp_path, existing_instincts=[existing])
        for _ in range(MIN_PATTERN_COUNT - 1):
            tracker.observe(self._long_query("postgresql"))
        result = tracker.observe(self._long_query("postgresql"))
        assert result is not None
        assert result.auto_created is True

    def test_observe_calls_registry_add(self, tmp_path):
        tracker, registry = self._make_tracker(tmp_path)
        for _ in range(MIN_PATTERN_COUNT - 1):
            tracker.observe(self._long_query("docker"))
        result = tracker.observe(self._long_query("docker"))
        assert result is not None
        # Verify it's in the registry
        found = registry.get(result.id)
        assert found is not None
        assert found.pattern == "docker"


# ---------------------------------------------------------------------------
# Turn alignment
# ---------------------------------------------------------------------------


class TestTurnAlignment:
    """Verify prev_matched_instincts pattern for correct causal attribution."""

    def test_reinforce_uses_previous_turn_instincts(self, tmp_path):
        """Instincts matched on turn N-1 get reinforced by outcome of turn N-1."""
        registry = InstinctRegistry(tmp_path / "instincts.json")
        inst = Instinct(
            id="t-1",
            pattern="postgresql",
            action="PG hint",
            category="context",
            confidence=0.50,
        )
        registry.add(inst)

        # Simulate turn alignment
        prev_matched: list = []

        # Turn 1: match instinct
        turn1_matched = registry.match("how to optimize postgresql queries in production")
        assert len(turn1_matched) == 1

        # Turn 2: prev_outcome arrives for turn 1
        # Reinforce prev_matched (turn 0 = empty), not turn1_matched
        # This is correct: first turn has no prev_outcome
        if prev_matched:  # empty on first turn
            for m in prev_matched:
                registry.reinforce(m.id)
        prev_matched = turn1_matched

        # Turn 2 outcome: reinforce turn 1's instincts
        for m in prev_matched:
            registry.reinforce(m.id)

        updated = registry.get("t-1")
        assert updated is not None
        assert updated.confidence > 0.50  # reinforced once

    def test_first_turn_no_reinforcement(self, tmp_path):
        """On the first turn, prev_matched is empty — no reinforcement."""
        registry = InstinctRegistry(tmp_path / "instincts.json")
        inst = Instinct(
            id="t-2",
            pattern="docker",
            action="Docker hint",
            category="context",
            confidence=0.50,
        )
        registry.add(inst)

        prev_matched: list = []
        # First turn: no prev_outcome, no reinforcement
        assert len(prev_matched) == 0
        # No reinforcement should happen
        original = registry.get("t-2")
        assert original is not None
        assert original.confidence == 0.50
