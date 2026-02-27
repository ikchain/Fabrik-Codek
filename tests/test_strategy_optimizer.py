"""Tests for the MAB Strategy Optimizer (FC-42)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.core.strategy_optimizer import (
    ARM_DEFINITIONS,
    DEFAULT_PRIORS,
    MABStrategyOptimizer,
)
from src.core.task_router import TASK_STRATEGIES, RetrievalStrategy

# ---------------------------------------------------------------------------
# Task 1 tests: Arm definitions & selection
# ---------------------------------------------------------------------------


class TestArmDefinitions:
    def test_four_arms_defined(self) -> None:
        """ARM_DEFINITIONS has exactly 4 arms."""
        assert len(ARM_DEFINITIONS) == 4
        assert set(ARM_DEFINITIONS.keys()) == {
            "default",
            "graph_boost",
            "deep_graph",
            "vector_focus",
        }

    def test_default_arm_no_modifications(self) -> None:
        """The default arm applies zero modifications."""
        arm = ARM_DEFINITIONS["default"]
        assert arm["graph_depth_delta"] == 0
        assert arm["vector_weight_delta"] == 0.0
        assert arm["graph_weight_delta"] == 0.0
        assert arm["fulltext_weight"] == 0.0


class TestSelectArm:
    def test_select_arm_returns_valid_arm_id(self, tmp_path: Path) -> None:
        """select_arm returns one of the 4 defined arm IDs."""
        mab = MABStrategyOptimizer(tmp_path)
        arm_id, strategy = mab.select_arm("debugging", "postgresql")
        assert arm_id in ARM_DEFINITIONS
        assert isinstance(strategy, RetrievalStrategy)

    def test_select_arm_cold_start_uses_priors(self, tmp_path: Path) -> None:
        """On first call for a new context, priors are created automatically."""
        mab = MABStrategyOptimizer(tmp_path)
        # Context doesn't exist yet — should create it with priors
        arm_id, strategy = mab.select_arm("architecture", "fastapi")
        assert arm_id in ARM_DEFINITIONS
        # State should now contain the context
        assert "architecture_fastapi" in mab._state

    def test_select_arm_no_topic(self, tmp_path: Path) -> None:
        """When topic is None, context key is just task_type."""
        mab = MABStrategyOptimizer(tmp_path)
        arm_id, strategy = mab.select_arm("debugging", None)
        assert arm_id in ARM_DEFINITIONS
        assert "debugging" in mab._state

    def test_select_arm_strategy_has_valid_weights(self, tmp_path: Path) -> None:
        """Returned strategy has non-negative weights."""
        mab = MABStrategyOptimizer(tmp_path)
        _, strategy = mab.select_arm("debugging", "postgresql")
        assert strategy.graph_depth >= 1
        assert strategy.vector_weight >= 0.0
        assert strategy.graph_weight >= 0.0
        assert strategy.fulltext_weight >= 0.0

    def test_select_arm_deterministic_with_seed(self, tmp_path: Path) -> None:
        """With same random seed, select_arm returns the same arm."""
        mab = MABStrategyOptimizer(tmp_path)
        random.seed(42)
        arm1, _ = mab.select_arm("debugging", "postgresql")
        # Reset state and seed
        mab2 = MABStrategyOptimizer(tmp_path)
        random.seed(42)
        arm2, _ = mab2.select_arm("debugging", "postgresql")
        assert arm1 == arm2

    def test_select_arm_does_not_increment_total_pulls(self, tmp_path: Path) -> None:
        """select_arm alone does not count as a training observation."""
        mab = MABStrategyOptimizer(tmp_path)
        for _ in range(5):
            mab.select_arm("debugging", "postgresql")
        assert mab._state["debugging_postgresql"]["total_pulls"] == 0


# ---------------------------------------------------------------------------
# Task 2 tests: Update and persistence
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_accepted_increments_alpha(self, tmp_path: Path) -> None:
        """Accepted outcome (reward=1) increments alpha."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")  # ensure context exists
        before = mab._state["debugging_postgresql"]["arms"]["default"]["alpha"]
        mab.update("debugging", "postgresql", "default", reward=1.0)
        after = mab._state["debugging_postgresql"]["arms"]["default"]["alpha"]
        assert after == before + 1

    def test_update_rejected_increments_beta(self, tmp_path: Path) -> None:
        """Rejected outcome (reward=0) increments beta."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        before = mab._state["debugging_postgresql"]["arms"]["default"]["beta"]
        mab.update("debugging", "postgresql", "default", reward=0.0)
        after = mab._state["debugging_postgresql"]["arms"]["default"]["beta"]
        assert after == before + 1

    def test_update_increments_total_pulls(self, tmp_path: Path) -> None:
        """Each update increments total_pulls for the context."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        mab.update("debugging", "postgresql", "default", reward=1.0)
        assert mab._state["debugging_postgresql"]["total_pulls"] == 1

    def test_update_unknown_arm_no_crash(self, tmp_path: Path) -> None:
        """Updating a nonexistent arm logs warning but doesn't crash."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        mab.update("debugging", "postgresql", "nonexistent_arm", reward=1.0)
        # total_pulls should NOT increment for invalid arm
        assert mab._state["debugging_postgresql"]["total_pulls"] == 0

    def test_update_marks_dirty(self, tmp_path: Path) -> None:
        """After update, the state is marked dirty for persistence."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        mab._dirty = False  # reset
        mab.update("debugging", "postgresql", "default", reward=1.0)
        assert mab._dirty is True


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """State survives save + load cycle."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        mab.update("debugging", "postgresql", "default", reward=1.0)
        mab.update("debugging", "postgresql", "graph_boost", reward=0.0)
        mab.save_state()

        # Load into new instance
        mab2 = MABStrategyOptimizer(tmp_path)
        ctx = mab2._state["debugging_postgresql"]
        assert ctx["arms"]["default"]["alpha"] == DEFAULT_PRIORS["default"]["alpha"] + 1
        assert ctx["arms"]["graph_boost"]["beta"] == DEFAULT_PRIORS["graph_boost"]["beta"] + 1
        assert ctx["total_pulls"] == 2

    def test_save_only_when_dirty(self, tmp_path: Path) -> None:
        """save_state is a no-op when state hasn't changed."""
        mab = MABStrategyOptimizer(tmp_path)
        mab._dirty = False
        mab.save_state()
        assert not mab._state_path.exists()

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """When mab_state.json doesn't exist, state starts empty."""
        mab = MABStrategyOptimizer(tmp_path)
        assert mab._state == {}

    def test_load_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        """Corrupt JSON file results in empty state (graceful degradation)."""
        state_path = tmp_path / "profile" / "mab_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("not valid json{{{")
        mab = MABStrategyOptimizer(tmp_path)
        assert mab._state == {}


# ---------------------------------------------------------------------------
# Task 3 tests: Thompson convergence & export
# ---------------------------------------------------------------------------


class TestThompsonConvergence:
    def test_dominant_arm_selected_most_often(self, tmp_path: Path) -> None:
        """After many positive updates on one arm, it should be selected >60% of the time."""
        mab = MABStrategyOptimizer(tmp_path)
        # Heavily train "graph_boost" for this context
        for _ in range(50):
            mab.update("debugging", "postgresql", "graph_boost", reward=1.0)
        for _ in range(5):
            mab.update("debugging", "postgresql", "graph_boost", reward=0.0)
        # Other arms untouched (priors only)

        selections: dict[str, int] = {"graph_boost": 0}
        for _ in range(200):
            arm_id, _ = mab.select_arm("debugging", "postgresql")
            selections[arm_id] = selections.get(arm_id, 0) + 1

        # graph_boost should dominate (Beta(51,6) ~ 0.89 expected value)
        assert selections["graph_boost"] > 120  # >60% of 200

    def test_poorly_performing_arm_avoided(self, tmp_path: Path) -> None:
        """An arm with many failures should be selected rarely."""
        mab = MABStrategyOptimizer(tmp_path)
        # Make deep_graph fail a lot
        for _ in range(30):
            mab.update("debugging", "postgresql", "deep_graph", reward=0.0)

        selections: dict[str, int] = {"deep_graph": 0}
        for _ in range(100):
            arm_id, _ = mab.select_arm("debugging", "postgresql")
            selections[arm_id] = selections.get(arm_id, 0) + 1

        # deep_graph should be very rare (Beta(1,31) ~ 0.03 expected value)
        assert selections.get("deep_graph", 0) < 10


class TestExportOverrides:
    def test_export_empty_state(self, tmp_path: Path) -> None:
        """Empty state exports empty overrides."""
        mab = MABStrategyOptimizer(tmp_path)
        assert mab.export_overrides() == {}

    def test_export_default_arm_no_override(self, tmp_path: Path) -> None:
        """When default arm is best, no override is exported."""
        mab = MABStrategyOptimizer(tmp_path)
        # default starts with Beta(2,1), others Beta(1,1)
        # With no updates, default has highest expected value
        mab.select_arm("debugging", "postgresql")
        overrides = mab.export_overrides()
        assert "debugging_postgresql" not in overrides

    def test_export_non_default_arm_produces_override(self, tmp_path: Path) -> None:
        """When a non-default arm dominates, an override is exported."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        # Make graph_boost clearly better
        for _ in range(20):
            mab.update("debugging", "postgresql", "graph_boost", reward=1.0)

        overrides = mab.export_overrides()
        assert "debugging_postgresql" in overrides
        override = overrides["debugging_postgresql"]
        assert override["arm_id"] == "graph_boost"
        assert "graph_depth" in override
        assert "expected_value" in override

    def test_export_underscore_task_type(self, tmp_path: Path) -> None:
        """export_overrides correctly handles task types with underscores (code_review, ml_engineering)."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("code_review", "postgresql")
        for _ in range(20):
            mab.update("code_review", "postgresql", "graph_boost", reward=1.0)

        overrides = mab.export_overrides()
        assert "code_review_postgresql" in overrides
        override = overrides["code_review_postgresql"]
        # Should use code_review base (graph_depth=1, not general's 2)
        base = TASK_STRATEGIES["code_review"]
        assert override["graph_depth"] == base["graph_depth"] + 1  # graph_boost delta

    def test_save_overrides_backward_compat(self, tmp_path: Path) -> None:
        """save_overrides writes JSON and returns count (same API as old StrategyOptimizer)."""
        mab = MABStrategyOptimizer(tmp_path)
        mab.select_arm("debugging", "postgresql")
        for _ in range(20):
            mab.update("debugging", "postgresql", "graph_boost", reward=1.0)

        output = tmp_path / "overrides.json"
        count = mab.save_overrides(output)
        assert count >= 1
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Task 7 tests: End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_mab_lifecycle(self, tmp_path: Path) -> None:
        """End-to-end: select -> update -> save -> reload -> verify state."""
        mab = MABStrategyOptimizer(tmp_path)

        # 1. Cold start selection
        arm_id, strategy = mab.select_arm("debugging", "postgresql")
        assert arm_id in ARM_DEFINITIONS

        # 2. Simulate 23 outcomes: graph_boost mostly wins
        for _ in range(15):
            mab.update("debugging", "postgresql", "graph_boost", reward=1.0)
        for _ in range(3):
            mab.update("debugging", "postgresql", "graph_boost", reward=0.0)
        for _ in range(5):
            mab.update("debugging", "postgresql", "default", reward=0.0)

        # 3. Persist
        mab.save_state()
        assert mab._state_path.exists()

        # 4. Reload
        mab2 = MABStrategyOptimizer(tmp_path)
        ctx = mab2._state["debugging_postgresql"]
        assert ctx["arms"]["graph_boost"]["alpha"] == DEFAULT_PRIORS["graph_boost"]["alpha"] + 15
        assert ctx["arms"]["graph_boost"]["beta"] == DEFAULT_PRIORS["graph_boost"]["beta"] + 3
        assert ctx["total_pulls"] == 23

        # 5. Export overrides — graph_boost should dominate
        overrides = mab2.export_overrides()
        assert "debugging_postgresql" in overrides
        assert overrides["debugging_postgresql"]["arm_id"] == "graph_boost"

        # 6. Selection should favor graph_boost
        selections: dict[str, int] = {}
        for _ in range(100):
            arm, _ = mab2.select_arm("debugging", "postgresql")
            selections[arm] = selections.get(arm, 0) + 1
        assert selections.get("graph_boost", 0) > 50

    def test_multiple_contexts_independent(self, tmp_path: Path) -> None:
        """Different contexts maintain independent arm states."""
        mab = MABStrategyOptimizer(tmp_path)

        # Train debugging_postgresql -> graph_boost
        for _ in range(20):
            mab.update("debugging", "postgresql", "graph_boost", reward=1.0)

        # Train architecture_fastapi -> vector_focus
        for _ in range(20):
            mab.update("architecture", "fastapi", "vector_focus", reward=1.0)

        mab.save_state()
        mab2 = MABStrategyOptimizer(tmp_path)

        # Verify contexts are independent
        pg = mab2._state["debugging_postgresql"]["arms"]
        fa = mab2._state["architecture_fastapi"]["arms"]

        assert pg["graph_boost"]["alpha"] > pg["vector_focus"]["alpha"]
        assert fa["vector_focus"]["alpha"] > fa["graph_boost"]["alpha"]
