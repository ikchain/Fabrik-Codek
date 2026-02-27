# MAB Strategy Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fixed-threshold strategy optimization with Thompson Sampling Multi-Armed Bandit that learns optimal retrieval configurations per (task_type, topic) from outcome feedback.

**Architecture:** Each (task_type, topic) context has 4 discrete arms (retrieval configurations) with Beta distributions. Thompson Sampling selects the arm with highest sampled value. Binary rewards from outcome feedback (accepted=1, rejected=0) update the Beta parameters online. State persists to `data/profile/mab_state.json`.

**Tech Stack:** Python 3.11+, `random.betavariate` (stdlib), pytest, structlog

**Design doc:** `docs/plans/2026-02-26-mab-strategy-optimizer-design.md`

---

### Task 1: MABStrategyOptimizer Core — Arm Definitions and Selection

**Files:**
- Modify: `src/core/strategy_optimizer.py` (full rewrite, currently 171 lines)
- Test: `tests/test_strategy_optimizer.py` (full rewrite, currently 185 lines)

**Context:** The current `StrategyOptimizer` reads outcomes and applies fixed thresholds. We replace it entirely with `MABStrategyOptimizer` that uses Thompson Sampling. We keep the import path (`src.core.strategy_optimizer`) and the class in the same file so existing consumers (`cli.py:1574`) need minimal changes.

**Step 1: Write the failing tests for arm selection**

```python
# tests/test_strategy_optimizer.py — FULL REWRITE
"""Tests for the MAB Strategy Optimizer."""

import json
import random
from datetime import datetime
from pathlib import Path

import pytest

from src.core.strategy_optimizer import (
    ARM_DEFINITIONS,
    DEFAULT_PRIORS,
    MABStrategyOptimizer,
)
from src.core.task_router import TASK_STRATEGIES, RetrievalStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outcome(
    task_type: str = "debugging",
    topic: str = "postgresql",
    outcome: str = "accepted",
    arm_id: str = "default",
) -> dict:
    return {
        "task_type": task_type,
        "topic": topic,
        "outcome": outcome,
        "strategy": {"arm_id": arm_id},
        "timestamp": datetime.now().isoformat(),
    }


def _write_outcomes(path: Path, outcomes: list[dict], date_str: str | None = None) -> None:
    outdir = path / "01-raw" / "outcomes"
    outdir.mkdir(parents=True, exist_ok=True)
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    filepath = outdir / f"{today}_outcomes.jsonl"
    with open(filepath, "w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


# ---------------------------------------------------------------------------
# Task 1 tests: Arm definitions & selection
# ---------------------------------------------------------------------------


class TestArmDefinitions:
    def test_four_arms_defined(self) -> None:
        """ARM_DEFINITIONS has exactly 4 arms."""
        assert len(ARM_DEFINITIONS) == 4
        assert set(ARM_DEFINITIONS.keys()) == {"default", "graph_boost", "deep_graph", "vector_focus"}

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
```

**Step 2: Run tests to verify they fail**

Run: `cd /path/to/project && python -m pytest tests/test_strategy_optimizer.py::TestArmDefinitions -v 2>&1 | head -30`
Expected: FAIL with `ImportError: cannot import name 'ARM_DEFINITIONS'`

**Step 3: Implement MABStrategyOptimizer core**

```python
# src/core/strategy_optimizer.py — FULL REWRITE
"""MAB Strategy Optimizer — Thompson Sampling for retrieval strategy selection.

Replaces fixed-threshold optimization with a Multi-Armed Bandit that learns
optimal retrieval configurations per (task_type, topic) from outcome feedback.
Each context has 4 discrete arms with Beta distributions.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.core.task_router import TASK_STRATEGIES, RetrievalStrategy

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Arm definitions (deltas relative to TASK_STRATEGIES base)
# ---------------------------------------------------------------------------

ARM_DEFINITIONS: dict[str, dict[str, Any]] = {
    "default": {
        "graph_depth_delta": 0,
        "vector_weight_delta": 0.0,
        "graph_weight_delta": 0.0,
        "fulltext_weight": 0.0,
    },
    "graph_boost": {
        "graph_depth_delta": 1,
        "vector_weight_delta": 0.0,
        "graph_weight_delta": 0.1,
        "fulltext_weight": 0.0,
    },
    "deep_graph": {
        "graph_depth_delta": 2,
        "vector_weight_delta": 0.0,
        "graph_weight_delta": 0.2,
        "fulltext_weight": 0.1,
    },
    "vector_focus": {
        "graph_depth_delta": 0,
        "vector_weight_delta": 0.1,
        "graph_weight_delta": -0.1,
        "fulltext_weight": 0.0,
    },
}

# Default priors: Beta(alpha, beta) for each arm
# "default" gets Beta(2,1) warm start; others get Beta(1,1) uniform
DEFAULT_PRIORS: dict[str, dict[str, int]] = {
    "default": {"alpha": 2, "beta": 1},
    "graph_boost": {"alpha": 1, "beta": 1},
    "deep_graph": {"alpha": 1, "beta": 1},
    "vector_focus": {"alpha": 1, "beta": 1},
}


def _context_key(task_type: str, topic: str | None) -> str:
    """Build context key from task_type and optional topic."""
    return f"{task_type}_{topic}" if topic else task_type


def _build_strategy(task_type: str, arm_id: str) -> RetrievalStrategy:
    """Compute a RetrievalStrategy by applying arm deltas to the base strategy."""
    base = TASK_STRATEGIES.get(task_type, TASK_STRATEGIES["general"])
    arm = ARM_DEFINITIONS[arm_id]

    return RetrievalStrategy(
        graph_depth=max(1, base["graph_depth"] + arm["graph_depth_delta"]),
        vector_weight=max(0.0, round(base["vector_weight"] + arm["vector_weight_delta"], 4)),
        graph_weight=max(0.0, round(base["graph_weight"] + arm["graph_weight_delta"], 4)),
        fulltext_weight=arm["fulltext_weight"],
    )


# ---------------------------------------------------------------------------
# MABStrategyOptimizer
# ---------------------------------------------------------------------------


class MABStrategyOptimizer:
    """Multi-Armed Bandit strategy optimizer using Thompson Sampling.

    Maintains Beta distributions per arm per (task_type, topic) context.
    Selects arms via Thompson Sampling, updates with binary rewards.

    Parameters
    ----------
    data_dir:
        Directory containing ``profile/mab_state.json`` for persistence.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._state_path = data_dir / "profile" / "mab_state.json"
        self._state: dict[str, dict] = self._load_state()
        self._dirty: bool = False

    # -- state management ---------------------------------------------------

    def _load_state(self) -> dict:
        """Load MAB state from disk, or return empty dict."""
        if not self._state_path.exists():
            return {}
        try:
            with self._state_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("mab_state_load_failed", path=str(self._state_path))
            return {}

    def save_state(self) -> None:
        """Persist MAB state to disk (only if dirty)."""
        if not self._dirty:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_path.open("w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, ensure_ascii=False)
        self._dirty = False
        logger.info("mab_state_saved", contexts=len(self._state), path=str(self._state_path))

    def _ensure_context(self, key: str) -> None:
        """Create a context entry with default priors if it doesn't exist."""
        if key not in self._state:
            self._state[key] = {
                "arms": {
                    arm_id: {"alpha": prior["alpha"], "beta": prior["beta"]}
                    for arm_id, prior in DEFAULT_PRIORS.items()
                },
                "total_pulls": 0,
                "last_updated": datetime.now().isoformat(),
            }
            self._dirty = True

    # -- core MAB operations ------------------------------------------------

    def select_arm(self, task_type: str, topic: str | None) -> tuple[str, RetrievalStrategy]:
        """Select an arm using Thompson Sampling.

        Returns (arm_id, RetrievalStrategy) tuple.
        """
        key = _context_key(task_type, topic)
        self._ensure_context(key)

        arms = self._state[key]["arms"]
        samples = {
            arm_id: random.betavariate(arm["alpha"], arm["beta"])
            for arm_id, arm in arms.items()
        }
        best_arm = max(samples, key=samples.get)
        strategy = _build_strategy(task_type, best_arm)

        return best_arm, strategy

    def update(self, task_type: str, topic: str | None, arm_id: str, reward: float) -> None:
        """Update arm Beta parameters after observing a reward.

        Parameters
        ----------
        reward:
            1.0 for accepted, 0.0 for rejected.
        """
        key = _context_key(task_type, topic)
        self._ensure_context(key)

        arm = self._state[key]["arms"].get(arm_id)
        if arm is None:
            logger.warning("mab_unknown_arm", arm_id=arm_id, context=key)
            return

        if reward > 0:
            arm["alpha"] += 1
        else:
            arm["beta"] += 1

        self._state[key]["total_pulls"] += 1
        self._state[key]["last_updated"] = datetime.now().isoformat()
        self._dirty = True

    # -- export for backward compatibility ----------------------------------

    def export_overrides(self) -> dict:
        """Export the best arm per context as strategy_overrides.json format.

        For backward compatibility with code that reads static overrides.
        Returns the arm with highest expected value (alpha / (alpha + beta))
        per context, formatted as the old override dict.
        """
        overrides: dict[str, dict] = {}
        for key, ctx in self._state.items():
            arms = ctx["arms"]
            # Find arm with highest expected value
            best_arm = max(
                arms.items(),
                key=lambda item: item[1]["alpha"] / (item[1]["alpha"] + item[1]["beta"]),
            )
            arm_id, params = best_arm

            if arm_id == "default":
                # No override needed — default strategy is fine
                continue

            # Resolve task_type from key
            task_type = key.split("_")[0]
            arm_def = ARM_DEFINITIONS[arm_id]
            base = TASK_STRATEGIES.get(task_type, TASK_STRATEGIES["general"])

            expected_value = params["alpha"] / (params["alpha"] + params["beta"])
            overrides[key] = {
                "graph_depth": max(1, base["graph_depth"] + arm_def["graph_depth_delta"]),
                "vector_weight": round(base["vector_weight"] + arm_def["vector_weight_delta"], 4),
                "graph_weight": round(base["graph_weight"] + arm_def["graph_weight_delta"], 4),
                "fulltext_weight": arm_def["fulltext_weight"],
                "arm_id": arm_id,
                "expected_value": round(expected_value, 4),
                "sample_size": ctx["total_pulls"],
                "updated_at": ctx["last_updated"],
            }

        return overrides

    def save_overrides(self, output_path: Path) -> int:
        """Export overrides and write to JSON file.

        Backward-compatible with old StrategyOptimizer.save_overrides().
        """
        overrides = self.export_overrides()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(overrides, fh, indent=2, ensure_ascii=False)
        logger.info("strategy_overrides_exported", count=len(overrides), path=str(output_path))
        return len(overrides)
```

**Step 4: Run tests to verify they pass**

Run: `cd /path/to/project && python -m pytest tests/test_strategy_optimizer.py::TestArmDefinitions tests/test_strategy_optimizer.py::TestSelectArm -v`
Expected: 7 tests PASS

**Step 5: Commit**

```bash
git add src/core/strategy_optimizer.py tests/test_strategy_optimizer.py
git commit -m "FEAT: MABStrategyOptimizer core with Thompson Sampling arm selection"
```

---

### Task 2: MAB Update and Persistence

**Files:**
- Modify: `src/core/strategy_optimizer.py` (already rewritten in Task 1)
- Test: `tests/test_strategy_optimizer.py`

**Context:** Task 1 implemented select_arm. Now we test update() and persistence (save_state/load_state round-trip).

**Step 1: Write the failing tests**

Add these test classes to `tests/test_strategy_optimizer.py`:

```python
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
```

**Step 2: Run tests to verify they pass**

Run: `cd /path/to/project && python -m pytest tests/test_strategy_optimizer.py::TestUpdate tests/test_strategy_optimizer.py::TestPersistence -v`
Expected: 9 tests PASS (implementation was already written in Task 1)

**Step 3: Commit**

```bash
git add tests/test_strategy_optimizer.py
git commit -m "TEST: Add MAB update and persistence tests"
```

---

### Task 3: Thompson Sampling Convergence and Export

**Files:**
- Test: `tests/test_strategy_optimizer.py`

**Context:** Verify that Thompson Sampling converges (prefers a dominant arm) and that export_overrides produces backward-compatible output.

**Step 1: Write the convergence and export tests**

Add these test classes to `tests/test_strategy_optimizer.py`:

```python
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

        selections = {"graph_boost": 0}
        for _ in range(200):
            arm_id, _ = mab.select_arm("debugging", "postgresql")
            selections[arm_id] = selections.get(arm_id, 0) + 1

        # graph_boost should dominate (Beta(51,6) ≈ 0.89 expected value)
        assert selections["graph_boost"] > 120  # >60% of 200

    def test_poorly_performing_arm_avoided(self, tmp_path: Path) -> None:
        """An arm with many failures should be selected rarely."""
        mab = MABStrategyOptimizer(tmp_path)
        # Make deep_graph fail a lot
        for _ in range(30):
            mab.update("debugging", "postgresql", "deep_graph", reward=0.0)

        selections = {"deep_graph": 0}
        for _ in range(100):
            arm_id, _ = mab.select_arm("debugging", "postgresql")
            selections[arm_id] = selections.get(arm_id, 0) + 1

        # deep_graph should be very rare (Beta(1,31) ≈ 0.03 expected value)
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
```

**Step 2: Run tests to verify they pass**

Run: `cd /path/to/project && python -m pytest tests/test_strategy_optimizer.py::TestThompsonConvergence tests/test_strategy_optimizer.py::TestExportOverrides -v`
Expected: 6 tests PASS

**Step 3: Commit**

```bash
git add tests/test_strategy_optimizer.py
git commit -m "TEST: Add Thompson convergence and export tests"
```

---

### Task 4: TaskRouter Integration — MAB Arm Selection

**Files:**
- Modify: `src/core/task_router.py:349-412` (TaskRouter class)
- Test: `tests/test_task_router.py` (add MAB integration tests)

**Context:** TaskRouter currently loads static `strategy_overrides.json` in `__init__` and applies them in `route()` step 5. We need to: (a) accept an optional `MABStrategyOptimizer` instance, (b) use `mab.select_arm()` in step 5 when MAB is available, (c) fall back to static overrides when MAB is not available, (d) include `arm_id` in the `RoutingDecision`.

**Step 1: Write the failing test**

Add to `tests/test_task_router.py`:

```python
class TestMABIntegration:
    """Tests for TaskRouter integration with MABStrategyOptimizer."""

    @pytest.fixture
    def mab_router(self, tmp_path):
        """Router with a MABStrategyOptimizer attached."""
        from src.core.strategy_optimizer import MABStrategyOptimizer

        competence_map = CompetenceMap(topics=[], built_at="")
        profile = PersonalProfile(domain="", stack=[], patterns=[], tools=[])
        settings = SimpleNamespace(
            default_model="test-model",
            fallback_model="fallback-model",
            data_dir=tmp_path,
        )
        mab = MABStrategyOptimizer(tmp_path)
        return TaskRouter(competence_map, profile, settings, mab=mab)

    @pytest.mark.asyncio
    async def test_route_with_mab_returns_arm_id(self, mab_router):
        """When MAB is provided, RoutingDecision.arm_id is set."""
        decision = await mab_router.route("fix this error in postgresql")
        assert hasattr(decision, "arm_id")
        assert decision.arm_id in {"default", "graph_boost", "deep_graph", "vector_focus"}

    @pytest.mark.asyncio
    async def test_route_without_mab_arm_id_is_none(self, tmp_path):
        """When no MAB is provided, arm_id is None (backward compat)."""
        competence_map = CompetenceMap(topics=[], built_at="")
        profile = PersonalProfile(domain="", stack=[], patterns=[], tools=[])
        settings = SimpleNamespace(
            default_model="test-model",
            fallback_model="fallback-model",
            data_dir=tmp_path,
        )
        router = TaskRouter(competence_map, profile, settings)
        decision = await router.route("fix this error")
        assert decision.arm_id is None
```

**Step 2: Run test to verify it fails**

Run: `cd /path/to/project && python -m pytest tests/test_task_router.py::TestMABIntegration::test_route_with_mab_returns_arm_id -v 2>&1 | tail -10`
Expected: FAIL (TaskRouter doesn't accept `mab` parameter yet)

**Step 3: Modify TaskRouter**

In `src/core/task_router.py`, make these changes:

1. Add `arm_id` field to `RoutingDecision`:

```python
@dataclass
class RoutingDecision:
    """Complete routing decision for a user query."""

    task_type: str
    topic: str | None
    competence_level: str
    model: str
    strategy: RetrievalStrategy
    system_prompt: str
    classification_method: str  # "keyword" or "llm"
    arm_id: str | None = None   # MAB arm ID, None if no MAB
```

2. Modify `TaskRouter.__init__` to accept optional MAB:

```python
def __init__(
    self,
    competence_map: CompetenceMap,
    profile: PersonalProfile,
    settings: Any,
    mab: Any = None,  # Optional MABStrategyOptimizer
) -> None:
    self.competence_map = competence_map
    self.profile = profile
    self.default_model: str = getattr(settings, "default_model", "")
    self.fallback_model: str = getattr(settings, "fallback_model", "")
    self._strategy_overrides: dict = self._load_overrides(settings)
    self._mab = mab
```

3. Modify `route()` step 5 to use MAB when available:

```python
    # 5. Get retrieval strategy
    arm_id: str | None = None

    if self._mab is not None:
        # MAB Thompson Sampling
        arm_id, strategy = self._mab.select_arm(task_type, topic)
    else:
        strategy = get_strategy(task_type)

        # Apply static strategy override if available ( fallback)
        override_key = f"{task_type}_{topic}" if topic else task_type
        override = self._strategy_overrides.get(override_key)
        if override:
            strategy = RetrievalStrategy(
                use_rag=strategy.use_rag,
                use_graph=strategy.use_graph,
                graph_depth=override.get("graph_depth", strategy.graph_depth),
                vector_weight=override.get("vector_weight", strategy.vector_weight),
                graph_weight=override.get("graph_weight", strategy.graph_weight),
                fulltext_weight=override.get("fulltext_weight", strategy.fulltext_weight),
            )
```

4. Include `arm_id` in the `RoutingDecision` construction:

```python
    decision = RoutingDecision(
        task_type=task_type,
        topic=topic,
        competence_level=competence_level,
        model=model,
        strategy=strategy,
        system_prompt=system_prompt,
        classification_method=classification_method,
        arm_id=arm_id,
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd /path/to/project && python -m pytest tests/test_task_router.py -v 2>&1 | tail -20`
Expected: ALL existing tests PASS + 2 new tests PASS (73 total)

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Integrate MAB arm selection in TaskRouter"
```

---

### Task 5: CLI Integration — MAB in Chat and Ask Loops

**Files:**
- Modify: `src/interfaces/cli.py:64-161` (chat command), `src/interfaces/cli.py:186-294` (ask command), `src/interfaces/cli.py:1574-1582` (competence build)
- Test: `tests/test_cli.py` (smoke test for MAB integration)

**Context:** The CLI needs 3 changes:
1. **chat** command: Create MABStrategyOptimizer, pass to TaskRouter, update after each outcome
2. **ask** command: Create MABStrategyOptimizer, pass to TaskRouter
3. **competence build**: Use MABStrategyOptimizer.save_overrides() instead of old StrategyOptimizer

**Step 1: Modify the `chat` command (lines 64-161)**

After the TaskRouter creation (line 71), add MAB initialization:

```python
# After line 71: router = TaskRouter(competence_map, active_profile, settings)
# Replace with:
from src.core.strategy_optimizer import MABStrategyOptimizer

mab = MABStrategyOptimizer(settings.data_dir)
router = TaskRouter(competence_map, active_profile, settings, mab=mab)
```

After the outcome bridge block (lines 138-146), add MAB update:

```python
            # Update MAB with outcome feedback
            if prev_outcome is not None and prev_outcome.outcome != "neutral":
                arm_id = prev_outcome.strategy.get("arm_id")
                if arm_id:
                    mab.update(
                        task_type=prev_outcome.task_type,
                        topic=prev_outcome.topic,
                        arm_id=arm_id,
                        reward=1.0 if prev_outcome.outcome == "accepted" else 0.0,
                    )
```

Before `tracker.close_session()` (line 150), add MAB save:

```python
        mab.save_state()
```

**Step 2: Modify the `ask` command (lines 186-294)**

After the TaskRouter creation (line 196), add MAB:

```python
# Replace lines 196-197 with:
from src.core.strategy_optimizer import MABStrategyOptimizer

mab = MABStrategyOptimizer(settings.data_dir)
router = TaskRouter(competence_map, active_profile, settings, mab=mab)
decision = await router.route(prompt)
```

After `ot.close_session()` (line 292), add MAB save:

```python
            mab.save_state()
```

**Step 3: Modify `competence build` (lines 1574-1582)**

Replace the old StrategyOptimizer usage:

```python
        # Generate strategy overrides from MAB state
        from src.core.strategy_optimizer import MABStrategyOptimizer

        mab = MABStrategyOptimizer(settings.data_dir)
        overrides_path = settings.data_dir / "profile" / "strategy_overrides.json"
        override_count = mab.save_overrides(overrides_path)
        if override_count > 0:
            console.print(f"[bold]Strategy overrides:[/bold] {override_count} generated (MAB)")
        else:
            console.print("[dim]No strategy overrides needed (MAB defaults are fine)[/dim]")
```

**Step 4: Run full test suite**

Run: `cd /path/to/project && python -m pytest tests/test_strategy_optimizer.py tests/test_task_router.py tests/test_cli.py -v 2>&1 | tail -30`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/interfaces/cli.py
git commit -m "FEAT: Integrate MAB in CLI chat, ask, and competence build"
```

---

### Task 6: Store arm_id in OutcomeRecord Strategy Dict

**Files:**
- Modify: `src/flywheel/outcome_tracker.py:229-239` (record_turn method)
- Test: `tests/test_outcome_tracker.py` (add arm_id test)

**Context:** When OutcomeTracker stores the strategy dict from a RoutingDecision, it should include the `arm_id` so that when the outcome is later used for MAB updates, we know which arm was selected.

**Step 1: Write the failing test**

Add to `tests/test_outcome_tracker.py`:

```python
class TestArmIdStorage:
    """Verify arm_id from RoutingDecision is stored in outcome strategy dict."""

    def test_arm_id_stored_in_strategy(self, tmp_path: Path) -> None:
        """When RoutingDecision has arm_id, it appears in outcome strategy."""
        tracker = OutcomeTracker(tmp_path, "test-session")

        Decision = type(
            "Decision",
            (),
            {
                "task_type": "debugging",
                "topic": "postgresql",
                "competence_level": "Competent",
                "model": "test-model",
                "strategy": type("S", (), {
                    "use_rag": True,
                    "use_graph": True,
                    "graph_depth": 2,
                    "vector_weight": 0.6,
                    "graph_weight": 0.4,
                    "fulltext_weight": 0.0,
                })(),
                "arm_id": "graph_boost",
            },
        )

        tracker.record_turn("query1", "response1", Decision(), 100.0)
        outcome = tracker.record_turn("different topic query", "response2", Decision(), 100.0)

        assert outcome is not None
        assert outcome.strategy.get("arm_id") == "graph_boost"
```

**Step 2: Run test to verify it fails**

Run: `cd /path/to/project && python -m pytest tests/test_outcome_tracker.py::TestArmIdStorage -v 2>&1 | tail -10`
Expected: FAIL (arm_id not yet stored)

**Step 3: Modify OutcomeTracker.record_turn**

In `src/flywheel/outcome_tracker.py`, modify the `_pending` dict construction in `record_turn` (around line 230-239):

```python
        # Store current turn as new pending
        strategy_dict = asdict(decision.strategy) if hasattr(decision, "strategy") else {}
        arm_id = getattr(decision, "arm_id", None)
        if arm_id is not None:
            strategy_dict["arm_id"] = arm_id

        self._pending = {
            "query": query,
            "response": response[:_SUMMARY_MAX_LEN],
            "task_type": getattr(decision, "task_type", "general"),
            "topic": getattr(decision, "topic", None),
            "competence_level": getattr(decision, "competence_level", "Unknown"),
            "model": getattr(decision, "model", ""),
            "strategy": strategy_dict,
            "latency_ms": latency_ms,
        }
```

**Step 4: Run test to verify it passes**

Run: `cd /path/to/project && python -m pytest tests/test_outcome_tracker.py -v 2>&1 | tail -20`
Expected: ALL existing tests PASS + 1 new test PASS

**Step 5: Commit**

```bash
git add src/flywheel/outcome_tracker.py tests/test_outcome_tracker.py
git commit -m "FEAT: Store arm_id in OutcomeRecord strategy dict"
```

---

### Task 7: Full Integration Test and Regression Suite

**Files:**
- Test: `tests/test_strategy_optimizer.py` (add integration test)

**Context:** Run the full test suite to verify no regressions. Add one end-to-end integration test that exercises the full MAB flow: select → update → save → load → verify convergence.

**Step 1: Write the integration test**

Add to `tests/test_strategy_optimizer.py`:

```python
class TestEndToEnd:
    def test_full_mab_lifecycle(self, tmp_path: Path) -> None:
        """End-to-end: select → update → save → reload → verify state."""
        mab = MABStrategyOptimizer(tmp_path)

        # 1. Cold start selection
        arm_id, strategy = mab.select_arm("debugging", "postgresql")
        assert arm_id in ARM_DEFINITIONS

        # 2. Simulate 20 outcomes: graph_boost mostly wins
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
        selections = {}
        for _ in range(100):
            arm, _ = mab2.select_arm("debugging", "postgresql")
            selections[arm] = selections.get(arm, 0) + 1
        assert selections.get("graph_boost", 0) > 50

    def test_multiple_contexts_independent(self, tmp_path: Path) -> None:
        """Different contexts maintain independent arm states."""
        mab = MABStrategyOptimizer(tmp_path)

        # Train debugging_postgresql → graph_boost
        for _ in range(20):
            mab.update("debugging", "postgresql", "graph_boost", reward=1.0)

        # Train architecture_fastapi → vector_focus
        for _ in range(20):
            mab.update("architecture", "fastapi", "vector_focus", reward=1.0)

        mab.save_state()
        mab2 = MABStrategyOptimizer(tmp_path)

        # Verify contexts are independent
        pg = mab2._state["debugging_postgresql"]["arms"]
        fa = mab2._state["architecture_fastapi"]["arms"]

        assert pg["graph_boost"]["alpha"] > pg["vector_focus"]["alpha"]
        assert fa["vector_focus"]["alpha"] > fa["graph_boost"]["alpha"]
```

**Step 2: Run the integration tests**

Run: `cd /path/to/project && python -m pytest tests/test_strategy_optimizer.py::TestEndToEnd -v`
Expected: 2 tests PASS

**Step 3: Run full regression suite**

Run: `cd /path/to/project && python -m pytest --tb=short 2>&1 | tail -20`
Expected: ALL tests PASS (existing 847+ tests, plus ~25 new ones)

**Step 4: Commit**

```bash
git add tests/test_strategy_optimizer.py
git commit -m "TEST: Add MAB end-to-end integration tests"
```

---

### Task 8: Final Cleanup and Feature Commit

**Files:**
- Review all modified files

**Step 1: Verify all tests pass**

Run: `cd /path/to/project && python -m pytest --tb=short -q 2>&1 | tail -5`
Expected: All PASS, 0 failures

**Step 2: Verify no import errors**

Run: `cd /path/to/project && python -c "from src.core.strategy_optimizer import MABStrategyOptimizer, ARM_DEFINITIONS, DEFAULT_PRIORS; print('OK')" && python -c "from src.core.task_router import RoutingDecision; r = RoutingDecision('a', None, 'b', 'c', None, '', 'keyword'); print(f'arm_id={r.arm_id}')" 2>&1`
Expected: `OK` and `arm_id=None`

**Step 3: Review git diff for any missed changes**

Run: `cd /path/to/project && git diff --stat HEAD~7`
Expected: 5 files changed:
- `src/core/strategy_optimizer.py`
- `src/core/task_router.py`
- `src/flywheel/outcome_tracker.py`
- `src/interfaces/cli.py`
- `tests/test_strategy_optimizer.py`
- `tests/test_task_router.py`
- `tests/test_outcome_tracker.py`
