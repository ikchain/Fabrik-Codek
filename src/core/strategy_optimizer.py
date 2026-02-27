"""MAB Strategy Optimizer — Thompson Sampling for retrieval strategy selection (FC-42).

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
        use_rag=base.get("use_rag", True),
        use_graph=base.get("use_graph", True),
        graph_depth=max(1, base["graph_depth"] + arm["graph_depth_delta"]),
        vector_weight=max(0.0, round(base["vector_weight"] + arm["vector_weight_delta"], 4)),
        graph_weight=max(0.0, round(base["graph_weight"] + arm["graph_weight_delta"], 4)),
        fulltext_weight=arm["fulltext_weight"],
        confidence_threshold=base.get("confidence_threshold", 0.7),
        min_k=base.get("min_k", 1),
        max_k=base.get("max_k", 8),
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
            arm_id: random.betavariate(arm["alpha"], arm["beta"]) for arm_id, arm in arms.items()
        }
        best_arm = max(samples, key=samples.get)  # type: ignore[arg-type]
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

            # Resolve task_type from key by checking against known task types
            task_type = "general"
            for tt in TASK_STRATEGIES:
                if key == tt or key.startswith(f"{tt}_"):
                    task_type = tt
                    break
            arm_def = ARM_DEFINITIONS[arm_id]
            base = TASK_STRATEGIES.get(task_type, TASK_STRATEGIES["general"])

            expected_value = params["alpha"] / (params["alpha"] + params["beta"])
            overrides[key] = {
                "graph_depth": max(1, base["graph_depth"] + arm_def["graph_depth_delta"]),
                "vector_weight": max(
                    0.0, round(base["vector_weight"] + arm_def["vector_weight_delta"], 4)
                ),
                "graph_weight": max(
                    0.0, round(base["graph_weight"] + arm_def["graph_weight_delta"], 4)
                ),
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
