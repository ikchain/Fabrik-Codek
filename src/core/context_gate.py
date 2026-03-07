"""Context Gate — decides WHETHER to inject personalization context.

Evaluates heuristic signals to determine if RAG/graph context will help
or hurt the response. Generic queries with no entity matches skip context
injection, avoiding the personalization paradox where irrelevant context
degrades LLM output quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# Task types that always benefit from context injection
ALWAYS_INJECT_TASKS: frozenset[str] = frozenset({"debugging", "code_review", "architecture"})

# Task types where context is usually noise
DEFAULT_SKIP_TASKS: frozenset[str] = frozenset({"general"})


@dataclass
class GateDecision:
    """Result of context gate evaluation."""

    inject: bool
    reason: str
    confidence: float
    signals: dict = field(default_factory=dict)


class ContextGate:
    """Decides whether to inject personalization context for a query.

    Uses 4 heuristic signals (no ML):
      1. Entity density -- no entities -> skip
      2. Competence level -- Unknown -> skip
      3. Task type rules -- debugging/code_review/architecture -> always inject
      4. Topic presence -- no topic detected -> skip vote

    Voting: inject_votes vs skip_votes, tie -> inject (conservative).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        always_inject_tasks: frozenset[str] | None = None,
        skip_tasks: frozenset[str] | None = None,
        entity_recognizer: Any = None,
    ) -> None:
        self.enabled = enabled
        self.always_inject_tasks = (
            always_inject_tasks if always_inject_tasks is not None else ALWAYS_INJECT_TASKS
        )
        self.skip_tasks = skip_tasks if skip_tasks is not None else DEFAULT_SKIP_TASKS
        self._entity_recognizer = entity_recognizer

    def evaluate(
        self,
        query: str,
        decision: Any,
        entity_count: int | None = None,
    ) -> GateDecision:
        """Evaluate whether to inject context for a query.

        Args:
            query: The user query.
            decision: A RoutingDecision (or any object with task_type, topic,
                      competence_level attributes).
            entity_count: Optional pre-computed entity count. If None and an
                         entity_recognizer is set, it will be called.

        Returns:
            GateDecision with inject=True/False and reasoning.
        """
        if not self.enabled:
            return GateDecision(
                inject=True,
                reason="gate_disabled",
                confidence=1.0,
                signals={"gate_enabled": False},
            )

        task_type = getattr(decision, "task_type", "general")
        topic = getattr(decision, "topic", None)
        competence_level = getattr(decision, "competence_level", "Unknown")

        inject_votes = 0
        skip_votes = 0
        signals: dict = {}

        # Signal 1: Entity density
        if entity_count is None and self._entity_recognizer is not None:
            entity_count = len(self._entity_recognizer(query))
        if entity_count is not None:
            signals["entity_count"] = entity_count
            if entity_count == 0:
                skip_votes += 1
                signals["entity_signal"] = "skip"
            else:
                inject_votes += 1
                signals["entity_signal"] = "inject"

        # Signal 2: Competence level
        signals["competence_level"] = competence_level
        if competence_level == "Unknown":
            skip_votes += 1
            signals["competence_signal"] = "skip"
        else:
            inject_votes += 1
            signals["competence_signal"] = "inject"

        # Signal 3: Task type rules -- always_inject is unconditional override
        signals["task_type"] = task_type
        if task_type in self.always_inject_tasks:
            signals["task_signal"] = "always_inject"
            signals["inject_votes"] = inject_votes + 1
            signals["skip_votes"] = skip_votes
            return GateDecision(
                inject=True,
                reason="inject(task_signal:always_inject)",
                confidence=1.0,
                signals=signals,
            )
        elif task_type in self.skip_tasks:
            skip_votes += 1
            signals["task_signal"] = "skip"
        else:
            inject_votes += 1
            signals["task_signal"] = "inject"

        # Signal 4: Topic presence
        signals["topic"] = topic
        if topic is None:
            skip_votes += 1
            signals["topic_signal"] = "skip"
        else:
            inject_votes += 1
            signals["topic_signal"] = "inject"

        signals["inject_votes"] = inject_votes
        signals["skip_votes"] = skip_votes

        # Voting: tie -> inject (conservative)
        inject = inject_votes >= skip_votes

        # Confidence: how decisive was the vote
        total = inject_votes + skip_votes
        if total > 0:
            confidence = abs(inject_votes - skip_votes) / total
        else:
            confidence = 0.0

        # Build reason
        if inject:
            reasons = [
                k
                for k in ("entity_signal", "competence_signal", "task_signal", "topic_signal")
                if signals.get(k) in ("inject", "always_inject")
            ]
            reason = f"inject({','.join(reasons)})"
        else:
            reasons = [
                k
                for k in ("entity_signal", "competence_signal", "task_signal", "topic_signal")
                if signals.get(k) == "skip"
            ]
            reason = f"skip({','.join(reasons)})"

        gate_decision = GateDecision(
            inject=inject,
            reason=reason,
            confidence=round(confidence, 3),
            signals=signals,
        )

        logger.info(
            "context_gate_evaluated",
            inject=gate_decision.inject,
            reason=gate_decision.reason,
            confidence=gate_decision.confidence,
            task_type=task_type,
            topic=topic,
        )

        return gate_decision
