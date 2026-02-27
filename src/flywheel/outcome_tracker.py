"""Outcome Tracker - Infers whether a response was accepted or rejected (FC-38).

Uses heuristics on consecutive queries to determine if the user was satisfied
with the previous response. This signal feeds back into the Competence Model
and strategy optimisation loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEGATION_KEYWORDS: tuple[str, ...] = (
    # English
    "no,",
    "no ",
    "not ",
    "wrong",
    "incorrect",
    "that's not",
    "thats not",
    "don't",
    "dont ",
    "didn't",
    "didnt ",
    "isn't",
    "isnt ",
    "wasn't",
    "wasnt ",
    # Spanish
    "no,",
    "no ",
    "mal,",
    "mal ",
    "incorrecto",
    "eso no",
    "esta mal",
    "está mal",
    "no es",
)

SIMILARITY_THRESHOLD: float = 0.5
NEGATION_SIMILARITY_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OutcomeRecord:
    """A single outcome observation for the flywheel feedback loop."""

    # Required fields (no default)
    query: str
    response_summary: str
    task_type: str
    model: str

    # Auto-populated
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Context from routing
    topic: str | None = None
    competence_level: str = "Unknown"
    strategy: dict = field(default_factory=dict)

    # Outcome
    outcome: Literal["accepted", "rejected", "neutral"] = "neutral"
    inference_reason: str = ""

    # Performance
    latency_ms: float = 0.0
    session_id: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Token similarity
# ---------------------------------------------------------------------------


def token_similarity(query_a: str, query_b: str) -> float:
    """Compute token overlap between two queries.

    Returns ``|intersection| / min(|a|, |b|)``, or 0.0 when either
    query is empty or whitespace-only.
    """
    tokens_a = set(query_a.lower().split())
    tokens_b = set(query_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    overlap = len(tokens_a & tokens_b)
    denominator = min(len(tokens_a), len(tokens_b))
    return overlap / denominator


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------


def _starts_with_negation(text: str) -> bool:
    """Check whether the first 50 characters contain a negation keyword."""
    if not text:
        return False

    prefix = text[:50].lower()
    return any(prefix.startswith(kw) or f" {kw}" in prefix for kw in NEGATION_KEYWORDS)


# ---------------------------------------------------------------------------
# Outcome inference (pure function)
# ---------------------------------------------------------------------------


def infer_outcome(
    previous_query: str,
    new_query: str,
) -> tuple[str, str]:
    """Infer whether the previous response was accepted or rejected.

    Heuristic rules (applied in order):
    1. If the new query starts with a negation keyword **and** token similarity
       exceeds ``NEGATION_SIMILARITY_THRESHOLD`` -> ``"rejected"`` (explicit
       negation on the same topic).
    2. If token similarity >= ``SIMILARITY_THRESHOLD`` (without negation)
       -> ``"rejected"`` (reformulation implies dissatisfaction).
    3. Otherwise -> ``"accepted"`` (topic change or constructive follow-up).

    Returns:
        A ``(outcome, reason)`` tuple.
    """
    if not previous_query or not previous_query.strip():
        return ("accepted", "no previous query")

    if not new_query or not new_query.strip():
        return ("accepted", "no new query")

    similarity = token_similarity(previous_query, new_query)
    has_negation = _starts_with_negation(new_query)

    # Rule 1: explicit negation on a related topic
    if has_negation and similarity > NEGATION_SIMILARITY_THRESHOLD:
        return (
            "rejected",
            f"negation detected with similarity {similarity:.2f}",
        )

    # Rule 2: high similarity implies reformulation
    if similarity >= SIMILARITY_THRESHOLD:
        return (
            "rejected",
            f"reformulation detected with similarity {similarity:.2f}",
        )

    # Rule 3: default
    return ("accepted", f"topic change or follow-up (similarity {similarity:.2f})")


# ---------------------------------------------------------------------------
# Stateful tracker
# ---------------------------------------------------------------------------

_SUMMARY_MAX_LEN = 200


class OutcomeTracker:
    """Stateful tracker that holds a pending turn and infers its outcome
    when the next turn arrives.

    Persists outcomes as JSONL files under
    ``datalake_path / "01-raw" / "outcomes"``.
    """

    def __init__(self, datalake_path: Path, session_id: str) -> None:
        self.datalake_path = datalake_path
        self.session_id = session_id
        self._pending: dict | None = None
        self._outcomes: list[OutcomeRecord] = []

        # Ensure output directory exists
        self._output_dir = datalake_path / "01-raw" / "outcomes"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------

    def record_turn(
        self,
        query: str,
        response: str,
        decision: Any,
        latency_ms: float,
    ) -> OutcomeRecord | None:
        """Record a conversational turn.

        If a pending turn exists, infer its outcome by comparing its query
        with the new *query*, then persist and return the resulting
        :class:`OutcomeRecord`.  The current turn becomes the new pending.

        Returns ``None`` on the very first turn (no previous to evaluate).
        """
        result: OutcomeRecord | None = None

        if self._pending is not None:
            outcome, reason = infer_outcome(self._pending["query"], query)
            result = self._finalize_pending(outcome, reason)

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

        return result

    def close_session(self) -> OutcomeRecord | None:
        """Finalize the pending turn (if any) as *neutral* with reason
        ``"session_close"``.  Returns the record or ``None`` if the session
        had no pending turn."""
        if self._pending is None:
            return None
        return self._finalize_pending("neutral", "session_close")

    def get_session_stats(self) -> dict:
        """Return aggregate counts for this session."""
        accepted = sum(1 for r in self._outcomes if r.outcome == "accepted")
        rejected = sum(1 for r in self._outcomes if r.outcome == "rejected")
        neutral = sum(1 for r in self._outcomes if r.outcome == "neutral")
        return {
            "total": len(self._outcomes),
            "accepted": accepted,
            "rejected": rejected,
            "neutral": neutral,
        }

    # -- internals ---------------------------------------------------------

    def _finalize_pending(self, outcome: str, reason: str) -> OutcomeRecord:
        """Create an :class:`OutcomeRecord` from pending data, persist it,
        and clear the pending slot."""
        assert self._pending is not None  # caller guarantees

        record = OutcomeRecord(
            query=self._pending["query"],
            response_summary=self._pending["response"],
            task_type=self._pending["task_type"],
            model=self._pending["model"],
            topic=self._pending["topic"],
            competence_level=self._pending["competence_level"],
            strategy=self._pending["strategy"],
            outcome=outcome,  # type: ignore[arg-type]
            inference_reason=reason,
            latency_ms=self._pending["latency_ms"],
            session_id=self.session_id,
        )

        self._outcomes.append(record)
        self._persist(record)
        self._pending = None

        logger.info(
            "outcome_recorded",
            outcome=outcome,
            reason=reason,
            query=record.query[:80],
            session_id=self.session_id,
        )

        return record

    def _persist(self, record: OutcomeRecord) -> None:
        """Append a single record as one JSON line to today's JSONL file."""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self._output_dir / f"{today}_outcomes.jsonl"
        with filepath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
