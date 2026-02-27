# Outcome Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the cognitive loop by tracking whether Fabrik's responses were useful, and feeding that data back into the Competence Model and retrieval strategies.

**Architecture:** Observer pattern — an OutcomeTracker silently observes conversational patterns in the CLI chat loop to infer outcomes (accepted/rejected/neutral) without any UX changes. Outcomes feed into CompetenceModel as a 4th signal and a StrategyOptimizer adjusts retrieval parameters for underperforming task_type+topic combos.

**Tech Stack:** Python 3.11+, dataclasses, structlog, pytest, typer/rich (CLI)

**Design doc:** `docs/plans/2026-02-24-outcome-tracking-design.md`

---

### Task 1: OutcomeRecord Data Model + infer_outcome

**Files:**
- Create: `src/flywheel/outcome_tracker.py`
- Create: `tests/test_outcome_tracker.py`

**Step 1: Write failing tests for infer_outcome**

```python
# tests/test_outcome_tracker.py
"""Tests for OutcomeTracker."""

import pytest

from src.flywheel.outcome_tracker import infer_outcome


class TestInferOutcome:
    """Tests for the pure inference function."""

    def test_topic_change_is_accepted(self):
        outcome, reason = infer_outcome(
            "fix the postgres query", "how do I deploy with docker"
        )
        assert outcome == "accepted"

    def test_constructive_followup_is_accepted(self):
        outcome, reason = infer_outcome(
            "explain async in python", "great, and how does asyncio.gather work?"
        )
        assert outcome == "accepted"

    def test_rephrased_question_is_rejected(self):
        outcome, reason = infer_outcome(
            "fix the postgres query",
            "the postgres query is still broken, fix it",
        )
        assert outcome == "rejected"
        assert "similarity" in reason

    def test_explicit_negation_is_rejected(self):
        outcome, reason = infer_outcome(
            "explain this error", "no, eso no es correcto"
        )
        assert outcome == "rejected"
        assert "negation" in reason

    def test_negation_unrelated_topic_not_rejected(self):
        """'no' about a different topic should not reject."""
        outcome, reason = infer_outcome(
            "explain async in python",
            "no quiero usar docker para esto",
        )
        assert outcome == "accepted"

    def test_empty_queries(self):
        outcome, reason = infer_outcome("", "hello")
        assert outcome == "accepted"

    def test_same_query_verbatim_is_rejected(self):
        outcome, reason = infer_outcome("fix the bug", "fix the bug")
        assert outcome == "rejected"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_outcome_tracker.py -v`
Expected: ImportError — module does not exist yet

**Step 3: Write OutcomeRecord + infer_outcome + token_similarity**

```python
# src/flywheel/outcome_tracker.py
"""Outcome Tracking for Fabrik-Codek.

Observes conversational patterns to infer whether responses were useful.
Zero friction — no manual feedback, no UX changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from uuid import uuid4

import structlog

logger = structlog.get_logger()

# -- Negation keywords (ES + EN) checked at start of message ----------------

NEGATION_KEYWORDS: list[str] = [
    "no,",
    "no ",
    "incorrecto",
    "mal,",
    "mal ",
    "eso no es",
    "eso no",
    "wrong",
    "incorrect",
    "that's not",
    "nope",
]

SIMILARITY_THRESHOLD: float = 0.5
NEGATION_SIMILARITY_THRESHOLD: float = 0.3


# -- Data model --------------------------------------------------------------


@dataclass
class OutcomeRecord:
    """A single outcome record for the flywheel."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    query: str = ""
    response_summary: str = ""
    task_type: str = ""
    topic: str | None = None
    competence_level: str = ""
    model: str = ""
    strategy: dict = field(default_factory=dict)
    outcome: str = "neutral"  # "accepted" | "rejected" | "neutral"
    inference_reason: str = ""
    latency_ms: float = 0.0
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# -- Pure inference functions -------------------------------------------------


def token_similarity(query_a: str, query_b: str) -> float:
    """Compute token overlap between two queries.

    Uses the smaller set as denominator to detect reformulations
    even when the new query is longer.
    """
    tokens_a = set(query_a.lower().split())
    tokens_b = set(query_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))
    return len(intersection) / smaller


def _starts_with_negation(text: str) -> bool:
    """Check if text starts with a negation keyword."""
    prefix = text.lower().strip()[:50]
    return any(prefix.startswith(kw) for kw in NEGATION_KEYWORDS)


def infer_outcome(previous_query: str, new_query: str) -> tuple[str, str]:
    """Infer outcome of previous turn from the new user message.

    Returns (outcome, reason) tuple.

    Conservative: prefers false-accepted over false-rejected.
    """
    similarity = token_similarity(previous_query, new_query)
    has_negation = _starts_with_negation(new_query)

    # Explicit negation + some topic overlap → rejected
    if has_negation and similarity > NEGATION_SIMILARITY_THRESHOLD:
        return "rejected", f"negation detected (similarity={similarity:.2f})"

    # High similarity without negation → reformulation → rejected
    if similarity >= SIMILARITY_THRESHOLD:
        return "rejected", f"high similarity ({similarity:.2f}) suggests reformulation"

    # Default: topic change or constructive follow-up → accepted
    return "accepted", f"topic change or follow-up (similarity={similarity:.2f})"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_outcome_tracker.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/flywheel/outcome_tracker.py tests/test_outcome_tracker.py
git commit -m "FEAT: Add OutcomeRecord and infer_outcome"
```

---

### Task 2: OutcomeTracker Class

**Files:**
- Modify: `src/flywheel/outcome_tracker.py`
- Modify: `tests/test_outcome_tracker.py`

**Step 1: Write failing tests for OutcomeTracker**

```python
# Append to tests/test_outcome_tracker.py

import json
from pathlib import Path

from src.flywheel.outcome_tracker import OutcomeTracker, OutcomeRecord
from src.core.task_router import RoutingDecision, RetrievalStrategy


def _make_decision(task_type: str = "debugging", topic: str | None = "postgresql") -> RoutingDecision:
    return RoutingDecision(
        task_type=task_type,
        topic=topic,
        competence_level="Expert",
        model="test-model",
        strategy=RetrievalStrategy(),
        system_prompt="test prompt",
        classification_method="keyword",
    )


class TestOutcomeTracker:
    """Tests for the stateful tracker."""

    def test_first_turn_returns_none(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        result = tracker.record_turn("hello", "world", _make_decision(), 100.0)
        assert result is None

    def test_second_turn_returns_outcome_of_first(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        tracker.record_turn("fix postgres query", "here is the fix", _make_decision(), 100.0)
        outcome = tracker.record_turn("now deploy with docker", "ok", _make_decision("devops", "docker"), 50.0)
        assert outcome is not None
        assert outcome.outcome == "accepted"
        assert outcome.query == "fix postgres query"
        assert outcome.task_type == "debugging"

    def test_close_session_marks_neutral(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        tracker.record_turn("explain async", "async is...", _make_decision("explanation"), 200.0)
        outcome = tracker.close_session()
        assert outcome is not None
        assert outcome.outcome == "neutral"
        assert "session_close" in outcome.inference_reason

    def test_close_empty_session_returns_none(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        assert tracker.close_session() is None

    def test_outcomes_persisted_to_jsonl(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        tracker.record_turn("fix bug", "fixed", _make_decision(), 100.0)
        tracker.record_turn("deploy docker", "deployed", _make_decision("devops", "docker"), 50.0)
        tracker.close_session()

        today = datetime.now().strftime("%Y-%m-%d")
        outcome_file = tmp_path / "01-raw" / "outcomes" / f"{today}_outcomes.jsonl"
        assert outcome_file.exists()

        lines = outcome_file.read_text().strip().split("\n")
        assert len(lines) == 2  # first turn + close

        first = json.loads(lines[0])
        assert first["outcome"] == "accepted"
        assert first["session_id"] == "session-1"

    def test_session_stats(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        tracker.record_turn("q1", "a1", _make_decision(), 100.0)
        tracker.record_turn("q2", "a2", _make_decision(), 100.0)
        tracker.record_turn("no, q1 again", "a3", _make_decision(), 100.0)
        tracker.close_session()
        stats = tracker.get_session_stats()
        assert stats["total"] == 3
        assert stats["accepted"] >= 1
        assert stats["neutral"] >= 1

    def test_response_summary_truncated(self, tmp_path: Path):
        tracker = OutcomeTracker(tmp_path, "session-1")
        long_response = "x" * 500
        tracker.record_turn("q", long_response, _make_decision(), 100.0)
        outcome = tracker.close_session()
        assert len(outcome.response_summary) <= 200
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_outcome_tracker.py::TestOutcomeTracker -v`
Expected: FAIL — OutcomeTracker class not defined

**Step 3: Implement OutcomeTracker class**

Append to `src/flywheel/outcome_tracker.py`:

```python
import json
from dataclasses import asdict
from pathlib import Path


class OutcomeTracker:
    """Tracks response outcomes by observing conversational patterns.

    Synchronous, stateful. One instance per session.
    Holds a pending_turn and infers its outcome when the next turn arrives.
    """

    def __init__(self, datalake_path: Path, session_id: str) -> None:
        self.datalake_path = Path(datalake_path)
        self.session_id = session_id
        self._pending: dict | None = None
        self._outcomes: list[OutcomeRecord] = []

        # Ensure output dir exists
        self._output_dir = self.datalake_path / "01-raw" / "outcomes"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def record_turn(
        self,
        query: str,
        response: str,
        decision: RoutingDecision,
        latency_ms: float,
    ) -> OutcomeRecord | None:
        """Record a turn. Returns OutcomeRecord for the PREVIOUS turn
        if one was pending, or None if this is the first turn."""
        from src.core.task_router import RoutingDecision  # noqa: F811 — type only

        previous_outcome: OutcomeRecord | None = None

        # Resolve pending turn
        if self._pending is not None:
            outcome, reason = infer_outcome(self._pending["query"], query)
            previous_outcome = self._finalize_pending(outcome, reason)

        # Store current turn as pending
        self._pending = {
            "query": query,
            "response_summary": response[:200],
            "task_type": decision.task_type,
            "topic": decision.topic,
            "competence_level": decision.competence_level,
            "model": decision.model,
            "strategy": asdict(decision.strategy),
            "latency_ms": latency_ms,
        }

        return previous_outcome

    def close_session(self) -> OutcomeRecord | None:
        """Close session. Marks pending turn as neutral."""
        if self._pending is None:
            return None
        return self._finalize_pending("neutral", "session_close")

    def get_session_stats(self) -> dict:
        """Stats for this session."""
        total = len(self._outcomes)
        accepted = sum(1 for o in self._outcomes if o.outcome == "accepted")
        rejected = sum(1 for o in self._outcomes if o.outcome == "rejected")
        neutral = sum(1 for o in self._outcomes if o.outcome == "neutral")
        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "neutral": neutral,
        }

    def _finalize_pending(self, outcome: str, reason: str) -> OutcomeRecord:
        """Create OutcomeRecord from pending turn, persist, and clear pending."""
        record = OutcomeRecord(
            query=self._pending["query"],
            response_summary=self._pending["response_summary"],
            task_type=self._pending["task_type"],
            topic=self._pending["topic"],
            competence_level=self._pending["competence_level"],
            model=self._pending["model"],
            strategy=self._pending["strategy"],
            outcome=outcome,
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
            task_type=record.task_type,
            topic=record.topic,
            reason=reason,
        )
        return record

    def _persist(self, record: OutcomeRecord) -> None:
        """Append outcome to daily JSONL file."""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self._output_dir / f"{today}_outcomes.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
```

Note: the `RoutingDecision` import inside `record_turn` is a type-only re-import for clarity. The actual import at file top uses `from __future__ import annotations` so the type hint is a string. Remove the inner import if you want — it's only for documentation.

Actually, `RoutingDecision` is not imported at the top of the file. Add this import at the top, under the existing imports:

```python
from src.core.task_router import RoutingDecision, RetrievalStrategy
```

But this creates a circular dependency risk. Since `task_router.py` does NOT import from `outcome_tracker.py`, this is safe. However, to keep things clean, use `TYPE_CHECKING`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.task_router import RoutingDecision
```

And in `record_turn`, accept `decision: Any` at runtime but the type checker sees `RoutingDecision`. Simpler: just use `Any` with a docstring noting the expected type, like the existing codebase pattern for `settings: Any` in task_router.py.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_outcome_tracker.py -v`
Expected: All tests PASS (7 from Task 1 + 7 from Task 2 = 14)

**Step 5: Commit**

```bash
git add src/flywheel/outcome_tracker.py tests/test_outcome_tracker.py
git commit -m "FEAT: Add OutcomeTracker class with persistence"
```

---

### Task 3: StrategyOptimizer

**Files:**
- Create: `src/core/strategy_optimizer.py`
- Create: `tests/test_strategy_optimizer.py`

**Step 1: Write failing tests**

```python
# tests/test_strategy_optimizer.py
"""Tests for StrategyOptimizer."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.core.strategy_optimizer import StrategyOptimizer


def _write_outcomes(path: Path, outcomes: list[dict]) -> None:
    """Write outcome records to a JSONL file."""
    outdir = path / "01-raw" / "outcomes"
    outdir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = outdir / f"{today}_outcomes.jsonl"
    with open(filepath, "w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


def _make_outcome(
    task_type: str = "debugging",
    topic: str = "postgresql",
    outcome: str = "accepted",
) -> dict:
    return {
        "task_type": task_type,
        "topic": topic,
        "outcome": outcome,
        "timestamp": datetime.now().isoformat(),
    }


class TestStrategyOptimizer:

    def test_no_outcomes_no_overrides(self, tmp_path: Path):
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        assert overrides == {}

    def test_below_minimum_sample_no_override(self, tmp_path: Path):
        """Need >= 10 outcomes per combo to generate override."""
        outcomes = [_make_outcome() for _ in range(5)]
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        assert overrides == {}

    def test_high_acceptance_no_override(self, tmp_path: Path):
        """Acceptance >= 0.7 keeps current strategy."""
        outcomes = [_make_outcome(outcome="accepted") for _ in range(10)]
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        assert overrides == {}

    def test_medium_acceptance_mild_boost(self, tmp_path: Path):
        """Acceptance 0.5-0.7 gets graph_depth +1, graph_weight +0.1."""
        outcomes = (
            [_make_outcome(outcome="accepted") for _ in range(6)]
            + [_make_outcome(outcome="rejected") for _ in range(4)]
        )
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        key = "debugging_postgresql"
        assert key in overrides
        assert overrides[key]["graph_depth"] == 3  # default 2 + 1
        assert overrides[key]["graph_weight"] == pytest.approx(0.5 + 0.1, abs=0.01)

    def test_low_acceptance_strong_boost(self, tmp_path: Path):
        """Acceptance < 0.5 gets graph_depth +2, graph_weight +0.2, fulltext 0.1."""
        outcomes = (
            [_make_outcome(outcome="accepted") for _ in range(3)]
            + [_make_outcome(outcome="rejected") for _ in range(7)]
        )
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        key = "debugging_postgresql"
        assert key in overrides
        assert overrides[key]["graph_depth"] == 4  # default 2 + 2
        assert overrides[key]["fulltext_weight"] == pytest.approx(0.1, abs=0.01)

    def test_neutral_outcomes_excluded_from_rate(self, tmp_path: Path):
        """Neutral outcomes don't count toward acceptance rate."""
        outcomes = (
            [_make_outcome(outcome="accepted") for _ in range(8)]
            + [_make_outcome(outcome="neutral") for _ in range(20)]
            + [_make_outcome(outcome="rejected") for _ in range(2)]
        )
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        # 8 accepted / 10 non-neutral = 0.8 → no override
        assert overrides == {}

    def test_save_overrides(self, tmp_path: Path):
        outcomes = (
            [_make_outcome(outcome="accepted") for _ in range(3)]
            + [_make_outcome(outcome="rejected") for _ in range(7)]
        )
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        output = tmp_path / "profile" / "strategy_overrides.json"
        count = opt.save_overrides(output)
        assert count == 1
        assert output.exists()
        data = json.loads(output.read_text())
        assert "debugging_postgresql" in data

    def test_none_topic_uses_task_type_only(self, tmp_path: Path):
        """When topic is None, key is just task_type."""
        outcomes = (
            [_make_outcome(topic=None, outcome="accepted") for _ in range(3)]
            + [_make_outcome(topic=None, outcome="rejected") for _ in range(7)]
        )
        # Fix: topic=None needs to be serialized properly
        for o in outcomes:
            o["topic"] = None
        _write_outcomes(tmp_path, outcomes)
        opt = StrategyOptimizer(tmp_path)
        overrides = opt.compute_overrides()
        assert "debugging" in overrides
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_strategy_optimizer.py -v`
Expected: ImportError

**Step 3: Implement StrategyOptimizer**

```python
# src/core/strategy_optimizer.py
"""Strategy Optimizer for Fabrik-Codek.

Reads outcome data and generates retrieval strategy overrides
for underperforming task_type + topic combinations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from src.core.task_router import TASK_STRATEGIES

logger = structlog.get_logger()

MINIMUM_SAMPLE_SIZE: int = 10
HIGH_ACCEPTANCE_THRESHOLD: float = 0.7
MEDIUM_ACCEPTANCE_THRESHOLD: float = 0.5


class StrategyOptimizer:
    """Reads outcomes, computes acceptance rates, generates strategy overrides."""

    def __init__(self, datalake_path: Path) -> None:
        self.datalake_path = Path(datalake_path)
        self._outcomes_dir = self.datalake_path / "01-raw" / "outcomes"

    def _read_outcomes(self, days: int = 30) -> list[dict]:
        """Read outcome records from the last N days."""
        if not self._outcomes_dir.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        records: list[dict] = []

        for filepath in sorted(self._outcomes_dir.glob("*_outcomes.jsonl")):
            # Parse date from filename YYYY-MM-DD_outcomes.jsonl
            try:
                date_str = filepath.stem.split("_")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    continue
            except (ValueError, IndexError):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return records

    @staticmethod
    def _make_key(task_type: str, topic: str | None) -> str:
        if topic:
            return f"{task_type}_{topic}"
        return task_type

    def compute_overrides(self, days: int = 30) -> dict:
        """Compute strategy overrides for underperforming combinations."""
        records = self._read_outcomes(days)
        if not records:
            return {}

        # Aggregate: key → {accepted: int, rejected: int}
        agg: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})
        for r in records:
            outcome = r.get("outcome", "neutral")
            if outcome == "neutral":
                continue  # neutral doesn't count
            task_type = r.get("task_type", "general")
            topic = r.get("topic")
            key = self._make_key(task_type, topic)
            if outcome == "accepted":
                agg[key]["accepted"] += 1
            elif outcome == "rejected":
                agg[key]["rejected"] += 1

        overrides: dict[str, dict] = {}

        for key, counts in agg.items():
            total = counts["accepted"] + counts["rejected"]
            if total < MINIMUM_SAMPLE_SIZE:
                continue

            rate = counts["accepted"] / total

            if rate >= HIGH_ACCEPTANCE_THRESHOLD:
                continue  # working well

            # Determine base task_type for default strategy lookup
            task_type = key.split("_")[0]
            defaults = TASK_STRATEGIES.get(task_type, TASK_STRATEGIES["general"])

            override = {
                "graph_depth": defaults.get("graph_depth", 2),
                "vector_weight": defaults.get("vector_weight", 0.6),
                "graph_weight": defaults.get("graph_weight", 0.4),
                "fulltext_weight": defaults.get("fulltext_weight", 0.0),
                "acceptance_rate": round(rate, 3),
                "sample_size": total,
                "updated_at": datetime.now().isoformat(),
            }

            if rate >= MEDIUM_ACCEPTANCE_THRESHOLD:
                # Mild boost
                override["graph_depth"] += 1
                override["graph_weight"] = round(override["graph_weight"] + 0.1, 2)
            else:
                # Strong boost
                override["graph_depth"] += 2
                override["graph_weight"] = round(override["graph_weight"] + 0.2, 2)
                override["fulltext_weight"] = 0.1

            overrides[key] = override
            logger.info(
                "strategy_override",
                key=key,
                acceptance_rate=rate,
                sample_size=total,
                graph_depth=override["graph_depth"],
            )

        return overrides

    def save_overrides(self, output_path: Path) -> int:
        """Compute and save overrides. Returns number of overrides generated."""
        overrides = self.compute_overrides()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2, ensure_ascii=False)
        logger.info("strategy_overrides_saved", count=len(overrides), path=str(output_path))
        return len(overrides)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_strategy_optimizer.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/core/strategy_optimizer.py tests/test_strategy_optimizer.py
git commit -m "FEAT: Add StrategyOptimizer"
```

---

### Task 4: CompetenceModel — 4th Signal (outcome_rate)

**Files:**
- Modify: `src/core/competence_model.py`
- Modify: `tests/test_competence_model.py`

**Step 1: Write failing tests for outcome_rate signal**

Append to `tests/test_competence_model.py`:

```python
class TestOutcomeSignal:
    """Tests for outcome_rate as 4th signal in competence scoring."""

    def test_compute_competence_score_with_outcome(self):
        """With all 4 signals, uses updated WEIGHTS_ALL."""
        from src.core.competence_model import compute_competence_score, WEIGHTS_ALL

        ref = datetime.now()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=ref.isoformat(),
            reference_time=ref,
            outcome_rate=0.8,
        )
        assert "outcome" in WEIGHTS_ALL
        assert final > 0

    def test_outcome_rate_none_degrades_gracefully(self):
        """When outcome_rate is None, falls back to 3-signal weights."""
        from src.core.competence_model import compute_competence_score, WEIGHTS_ALL

        ref = datetime.now()
        final_without, _, _, _ = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=ref.isoformat(),
            reference_time=ref,
            outcome_rate=None,
        )
        # Should produce a valid score without outcome signal
        assert 0.0 <= final_without <= 1.0

    def test_high_outcome_rate_boosts_score(self):
        from src.core.competence_model import compute_competence_score

        ref = datetime.now()
        final_high, _, _, _ = compute_competence_score(
            entries=50, edge_count=50, last_activity_iso=ref.isoformat(),
            reference_time=ref, outcome_rate=0.9,
        )
        final_low, _, _, _ = compute_competence_score(
            entries=50, edge_count=50, last_activity_iso=ref.isoformat(),
            reference_time=ref, outcome_rate=0.2,
        )
        assert final_high > final_low


class TestCompetenceBuilderOutcomes:
    """Tests that CompetenceBuilder reads outcomes for the 4th signal."""

    def test_build_reads_outcome_rates(self, tmp_datalake: Path):
        """Build with outcome data produces different scores than without."""
        from src.core.competence_model import CompetenceBuilder

        # Write some outcomes for postgresql
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = outcomes_dir / f"{today}_outcomes.jsonl"
        outcomes = []
        for _ in range(10):
            outcomes.append(json.dumps({
                "task_type": "debugging", "topic": "postgresql",
                "outcome": "accepted", "timestamp": datetime.now().isoformat(),
            }))
        filepath.write_text("\n".join(outcomes) + "\n")

        builder = CompetenceBuilder(tmp_datalake)
        cmap = builder.build()

        # postgresql should have a score (may differ from without outcomes)
        pg = next((t for t in cmap.topics if t.topic == "postgresql"), None)
        assert pg is not None
        assert pg.score > 0

    def test_build_without_outcomes_still_works(self, tmp_datalake: Path):
        """Build without any outcome data degrades gracefully."""
        from src.core.competence_model import CompetenceBuilder

        builder = CompetenceBuilder(tmp_datalake)
        cmap = builder.build()
        assert cmap.total_topics > 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_competence_model.py::TestOutcomeSignal -v`
Expected: FAIL — `compute_competence_score` doesn't accept `outcome_rate` parameter

**Step 3: Modify competence_model.py**

Changes needed in `src/core/competence_model.py`:

1. **Update weight sets** (lines 34-37) to include `"outcome"`:
```python
WEIGHTS_ALL: dict[str, float] = {"entry": 0.30, "density": 0.25, "recency": 0.20, "outcome": 0.25}
WEIGHTS_NO_GRAPH: dict[str, float] = {"entry": 0.40, "recency": 0.30, "outcome": 0.30}
WEIGHTS_NO_RECENCY: dict[str, float] = {"entry": 0.40, "density": 0.30, "outcome": 0.30}
WEIGHTS_ENTRY_ONLY: dict[str, float] = {"entry": 0.60, "outcome": 0.40}
```

2. **Add no-outcome weight sets** for graceful degradation (new constants after line 37):
```python
# Fallback weight sets when outcome_rate is not available (< 5 samples)
WEIGHTS_ALL_NO_OUTCOME: dict[str, float] = {"entry": 0.5, "density": 0.3, "recency": 0.2}
WEIGHTS_NO_GRAPH_NO_OUTCOME: dict[str, float] = {"entry": 0.7, "recency": 0.3}
WEIGHTS_NO_RECENCY_NO_OUTCOME: dict[str, float] = {"entry": 0.6, "density": 0.4}
WEIGHTS_ENTRY_ONLY_NO_OUTCOME: dict[str, float] = {"entry": 1.0}
```

Note: The no-outcome fallbacks are the ORIGINAL weight sets — so existing behavior is preserved when no outcomes exist.

3. **Modify `compute_competence_score`** (line 256) to accept optional `outcome_rate`:
```python
def compute_competence_score(
    entries: int,
    edge_count: int | None,
    last_activity_iso: str,
    reference_time: datetime | None = None,
    outcome_rate: float | None = None,
) -> tuple[float, float, float, float]:
```

Update weight selection logic (lines 281-294) to choose between outcome and no-outcome weight sets based on whether `outcome_rate` is not None:

```python
    has_outcome = outcome_rate is not None

    if edge_count is not None and recency_s is not None:
        weights = WEIGHTS_ALL if has_outcome else WEIGHTS_ALL_NO_OUTCOME
    elif edge_count is None and recency_s is not None:
        weights = WEIGHTS_NO_GRAPH if has_outcome else WEIGHTS_NO_GRAPH_NO_OUTCOME
    elif edge_count is not None and recency_s is None:
        weights = WEIGHTS_NO_RECENCY if has_outcome else WEIGHTS_NO_RECENCY_NO_OUTCOME
    else:
        weights = WEIGHTS_ENTRY_ONLY if has_outcome else WEIGHTS_ENTRY_ONLY_NO_OUTCOME

    final = (
        weights.get("entry", 0.0) * entry_s
        + weights.get("density", 0.0) * density_s
        + weights.get("recency", 0.0) * (recency_s or 0.0)
        + weights.get("outcome", 0.0) * (outcome_rate or 0.0)
    )
```

4. **Add `_get_outcome_rates` method** to CompetenceBuilder (after `_get_topic_recency`):
```python
def _get_outcome_rates(self, topics: list[str]) -> dict[str, float]:
    """Read outcomes and compute acceptance rate per topic.

    Returns rates only for topics with >= 5 non-neutral outcomes.
    """
    outcomes_dir = self.datalake_path / "01-raw" / "outcomes"
    if not outcomes_dir.exists():
        return {}

    from collections import defaultdict
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})

    for filepath in outcomes_dir.glob("*_outcomes.jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                outcome = record.get("outcome", "neutral")
                if outcome == "neutral":
                    continue
                topic = record.get("topic")
                if topic is None:
                    continue
                topic_lower = topic.lower()
                if outcome == "accepted":
                    counts[topic_lower]["accepted"] += 1
                elif outcome == "rejected":
                    counts[topic_lower]["rejected"] += 1

    rates: dict[str, float] = {}
    for topic in topics:
        topic_lower = topic.lower()
        c = counts.get(topic_lower)
        if c is None:
            continue
        total = c["accepted"] + c["rejected"]
        if total < 5:
            continue
        rates[topic_lower] = c["accepted"] / total

    return rates
```

5. **Modify `build()` method** to call `_get_outcome_rates` and pass to `compute_competence_score`:

After the line that gathers recency (approx line 500), add:
```python
outcome_rates = self._get_outcome_rates(topic_list)
```

In the scoring loop, pass outcome_rate:
```python
topic_outcome = outcome_rates.get(topic.lower())
final, entry_s, density_s, recency_s = compute_competence_score(
    entries=entry_count,
    edge_count=edge_count,
    last_activity_iso=last_activity,
    reference_time=ref_time,
    outcome_rate=topic_outcome,
)
```

**Step 4: Run all competence model tests**

Run: `python -m pytest tests/test_competence_model.py -v`
Expected: ALL tests PASS (existing 89 + new ~5)

**Step 5: Commit**

```bash
git add src/core/competence_model.py tests/test_competence_model.py
git commit -m "FEAT: Add outcome_rate as 4th competence signal"
```

---

### Task 5: TaskRouter — Load Strategy Overrides

**Files:**
- Modify: `src/core/task_router.py`
- Modify: `tests/test_task_router.py`

**Step 1: Write failing tests**

Append to `tests/test_task_router.py`:

```python
class TestStrategyOverrides:
    """Tests for TaskRouter loading strategy overrides from ."""

    def _make_router_with_overrides(self, tmp_path: Path, overrides: dict) -> TaskRouter:
        override_path = tmp_path / "profile" / "strategy_overrides.json"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(json.dumps(overrides))

        profile = PersonalProfile(domain="software_development", domain_confidence=0.95)
        cmap = CompetenceMap(topics=[
            CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
            CompetenceEntry(topic="kubernetes", score=0.05, level="Unknown"),
        ])
        mock_settings = MagicMock()
        mock_settings.default_model = "test-model"
        mock_settings.fallback_model = "fallback-model"
        mock_settings.data_dir = tmp_path
        return TaskRouter(cmap, profile, mock_settings)

    def test_override_applied_to_matching_combo(self, tmp_path: Path):
        overrides = {
            "debugging_postgresql": {
                "graph_depth": 4, "graph_weight": 0.6,
                "vector_weight": 0.4, "fulltext_weight": 0.1,
            },
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        decision = asyncio.run(router.route("fix the error in my postgresql query"))
        assert decision.strategy.graph_depth == 4
        assert decision.strategy.fulltext_weight == pytest.approx(0.1)

    def test_no_override_uses_default(self, tmp_path: Path):
        overrides = {
            "debugging_postgresql": {"graph_depth": 4, "graph_weight": 0.6},
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        decision = asyncio.run(router.route("explain how docker networking works"))
        # No override for explanation_docker → uses default
        assert decision.strategy.graph_depth == 2

    def test_no_overrides_file_works(self, tmp_path: Path):
        """Router works normally without overrides file."""
        router = self._make_router_with_overrides(tmp_path, {})
        # Delete the file to simulate no overrides
        override_path = tmp_path / "profile" / "strategy_overrides.json"
        override_path.unlink()
        # Re-create router without file
        profile = PersonalProfile(domain="software_development", domain_confidence=0.95)
        cmap = CompetenceMap(topics=[])
        mock_settings = MagicMock()
        mock_settings.default_model = "test-model"
        mock_settings.fallback_model = "fallback-model"
        mock_settings.data_dir = tmp_path
        router = TaskRouter(cmap, profile, mock_settings)
        decision = asyncio.run(router.route("hello"))
        assert decision.strategy is not None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_task_router.py::TestStrategyOverrides -v`
Expected: FAIL — TaskRouter doesn't load overrides

**Step 3: Modify TaskRouter**

Changes in `src/core/task_router.py`:

1. **Modify `__init__`** (line 306) to load overrides:
```python
def __init__(
    self,
    competence_map: CompetenceMap,
    profile: PersonalProfile,
    settings: Any,
) -> None:
    self.competence_map = competence_map
    self.profile = profile
    self.default_model: str = getattr(settings, "default_model", "")
    self.fallback_model: str = getattr(settings, "fallback_model", "")
    self._strategy_overrides: dict = self._load_overrides(settings)
```

2. **Add `_load_overrides` method** (after `__init__`):
```python
@staticmethod
def _load_overrides(settings: Any) -> dict:
    """Load strategy overrides generated by StrategyOptimizer."""
    data_dir = getattr(settings, "data_dir", None)
    if data_dir is None:
        return {}
    override_path = Path(data_dir) / "profile" / "strategy_overrides.json"
    if not override_path.exists():
        return {}
    try:
        with open(override_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        logger.info("strategy_overrides_loaded", count=len(overrides))
        return overrides
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("strategy_overrides_load_error", error=str(exc))
        return {}
```

3. **Modify `route()` method** — after getting the default strategy (approx line 339), check for override:
```python
strategy = get_strategy(task_type)

# Apply strategy override if available
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
    logger.info("strategy_override_applied", key=override_key)
```

Add `import json` and `from pathlib import Path` at top if not already present.

**Step 4: Run all task router tests**

Run: `python -m pytest tests/test_task_router.py -v`
Expected: ALL tests PASS (existing 63 + new 3)

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: TaskRouter loads strategy overrides"
```

---

### Task 6: CLI Integration — chat + ask + outcomes command

**Files:**
- Modify: `src/interfaces/cli.py`
- No new tests (CLI tests are integration-level, existing test suite covers)

**Step 1: Integrate OutcomeTracker in chat loop**

In `src/interfaces/cli.py`, in the `chat()` command:

1. After the router initialization (~line 75), create the tracker:
```python
from src.flywheel.outcome_tracker import OutcomeTracker
from uuid import uuid4

tracker = OutcomeTracker(settings.datalake_path, str(uuid4()))
```

2. After each response is printed and collector captures (~line 121), add:
```python
tracker.record_turn(
    query=user_input,
    response=response.content,
    decision=initial_decision,
    latency_ms=response.latency_ms,
)
```

Note: `initial_decision` is the decision from the router. For a more accurate tracking, route each message individually. But the current chat uses a single initial_decision for the system prompt. Use it as-is for now — the task_type from initial routing is "general" which is fine for the first iteration.

3. Before the `await collector.close()` line at exit (~line 122), add:
```python
final_outcome = tracker.close_session()
stats = tracker.get_session_stats()
if stats["total"] > 0:
    console.print(
        f"[dim]Outcomes: {stats['total']} tracked "
        f"({stats['accepted']} accepted, "
        f"{stats['rejected']} rejected)[/dim]"
    )
```

**Step 2: Integrate OutcomeTracker in ask command**

In the `ask()` async `run()` function, after the collector capture (~line 236):

```python
from src.flywheel.outcome_tracker import OutcomeTracker
from uuid import uuid4

tracker = OutcomeTracker(settings.datalake_path, str(uuid4()))
tracker.record_turn(
    query=prompt,
    response=response.content,
    decision=decision,
    latency_ms=response.latency_ms,
)
tracker.close_session()
```

**Step 3: Add `fabrik outcomes` CLI command**

Add new command group to `cli.py`:

```python
@app.command()
def outcomes(
    action: str = typer.Argument("stats", help="Action: show or stats"),
    topic: str = typer.Option(None, "--topic", "-t", help="Filter by topic"),
    task_type: str = typer.Option(None, "--task-type", help="Filter by task type"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of records to show"),
):
    """View outcome tracking data."""
    import json
    from collections import defaultdict
    from pathlib import Path
    from rich.table import Table

    outcomes_dir = settings.datalake_path / "01-raw" / "outcomes"

    if not outcomes_dir.exists():
        console.print("[yellow]No outcomes recorded yet.[/yellow]")
        return

    # Read all outcomes
    records = []
    for filepath in sorted(outcomes_dir.glob("*_outcomes.jsonl")):
        with open(filepath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if topic and r.get("topic", "").lower() != topic.lower():
                        continue
                    if task_type and r.get("task_type", "").lower() != task_type.lower():
                        continue
                    records.append(r)
                except json.JSONDecodeError:
                    continue

    if not records:
        console.print("[yellow]No outcomes found with those filters.[/yellow]")
        return

    if action == "show":
        table = Table(title=f"Recent Outcomes (last {limit})")
        table.add_column("Time", style="dim")
        table.add_column("Task Type")
        table.add_column("Topic")
        table.add_column("Outcome")
        table.add_column("Reason", style="dim")

        for r in records[-limit:]:
            ts = r.get("timestamp", "")[:16]
            outcome = r.get("outcome", "")
            style = {"accepted": "green", "rejected": "red", "neutral": "dim"}.get(outcome, "")
            table.add_row(
                ts,
                r.get("task_type", ""),
                r.get("topic", "—"),
                f"[{style}]{outcome}[/{style}]",
                r.get("inference_reason", "")[:40],
            )
        console.print(table)

    elif action == "stats":
        # Aggregate by task_type + topic
        agg = defaultdict(lambda: {"accepted": 0, "rejected": 0, "neutral": 0, "total": 0})
        for r in records:
            tt = r.get("task_type", "general")
            tp = r.get("topic") or "—"
            key = (tt, tp)
            outcome = r.get("outcome", "neutral")
            agg[key][outcome] = agg[key].get(outcome, 0) + 1
            agg[key]["total"] += 1

        table = Table(title="Outcome Stats")
        table.add_column("Task Type")
        table.add_column("Topic")
        table.add_column("Total", justify="right")
        table.add_column("Accepted", justify="right", style="green")
        table.add_column("Rate", justify="right")

        total_all = 0
        accepted_all = 0

        for (tt, tp), counts in sorted(agg.items(), key=lambda x: -x[1]["total"]):
            total = counts["total"]
            accepted = counts["accepted"]
            non_neutral = accepted + counts["rejected"]
            rate = f"{accepted / non_neutral * 100:.1f}%" if non_neutral > 0 else "—"
            table.add_row(tt, tp, str(total), str(accepted), rate)
            total_all += total
            accepted_all += accepted

        console.print(table)

        non_neutral_all = sum(
            c["accepted"] + c["rejected"] for c in agg.values()
        )
        overall_rate = f"{accepted_all / non_neutral_all * 100:.1f}%" if non_neutral_all > 0 else "—"
        console.print(f"\n[bold]Overall:[/bold] {total_all} outcomes, {accepted_all} accepted ({overall_rate})")

        # Check for strategy overrides
        overrides_path = settings.data_dir / "profile" / "strategy_overrides.json"
        if overrides_path.exists():
            overrides = json.loads(overrides_path.read_text())
            console.print(f"[bold]Strategy overrides active:[/bold] {len(overrides)}")
    else:
        console.print(f"[red]Unknown action: {action}. Use 'show' or 'stats'.[/red]")
```

**Step 4: Extend `fabrik competence build` to run StrategyOptimizer**

In `cli.py`, find the `competence` command's `build` action. After the competence map is built and saved, add:

```python
# Run StrategyOptimizer
from src.core.strategy_optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(settings.datalake_path)
overrides_path = settings.data_dir / "profile" / "strategy_overrides.json"
override_count = optimizer.save_overrides(overrides_path)
if override_count > 0:
    console.print(f"[bold]Strategy overrides:[/bold] {override_count} generated")
else:
    console.print("[dim]No strategy overrides needed (all acceptance rates OK)[/dim]")
```

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL tests PASS (712 existing + ~22 new ≈ 734+)

**Step 6: Commit**

```bash
git add src/interfaces/cli.py
git commit -m "FEAT: Integrate OutcomeTracker in CLI"
```

---

### Task 7: Final Integration Test + Cleanup

**Step 1: Run the full test suite one final time**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Manual smoke test**

```bash
# Test outcomes command with no data
fabrik outcomes stats

# Test ask (should record neutral outcome)
fabrik ask "what is a decorator in python"

# Verify outcome was recorded
ls -la "$(fabrik status 2>/dev/null | grep -o '/media/[^"]*datalake')/01-raw/outcomes/"

# Test chat (2-3 turns, then exit)
fabrik chat
# > explain async in python
# > now how does docker networking work  (topic change → accepted)
# > exit

# Check outcomes
fabrik outcomes show
fabrik outcomes stats
```

**Step 3: Run competence build to test full loop**

```bash
fabrik competence build
# Should show strategy overrides line (0 if not enough data yet)
```

**Step 4: Final commit with all changes**

```bash
git add -A
git status  # Review — no sensitive files
git commit -m "FEAT: Complete Outcome Tracking system"
```
