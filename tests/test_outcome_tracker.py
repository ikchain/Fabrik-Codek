"""Tests for OutcomeRecord, infer_outcome, and OutcomeTracker (FC-38)."""

import json
from pathlib import Path

from src.core.task_router import RetrievalStrategy, RoutingDecision
from src.flywheel.outcome_tracker import (
    NEGATION_KEYWORDS,
    NEGATION_SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD,
    OutcomeRecord,
    OutcomeTracker,
    _starts_with_negation,
    infer_outcome,
    token_similarity,
)

# ---------------------------------------------------------------------------
# OutcomeRecord data model
# ---------------------------------------------------------------------------


class TestOutcomeRecord:
    def test_defaults(self):
        rec = OutcomeRecord(
            query="how to debug segfaults",
            response_summary="Use gdb with bt full",
            task_type="debugging",
            model="qwen2.5-coder:7b",
        )
        assert rec.query == "how to debug segfaults"
        assert rec.response_summary == "Use gdb with bt full"
        assert rec.task_type == "debugging"
        assert rec.model == "qwen2.5-coder:7b"
        # Auto-populated defaults
        assert rec.id  # uuid, not empty
        assert rec.timestamp  # iso string, not empty
        assert rec.outcome == "neutral"
        assert rec.topic is None
        assert rec.competence_level == "Unknown"
        assert rec.strategy == {}
        assert rec.inference_reason == ""
        assert rec.latency_ms == 0.0
        assert rec.session_id == ""

    def test_to_dict(self):
        rec = OutcomeRecord(
            query="explain rag",
            response_summary="RAG combines retrieval...",
            task_type="explanation",
            model="test-model",
        )
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["query"] == "explain rag"
        assert d["task_type"] == "explanation"
        assert "id" in d
        assert "timestamp" in d

    def test_all_fields_in_dict(self):
        rec = OutcomeRecord(
            query="q",
            response_summary="r",
            task_type="general",
            model="m",
            topic="docker",
            competence_level="Expert",
            strategy={"vector_weight": 0.6},
            outcome="accepted",
            inference_reason="topic change",
            latency_ms=123.4,
            session_id="sess-abc",
        )
        d = rec.to_dict()
        assert d["topic"] == "docker"
        assert d["competence_level"] == "Expert"
        assert d["strategy"] == {"vector_weight": 0.6}
        assert d["outcome"] == "accepted"
        assert d["inference_reason"] == "topic change"
        assert d["latency_ms"] == 123.4
        assert d["session_id"] == "sess-abc"


# ---------------------------------------------------------------------------
# token_similarity
# ---------------------------------------------------------------------------


class TestTokenSimilarity:
    def test_identical_queries(self):
        sim = token_similarity("how to fix docker", "how to fix docker")
        assert sim == 1.0

    def test_completely_different(self):
        sim = token_similarity("python async await", "kubernetes helm chart")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = token_similarity("fix docker error", "fix docker networking issue")
        # overlap: {"fix", "docker"} = 2, min(3, 4) = 3 -> 2/3 ~ 0.667
        assert 0.6 < sim < 0.7

    def test_empty_a(self):
        assert token_similarity("", "some query") == 0.0

    def test_empty_b(self):
        assert token_similarity("some query", "") == 0.0

    def test_both_empty(self):
        assert token_similarity("", "") == 0.0

    def test_whitespace_only(self):
        assert token_similarity("   ", "hello") == 0.0

    def test_case_insensitive(self):
        sim = token_similarity("Docker Fix", "docker fix")
        assert sim == 1.0


# ---------------------------------------------------------------------------
# _starts_with_negation
# ---------------------------------------------------------------------------


class TestStartsWithNegation:
    def test_no_negation(self):
        assert _starts_with_negation("how to deploy docker") is False

    def test_english_negation_no(self):
        assert _starts_with_negation("no, that's wrong") is True

    def test_english_negation_not(self):
        assert _starts_with_negation("not what I asked") is True

    def test_english_negation_wrong(self):
        assert _starts_with_negation("wrong answer, try again") is True

    def test_spanish_negation_no(self):
        assert _starts_with_negation("no, eso no es correcto") is True

    def test_spanish_negation_mal(self):
        assert _starts_with_negation("mal, eso esta mal") is True

    def test_spanish_negation_incorrecto(self):
        assert _starts_with_negation("incorrecto, intenta de nuevo") is True

    def test_empty_string(self):
        assert _starts_with_negation("") is False

    def test_negation_beyond_50_chars(self):
        # "wrong" after 50 chars should not be detected
        text = "a" * 51 + "wrong approach"
        assert _starts_with_negation(text) is False

    def test_negation_keywords_constant_exists(self):
        assert isinstance(NEGATION_KEYWORDS, (list, tuple, frozenset, set))
        assert len(NEGATION_KEYWORDS) > 0


# ---------------------------------------------------------------------------
# infer_outcome
# ---------------------------------------------------------------------------


class TestInferOutcome:
    def test_topic_change_is_accepted(self):
        outcome, reason = infer_outcome(
            "how to fix docker networking",
            "explain kubernetes ingress controllers",
        )
        assert outcome == "accepted"
        assert reason  # non-empty reason string

    def test_constructive_followup_is_accepted(self):
        outcome, reason = infer_outcome(
            "how to fix docker networking",
            "can you also show me the docker-compose config",
        )
        assert outcome == "accepted"

    def test_rephrased_question_is_rejected(self):
        """High similarity without negation implies reformulation -> rejected."""
        outcome, reason = infer_outcome(
            "how to fix docker networking error",
            "how to fix docker networking issue",
        )
        assert outcome == "rejected"
        assert "reformulat" in reason.lower() or "similar" in reason.lower()

    def test_explicit_negation_is_rejected(self):
        outcome, reason = infer_outcome(
            "how to fix docker networking",
            "no, that's not what I asked about docker networking",
        )
        assert outcome == "rejected"
        assert "negat" in reason.lower()

    def test_negation_unrelated_topic_not_rejected(self):
        """Negation keyword present, but topics are completely different -> accepted."""
        outcome, reason = infer_outcome(
            "how to deploy kubernetes pods",
            "no, tell me about python async patterns",
        )
        assert outcome == "accepted"

    def test_empty_queries(self):
        outcome, reason = infer_outcome("", "")
        assert outcome == "accepted"

    def test_empty_previous_query(self):
        outcome, reason = infer_outcome("", "how to fix docker")
        assert outcome == "accepted"

    def test_empty_new_query(self):
        outcome, reason = infer_outcome("how to fix docker", "")
        assert outcome == "accepted"

    def test_same_query_verbatim_is_rejected(self):
        outcome, reason = infer_outcome(
            "how to fix docker networking",
            "how to fix docker networking",
        )
        assert outcome == "rejected"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_similarity_threshold(self):
        assert SIMILARITY_THRESHOLD == 0.5

    def test_negation_similarity_threshold(self):
        assert NEGATION_SIMILARITY_THRESHOLD == 0.3

    def test_thresholds_relationship(self):
        """Negation threshold should be lower than similarity threshold."""
        assert NEGATION_SIMILARITY_THRESHOLD < SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# OutcomeTracker
# ---------------------------------------------------------------------------


def _make_decision(task_type="debugging", topic="postgresql"):
    return RoutingDecision(
        task_type=task_type,
        topic=topic,
        competence_level="Expert",
        model="test-model",
        strategy=RetrievalStrategy(),
        system_prompt="test",
        classification_method="keyword",
    )


class TestOutcomeTrackerFirstTurn:
    def test_first_turn_returns_none(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-1")
        result = tracker.record_turn(
            query="how to fix segfault",
            response="Use gdb with bt full",
            decision=_make_decision(),
            latency_ms=120.0,
        )
        assert result is None


class TestOutcomeTrackerSecondTurn:
    def test_second_turn_returns_outcome_of_first(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-1")
        # First turn — no outcome yet
        tracker.record_turn(
            query="how to fix docker networking",
            response="Check iptables rules",
            decision=_make_decision(task_type="debugging", topic="docker"),
            latency_ms=100.0,
        )
        # Second turn — completely different topic -> accepted
        result = tracker.record_turn(
            query="explain kubernetes ingress",
            response="An ingress controller routes traffic",
            decision=_make_decision(task_type="explanation", topic="kubernetes"),
            latency_ms=200.0,
        )
        assert result is not None
        assert isinstance(result, OutcomeRecord)
        assert result.query == "how to fix docker networking"
        assert result.task_type == "debugging"
        assert result.outcome in ("accepted", "rejected", "neutral")
        assert result.session_id == "sess-1"


class TestOutcomeTrackerCloseSession:
    def test_close_session_marks_neutral(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-2")
        tracker.record_turn(
            query="explain rag",
            response="RAG combines retrieval with generation",
            decision=_make_decision(task_type="explanation"),
            latency_ms=80.0,
        )
        result = tracker.close_session()
        assert result is not None
        assert result.outcome == "neutral"
        assert result.inference_reason == "session_close"

    def test_close_empty_session_returns_none(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-3")
        result = tracker.close_session()
        assert result is None


class TestOutcomeTrackerPersistence:
    def test_outcomes_persisted_to_jsonl(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-4")
        # Two turns -> one outcome persisted
        tracker.record_turn(
            query="fix docker error",
            response="Try restarting the daemon",
            decision=_make_decision(),
            latency_ms=50.0,
        )
        tracker.record_turn(
            query="explain kubernetes pods",
            response="A pod is the smallest deployable unit",
            decision=_make_decision(task_type="explanation"),
            latency_ms=60.0,
        )
        # Close session -> second outcome persisted
        tracker.close_session()

        # Find the JSONL file
        outcomes_dir = tmp_path / "01-raw" / "outcomes"
        assert outcomes_dir.exists()
        jsonl_files = list(outcomes_dir.glob("*_outcomes.jsonl"))
        assert len(jsonl_files) == 1

        lines = jsonl_files[0].read_text().strip().split("\n")
        assert len(lines) == 2

        record_1 = json.loads(lines[0])
        assert record_1["query"] == "fix docker error"
        assert record_1["session_id"] == "sess-4"
        assert "outcome" in record_1

        record_2 = json.loads(lines[1])
        assert record_2["query"] == "explain kubernetes pods"
        assert record_2["outcome"] == "neutral"


class TestOutcomeTrackerStats:
    def test_session_stats(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-5")

        # Turn 1
        tracker.record_turn(
            query="how to fix docker networking error",
            response="Check iptables",
            decision=_make_decision(),
            latency_ms=100.0,
        )
        # Turn 2 — different topic -> first turn accepted
        tracker.record_turn(
            query="explain kubernetes pods",
            response="A pod is...",
            decision=_make_decision(task_type="explanation"),
            latency_ms=100.0,
        )
        # Close -> second turn neutral
        tracker.close_session()

        stats = tracker.get_session_stats()
        assert stats["total"] == 2
        assert stats["accepted"] + stats["rejected"] + stats["neutral"] == 2
        assert isinstance(stats["accepted"], int)
        assert isinstance(stats["rejected"], int)
        assert isinstance(stats["neutral"], int)


class TestOutcomeTrackerTruncation:
    def test_response_summary_truncated(self, tmp_path):
        tracker = OutcomeTracker(datalake_path=tmp_path, session_id="sess-6")
        long_response = "A" * 300  # > 200 chars

        tracker.record_turn(
            query="explain rag",
            response=long_response,
            decision=_make_decision(),
            latency_ms=50.0,
        )
        # Close to get the record
        result = tracker.close_session()
        assert result is not None
        assert len(result.response_summary) <= 200


# ---------------------------------------------------------------------------
# arm_id storage (FC-42)
# ---------------------------------------------------------------------------


class TestArmIdStorage:
    """Verify arm_id from RoutingDecision is stored in outcome strategy dict."""

    def test_arm_id_stored_in_strategy(self, tmp_path: Path) -> None:
        """When RoutingDecision has arm_id, it appears in outcome strategy."""
        from dataclasses import dataclass

        @dataclass
        class MockStrategy:
            use_rag: bool = True
            use_graph: bool = True
            graph_depth: int = 2
            vector_weight: float = 0.6
            graph_weight: float = 0.4
            fulltext_weight: float = 0.0

        @dataclass
        class MockDecision:
            task_type: str = "debugging"
            topic: str | None = "postgresql"
            competence_level: str = "Competent"
            model: str = "test-model"
            strategy: MockStrategy = None
            arm_id: str | None = "graph_boost"

            def __post_init__(self):
                if self.strategy is None:
                    self.strategy = MockStrategy()

        tracker = OutcomeTracker(tmp_path, "test-session")
        decision = MockDecision()

        tracker.record_turn("query about postgresql", "response1", decision, 100.0)
        outcome = tracker.record_turn("different topic entirely", "response2", decision, 100.0)

        assert outcome is not None
        assert outcome.strategy.get("arm_id") == "graph_boost"

    def test_no_arm_id_when_not_present(self, tmp_path: Path) -> None:
        """When decision has no arm_id, strategy dict does not contain arm_id."""
        from dataclasses import dataclass

        @dataclass
        class MockStrategy:
            use_rag: bool = True
            use_graph: bool = True
            graph_depth: int = 2
            vector_weight: float = 0.6
            graph_weight: float = 0.4
            fulltext_weight: float = 0.0

        @dataclass
        class MockDecisionNoArm:
            task_type: str = "debugging"
            topic: str | None = "postgresql"
            competence_level: str = "Competent"
            model: str = "test-model"
            strategy: MockStrategy = None

            def __post_init__(self):
                if self.strategy is None:
                    self.strategy = MockStrategy()

        tracker = OutcomeTracker(tmp_path, "test-session")
        decision = MockDecisionNoArm()

        tracker.record_turn("query about postgresql", "response1", decision, 100.0)
        outcome = tracker.record_turn("different topic entirely", "response2", decision, 100.0)

        assert outcome is not None
        assert "arm_id" not in outcome.strategy
