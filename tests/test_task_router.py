"""Tests for the Adaptive Task Router (FC-37, FC-47)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.competence_model import CompetenceEntry, CompetenceMap
from src.core.personal_profile import PersonalProfile, StyleProfile
from src.core.task_router import (
    TASK_INSTRUCTIONS,
    TASK_STRATEGIES,
    LearnedClassifier,
    RetrievalStrategy,
    RoutingDecision,
    TaskRouter,
    build_system_prompt,
    classify_by_keywords,
    classify_by_llm,
    detect_topic,
    get_model,
    get_strategy,
    load_learned_classifier,
    parse_llm_classification,
)

# ---------------------------------------------------------------------------
# Task 1: Data model
# ---------------------------------------------------------------------------


class TestDataModel:
    def test_retrieval_strategy_defaults(self):
        s = RetrievalStrategy()
        assert s.use_rag is True
        assert s.use_graph is True
        assert s.graph_depth == 2
        assert s.vector_weight == 0.6
        assert s.graph_weight == 0.4
        assert s.fulltext_weight == 0.0
        assert s.confidence_threshold == 0.7
        assert s.min_k == 1
        assert s.max_k == 8

    def test_routing_decision_fields(self):
        s = RetrievalStrategy(graph_depth=3, vector_weight=0.4, graph_weight=0.6)
        d = RoutingDecision(
            task_type="debugging",
            topic="postgresql",
            competence_level="Expert",
            model="qwen2.5-coder:14b",
            strategy=s,
            system_prompt="You are assisting...",
            classification_method="keyword",
        )
        assert d.task_type == "debugging"
        assert d.topic == "postgresql"
        assert d.strategy.graph_depth == 3
        assert d.classification_method == "keyword"


# ---------------------------------------------------------------------------
# Task 2: Keyword classification
# ---------------------------------------------------------------------------


class TestKeywordClassification:
    def test_debugging_keywords(self):
        task_type, confidence = classify_by_keywords("I have an error in my postgres query")
        assert task_type == "debugging"
        assert confidence > 0.0

    def test_code_review_keywords(self):
        task_type, confidence = classify_by_keywords("please review and refactor this code")
        assert task_type == "code_review"
        assert confidence > 0.0

    def test_architecture_keywords(self):
        task_type, confidence = classify_by_keywords(
            "what design pattern should I use for this module"
        )
        assert task_type == "architecture"

    def test_explanation_keywords(self):
        task_type, confidence = classify_by_keywords("explain how async await works")
        assert task_type == "explanation"

    def test_testing_keywords(self):
        task_type, confidence = classify_by_keywords("write a test with pytest and mock")
        assert task_type == "testing"

    def test_devops_keywords(self):
        task_type, confidence = classify_by_keywords("how to deploy with docker and kubernetes")
        assert task_type == "devops"

    def test_ml_engineering_keywords(self):
        task_type, confidence = classify_by_keywords("fine-tune the embedding model for RAG")
        assert task_type == "ml_engineering"

    def test_no_match_returns_general(self):
        task_type, confidence = classify_by_keywords("hello world")
        assert task_type == "general"
        assert confidence == 0.0

    def test_case_insensitive(self):
        task_type, _ = classify_by_keywords("FIX this BUG please")
        assert task_type == "debugging"

    def test_multiple_matches_picks_highest(self):
        # "error" and "fix" both match debugging, "test" matches testing
        task_type, _ = classify_by_keywords("error fix test")
        assert task_type == "debugging"  # 2 matches vs 1

    def test_confidence_above_threshold(self):
        _, confidence = classify_by_keywords("error bug crash fix broken")
        assert confidence > 0.3

    def test_empty_query(self):
        task_type, confidence = classify_by_keywords("")
        assert task_type == "general"
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Task 3: Topic detection
# ---------------------------------------------------------------------------


class TestTopicDetection:
    def _make_competence_map(self) -> CompetenceMap:
        return CompetenceMap(
            topics=[
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
                CompetenceEntry(topic="typescript", score=0.6, level="Competent"),
                CompetenceEntry(topic="docker", score=0.45, level="Competent"),
                CompetenceEntry(topic="kubernetes", score=0.05, level="Unknown"),
                CompetenceEntry(topic="angular", score=0.3, level="Novice"),
            ]
        )

    def test_direct_match(self):
        cmap = self._make_competence_map()
        topic = detect_topic("my postgresql query is slow", cmap)
        assert topic == "postgresql"

    def test_case_insensitive_match(self):
        cmap = self._make_competence_map()
        topic = detect_topic("How to use Docker compose", cmap)
        assert topic == "docker"

    def test_no_match_returns_none(self):
        cmap = self._make_competence_map()
        topic = detect_topic("hello world", cmap)
        assert topic is None

    def test_empty_query_returns_none(self):
        cmap = self._make_competence_map()
        topic = detect_topic("", cmap)
        assert topic is None

    def test_empty_competence_map(self):
        cmap = CompetenceMap()
        topic = detect_topic("postgresql query", cmap)
        assert topic is None

    def test_first_match_by_score_order(self):
        cmap = self._make_competence_map()
        # Both "typescript" and "angular" could match via keyword
        topic = detect_topic("typescript angular component", cmap)
        # CompetenceMap topics are ordered by score, typescript (0.6) > angular (0.3)
        assert topic == "typescript"

    def test_partial_word_no_match(self):
        cmap = self._make_competence_map()
        # "post" is not "postgresql"
        topic = detect_topic("post a message", cmap)
        assert topic is None

    def test_topic_in_compound_word(self):
        cmap = self._make_competence_map()
        # "docker-compose" splits on hyphen to include "docker"
        topic = detect_topic("docker-compose up", cmap)
        assert topic == "docker"


# ---------------------------------------------------------------------------
# Task 4: Strategy selection + escalation logic
# ---------------------------------------------------------------------------


class TestStrategySelection:
    def test_debugging_strategy(self):
        s = get_strategy("debugging")
        assert s.graph_depth == 2
        assert s.vector_weight == 0.5
        assert s.graph_weight == 0.5

    def test_code_review_strategy(self):
        s = get_strategy("code_review")
        assert s.graph_depth == 1
        assert s.vector_weight == 0.7
        assert s.graph_weight == 0.3

    def test_architecture_strategy(self):
        s = get_strategy("architecture")
        assert s.graph_depth == 3
        assert s.vector_weight == 0.4
        assert s.graph_weight == 0.6

    def test_general_uses_defaults(self):
        s = get_strategy("general")
        assert s.graph_depth == 2
        assert s.vector_weight == 0.6
        assert s.graph_weight == 0.4

    def test_unknown_task_type_uses_defaults(self):
        s = get_strategy("nonexistent_type")
        assert s.graph_depth == 2  # default

    def test_all_task_types_have_strategies(self):
        for task_type in [
            "debugging",
            "code_review",
            "architecture",
            "explanation",
            "testing",
            "devops",
            "ml_engineering",
            "general",
        ]:
            s = get_strategy(task_type)
            assert isinstance(s, RetrievalStrategy)

    def test_each_strategy_has_valid_weights(self):
        for task_type, params in TASK_STRATEGIES.items():
            assert params["vector_weight"] + params["graph_weight"] <= 1.01

    def test_all_task_types_have_instructions(self):
        for task_type in TASK_STRATEGIES:
            assert task_type in TASK_INSTRUCTIONS

    def test_general_has_empty_instruction(self):
        assert TASK_INSTRUCTIONS["general"] == ""

    def test_debugging_instruction_content(self):
        assert "root cause" in TASK_INSTRUCTIONS["debugging"].lower()


class TestEscalationLogic:
    def test_expert_uses_default_model(self):
        model = get_model("Expert", "qwen2.5-coder:14b", "qwen2.5-coder:32b")
        assert model == "qwen2.5-coder:14b"

    def test_competent_uses_default_model(self):
        model = get_model("Competent", "qwen2.5-coder:14b", "qwen2.5-coder:32b")
        assert model == "qwen2.5-coder:14b"

    def test_novice_escalates_to_fallback(self):
        model = get_model("Novice", "qwen2.5-coder:14b", "qwen2.5-coder:32b")
        assert model == "qwen2.5-coder:32b"

    def test_unknown_escalates_to_fallback(self):
        model = get_model("Unknown", "qwen2.5-coder:14b", "qwen2.5-coder:32b")
        assert model == "qwen2.5-coder:32b"

    def test_empty_level_escalates(self):
        model = get_model("", "default", "fallback")
        assert model == "fallback"

    def test_expert_topic_none_uses_default(self):
        # When no topic detected, competence_level defaults to "Unknown"
        model = get_model("Unknown", "default", "fallback")
        assert model == "fallback"

    def test_custom_models(self):
        model = get_model("Expert", "llama3", "gpt-4")
        assert model == "llama3"

    def test_same_model_for_both(self):
        model = get_model("Novice", "single-model", "single-model")
        assert model == "single-model"


# ---------------------------------------------------------------------------
# Task 5: System prompt construction
# ---------------------------------------------------------------------------


class TestSystemPromptConstruction:
    def _make_profile(self) -> PersonalProfile:
        return PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
            patterns=["Use Python with FastAPI"],
            style=StyleProfile(formality=0.5, verbosity=0.5, language="en"),
        )

    def _make_competence_map(self) -> CompetenceMap:
        return CompetenceMap(
            topics=[
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
                CompetenceEntry(topic="docker", score=0.45, level="Competent"),
            ]
        )

    def test_includes_profile(self):
        prompt = build_system_prompt(
            self._make_profile(),
            self._make_competence_map(),
            "debugging",
        )
        assert "software development" in prompt.lower()

    def test_includes_competence(self):
        prompt = build_system_prompt(
            self._make_profile(),
            self._make_competence_map(),
            "debugging",
        )
        assert "Expert in: postgresql" in prompt

    def test_includes_task_instruction(self):
        prompt = build_system_prompt(
            self._make_profile(),
            self._make_competence_map(),
            "debugging",
        )
        assert "root cause" in prompt.lower()

    def test_general_no_task_instruction(self):
        prompt = build_system_prompt(
            self._make_profile(),
            self._make_competence_map(),
            "general",
        )
        # Should still have profile and competence, but no task instruction
        assert "software development" in prompt.lower()
        assert "Expert in: postgresql" in prompt

    def test_empty_profile(self):
        prompt = build_system_prompt(
            PersonalProfile(),
            self._make_competence_map(),
            "debugging",
        )
        assert "root cause" in prompt.lower()

    def test_empty_competence(self):
        prompt = build_system_prompt(
            self._make_profile(),
            CompetenceMap(),
            "code_review",
        )
        assert "software development" in prompt.lower()
        assert "specific about issues" in prompt.lower()

    def test_all_three_layers_present(self):
        prompt = build_system_prompt(
            self._make_profile(),
            self._make_competence_map(),
            "architecture",
        )
        # Layer 1: profile
        assert "software development" in prompt.lower()
        # Layer 2: competence
        assert "Expert in:" in prompt
        # Layer 3: task instruction
        assert "trade-offs" in prompt.lower()

    def test_unknown_task_type_no_crash(self):
        prompt = build_system_prompt(
            self._make_profile(),
            self._make_competence_map(),
            "nonexistent",
        )
        assert "software development" in prompt.lower()


# ---------------------------------------------------------------------------
# Task 6: LLM fallback
# ---------------------------------------------------------------------------


class TestLLMFallback:
    def test_parse_valid_task_type(self):
        assert parse_llm_classification("debugging") == "debugging"

    def test_parse_with_whitespace(self):
        assert parse_llm_classification("  code_review  \n") == "code_review"

    def test_parse_unknown_returns_general(self):
        assert parse_llm_classification("something_else") == "general"

    def test_parse_empty_returns_general(self):
        assert parse_llm_classification("") == "general"

    def test_parse_explanation_in_response(self):
        # LLM might return "The task type is: debugging"
        assert parse_llm_classification("The task type is: debugging") == "debugging"

    @patch("src.core.task_router._get_llm_client")
    def test_classify_by_llm_returns_task_type(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.content = "debugging"
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_get_client.return_value = mock_client

        result = asyncio.run(classify_by_llm("why is this crashing"))
        assert result == "debugging"


# ---------------------------------------------------------------------------
# Task 7: TaskRouter integration
# ---------------------------------------------------------------------------


class TestTaskRouterIntegration:
    def _make_router(self) -> TaskRouter:
        profile = PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
            patterns=["Use Python with FastAPI"],
        )
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
                CompetenceEntry(topic="docker", score=0.45, level="Competent"),
                CompetenceEntry(topic="kubernetes", score=0.05, level="Unknown"),
            ]
        )
        mock_settings = MagicMock()
        mock_settings.default_model = "qwen2.5-coder:14b"
        mock_settings.fallback_model = "qwen2.5-coder:32b"
        return TaskRouter(cmap, profile, mock_settings)

    def test_route_debugging_expert(self):
        router = self._make_router()
        decision = asyncio.run(router.route("fix the error in my postgresql query"))
        assert decision.task_type == "debugging"
        assert decision.topic == "postgresql"
        assert decision.competence_level == "Expert"
        assert decision.model == "qwen2.5-coder:14b"
        assert decision.classification_method == "keyword"

    def test_route_unknown_topic_escalates(self):
        router = self._make_router()
        decision = asyncio.run(router.route("deploy kubernetes cluster"))
        assert decision.topic == "kubernetes"
        assert decision.competence_level == "Unknown"
        assert decision.model == "qwen2.5-coder:32b"

    def test_route_no_topic_escalates(self):
        router = self._make_router()
        decision = asyncio.run(router.route("fix this random error"))
        assert decision.topic is None
        assert decision.competence_level == "Unknown"
        assert decision.model == "qwen2.5-coder:32b"

    def test_route_system_prompt_has_three_layers(self):
        router = self._make_router()
        decision = asyncio.run(router.route("fix the error in postgresql"))
        assert "software development" in decision.system_prompt.lower()
        assert "Expert in:" in decision.system_prompt
        assert "root cause" in decision.system_prompt.lower()

    def test_route_strategy_matches_task_type(self):
        router = self._make_router()
        decision = asyncio.run(router.route("explain how docker works"))
        assert decision.task_type == "explanation"
        assert decision.topic == "docker"
        assert decision.strategy.graph_depth == 2

    @patch("src.core.task_router.classify_by_llm", new_callable=AsyncMock)
    def test_route_falls_back_to_llm(self, mock_llm):
        mock_llm.return_value = "architecture"
        router = self._make_router()
        decision = asyncio.run(router.route("make it better"))
        # "make it better" has no keyword matches -> LLM fallback
        assert decision.task_type == "architecture"
        assert decision.classification_method == "llm"
        mock_llm.assert_called_once()

    def test_route_keyword_match_skips_llm(self):
        router = self._make_router()
        with patch("src.core.task_router.classify_by_llm") as mock_llm:
            decision = asyncio.run(router.route("fix the error now"))
            mock_llm.assert_not_called()
            assert decision.classification_method == "keyword"

    def test_route_returns_routing_decision(self):
        router = self._make_router()
        decision = asyncio.run(router.route("test with pytest mock"))
        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.strategy, RetrievalStrategy)


# ---------------------------------------------------------------------------
# CLI integration smoke test
# ---------------------------------------------------------------------------


class TestCLIIntegrationSmoke:
    """Verify TaskRouter can be constructed with real settings (no Ollama needed)."""

    def test_router_with_real_settings(self):
        from src.config import settings

        profile = PersonalProfile()
        cmap = CompetenceMap()
        router = TaskRouter(cmap, profile, settings)
        assert router.default_model == settings.default_model
        assert router.fallback_model == settings.fallback_model


# ---------------------------------------------------------------------------
# Task 8: Strategy overrides (FC-38)
# ---------------------------------------------------------------------------


class TestStrategyOverrides:
    """Verify TaskRouter loads and applies strategy overrides from JSON."""

    def _make_router_with_overrides(self, tmp_path, overrides):
        """Build a TaskRouter with strategy overrides written to tmp_path."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        override_file = profile_dir / "strategy_overrides.json"
        with open(override_file, "w", encoding="utf-8") as f:
            json.dump(overrides, f)

        profile = PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
            patterns=["Use Python with FastAPI"],
        )
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
                CompetenceEntry(topic="kubernetes", score=0.05, level="Unknown"),
            ]
        )
        mock_settings = MagicMock()
        mock_settings.default_model = "qwen2.5-coder:14b"
        mock_settings.fallback_model = "qwen2.5-coder:32b"
        mock_settings.data_dir = str(tmp_path)
        return TaskRouter(cmap, profile, mock_settings)

    def test_override_applied_to_matching_combo(self, tmp_path):
        """Override for debugging_postgresql changes strategy values."""
        overrides = {
            "debugging_postgresql": {
                "graph_depth": 5,
                "vector_weight": 0.3,
                "graph_weight": 0.7,
            },
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        decision = asyncio.run(router.route("fix the error in postgresql"))
        assert decision.task_type == "debugging"
        assert decision.topic == "postgresql"
        assert decision.strategy.graph_depth == 5
        assert decision.strategy.vector_weight == 0.3
        assert decision.strategy.graph_weight == 0.7

    def test_no_override_uses_default(self, tmp_path):
        """Override for debugging_postgresql, but query is about docker -> default."""
        overrides = {
            "debugging_postgresql": {
                "graph_depth": 5,
                "vector_weight": 0.3,
                "graph_weight": 0.7,
            },
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        # "explain" maps to explanation, not debugging, and no topic override
        decision = asyncio.run(router.route("explain how kubernetes works"))
        # Default explanation strategy: graph_depth=2, vector_weight=0.6, graph_weight=0.4
        assert decision.strategy.graph_depth == 2
        assert decision.strategy.vector_weight == 0.6
        assert decision.strategy.graph_weight == 0.4

    def test_no_overrides_file_works(self, tmp_path):
        """No strategy_overrides.json at all -> router works with defaults."""
        profile = PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
        )
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
            ]
        )
        mock_settings = MagicMock()
        mock_settings.default_model = "qwen2.5-coder:14b"
        mock_settings.fallback_model = "qwen2.5-coder:32b"
        mock_settings.data_dir = str(tmp_path)
        router = TaskRouter(cmap, profile, mock_settings)
        decision = asyncio.run(router.route("fix error in postgresql"))
        # Default debugging strategy
        assert decision.strategy.graph_depth == 2
        assert decision.strategy.vector_weight == 0.5
        assert decision.strategy.graph_weight == 0.5

    def test_override_with_fulltext_weight(self, tmp_path):
        """Override can include fulltext_weight."""
        overrides = {
            "debugging_postgresql": {
                "graph_depth": 3,
                "vector_weight": 0.4,
                "graph_weight": 0.4,
                "fulltext_weight": 0.2,
            },
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        decision = asyncio.run(router.route("fix the error in postgresql"))
        assert decision.strategy.fulltext_weight == 0.2

    def test_override_task_only_key(self, tmp_path):
        """Override keyed by task_type only (no topic) when topic is None."""
        overrides = {
            "debugging": {
                "graph_depth": 4,
                "vector_weight": 0.2,
                "graph_weight": 0.8,
            },
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        # "fix this random error" has no topic match
        decision = asyncio.run(router.route("fix this random error"))
        assert decision.topic is None
        assert decision.strategy.graph_depth == 4
        assert decision.strategy.vector_weight == 0.2
        assert decision.strategy.graph_weight == 0.8

    def test_data_dir_none_returns_empty_overrides(self):
        """When settings.data_dir is None, overrides should be empty."""
        mock_settings = MagicMock()
        mock_settings.data_dir = None
        overrides = TaskRouter._load_overrides(mock_settings)
        assert overrides == {}

    def test_malformed_json_returns_empty_overrides(self, tmp_path):
        """Malformed JSON file -> empty overrides, no crash."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        override_file = profile_dir / "strategy_overrides.json"
        override_file.write_text("{invalid json", encoding="utf-8")

        mock_settings = MagicMock()
        mock_settings.data_dir = str(tmp_path)
        overrides = TaskRouter._load_overrides(mock_settings)
        assert overrides == {}

    def test_partial_override_preserves_defaults(self, tmp_path):
        """Override with only graph_depth keeps other values from default strategy."""
        overrides = {
            "debugging_postgresql": {
                "graph_depth": 10,
            },
        }
        router = self._make_router_with_overrides(tmp_path, overrides)
        decision = asyncio.run(router.route("fix the error in postgresql"))
        assert decision.strategy.graph_depth == 10
        # Default debugging strategy values preserved
        assert decision.strategy.vector_weight == 0.5
        assert decision.strategy.graph_weight == 0.5
        assert decision.strategy.fulltext_weight == 0.0


# ---------------------------------------------------------------------------
# Task 9: MAB integration (FC-42)
# ---------------------------------------------------------------------------


class TestMABIntegration:
    """Tests for TaskRouter integration with MABStrategyOptimizer."""

    @pytest.fixture
    def mab_router(self, tmp_path):
        """Router with a MABStrategyOptimizer attached."""
        from src.core.strategy_optimizer import MABStrategyOptimizer

        competence_map = CompetenceMap(topics=[], built_at="")
        profile = PersonalProfile(domain="", patterns=[])
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
        assert decision.arm_id is not None
        assert decision.arm_id in {"default", "graph_boost", "deep_graph", "vector_focus"}

    @pytest.mark.asyncio
    async def test_route_without_mab_arm_id_is_none(self, tmp_path):
        """When no MAB is provided, arm_id is None (backward compat)."""
        competence_map = CompetenceMap(topics=[], built_at="")
        profile = PersonalProfile(domain="", patterns=[])
        settings = SimpleNamespace(
            default_model="test-model",
            fallback_model="fallback-model",
            data_dir=tmp_path,
        )
        router = TaskRouter(competence_map, profile, settings)
        decision = await router.route("fix this error")
        assert decision.arm_id is None


# ---------------------------------------------------------------------------
# Task 10: Confidence threshold fields (FC-46)
# ---------------------------------------------------------------------------


class TestConfidenceThresholdFields:
    """Verify RetrievalStrategy has confidence fields and per-task thresholds."""

    def test_strategy_has_confidence_fields(self):
        """RetrievalStrategy has confidence_threshold, min_k, max_k."""
        s = RetrievalStrategy()
        assert hasattr(s, "confidence_threshold")
        assert hasattr(s, "min_k")
        assert hasattr(s, "max_k")
        assert s.confidence_threshold == 0.7
        assert s.min_k == 1
        assert s.max_k == 8

    def test_per_task_thresholds(self):
        """Architecture has low threshold, explanation has high threshold."""
        arch = get_strategy("architecture")
        expl = get_strategy("explanation")
        assert arch.confidence_threshold == 0.5
        assert expl.confidence_threshold == 0.8
        assert arch.confidence_threshold < expl.confidence_threshold

    def test_debugging_strategy_confidence_fields(self):
        """Debugging strategy has specific confidence settings."""
        s = get_strategy("debugging")
        assert s.confidence_threshold == 0.6
        assert s.min_k == 2
        assert s.max_k == 8

    def test_code_review_strategy_confidence_fields(self):
        """Code review strategy has specific confidence settings."""
        s = get_strategy("code_review")
        assert s.confidence_threshold == 0.7
        assert s.min_k == 1
        assert s.max_k == 5

    def test_all_strategies_have_confidence_fields(self):
        """All task strategies produce strategies with confidence fields."""
        for task_type in TASK_STRATEGIES:
            s = get_strategy(task_type)
            assert isinstance(s.confidence_threshold, float)
            assert isinstance(s.min_k, int)
            assert isinstance(s.max_k, int)
            assert 0.0 < s.confidence_threshold <= 1.0
            assert s.min_k >= 1
            assert s.max_k >= s.min_k


# ---------------------------------------------------------------------------
# Task 11: Learned classifier (FC-47)
# ---------------------------------------------------------------------------


def _make_outcome_entries(task_type: str, count: int, prefix: str = "") -> list[dict]:
    """Generate synthetic outcome entries for testing."""
    templates = {
        "debugging": [
            "fix the connection error in database",
            "why does this crash on startup",
            "traceback in the authentication module",
            "bug in the payment processing logic",
            "exception thrown when parsing JSON input",
            "the server fails to respond after timeout",
            "broken link in the navigation component",
            "wrong output from the calculation function",
            "error handling missing in file upload",
            "crash when user submits empty form",
        ],
        "code_review": [
            "review the new authentication service",
            "refactor the database connection pool",
            "clean up the legacy API handlers",
            "improve the error handling patterns",
            "quality check on the test suite",
            "review pull request for user management",
            "refactor the notification service code",
            "improve readability of the config parser",
            "clean up unused imports and variables",
            "review the middleware implementation",
        ],
        "explanation": [
            "explain how async context managers work",
            "what is dependency injection pattern",
            "how does the event loop handle coroutines",
            "difference between composition and inheritance",
            "explain the observer pattern in detail",
            "what is the purpose of middleware layers",
            "how does connection pooling improve performance",
            "compare REST and GraphQL approaches",
            "explain SOLID principles with examples",
            "what is eventual consistency in databases",
        ],
        "testing": [
            "write unit tests for the user service",
            "add integration test for API endpoints",
            "mock the external payment gateway",
            "improve test coverage for edge cases",
            "test the error handling in data pipeline",
            "write pytest fixtures for database tests",
            "add assertion for response schema validation",
            "create test suite for authentication flow",
            "test concurrent access to shared resources",
            "add regression tests for the bugfix",
        ],
        "architecture": [
            "design a microservice for notifications",
            "structure the domain layer with DDD patterns",
            "architect the event sourcing pipeline",
            "design the hexagonal port and adapter pattern",
            "structure the module boundaries correctly",
            "design the caching strategy for the API",
            "architect a scalable message queue system",
            "design the database schema for multi-tenancy",
            "structure the monorepo for shared libraries",
            "design the API gateway routing rules",
        ],
        "devops": [
            "deploy the application to kubernetes cluster",
            "configure docker compose for local development",
            "setup terraform modules for AWS infrastructure",
            "configure nginx reverse proxy with SSL",
            "create CI/CD pipeline for automated testing",
            "deploy containerized services to production",
            "setup monitoring and alerting with prometheus",
            "configure load balancer for high availability",
            "automate database backup and restore process",
            "deploy helm charts to the staging environment",
        ],
        "ml_engineering": [
            "fine-tune the embedding model for domain data",
            "build RAG pipeline with vector similarity search",
            "train the classification model on labeled dataset",
            "optimize the LLM inference latency",
            "build feature engineering pipeline for training",
            "evaluate the model performance on test set",
            "implement vector database indexing strategy",
            "tune hyperparameters for the neural network",
            "build data augmentation pipeline for training",
            "deploy the model serving endpoint with batching",
        ],
    }

    queries = templates.get(task_type, [f"query about {task_type} topic {i}" for i in range(10)])
    entries = []
    for i in range(count):
        q = f"{prefix}{queries[i % len(queries)]} variant {i}"
        entries.append(
            {
                "query": q,
                "task_type": task_type,
                "outcome": "accepted",
                "timestamp": f"2026-02-27T00:{i:02d}:00",
            }
        )
    return entries


class TestLearnedClassifier:
    """Tests for the TF-IDF learned classifier (FC-47)."""

    def test_build_corpus_reads_accepted_outcomes(self, tmp_path):
        """build_corpus extracts only accepted outcomes from JSONL files."""
        outcomes_dir = tmp_path / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True)
        entries = [
            {"query": "fix error", "task_type": "debugging", "outcome": "accepted"},
            {"query": "hello world", "task_type": "general", "outcome": "neutral"},
            {"query": "review code", "task_type": "code_review", "outcome": "accepted"},
            {"query": "rejected query", "task_type": "testing", "outcome": "rejected"},
        ]
        with open(outcomes_dir / "test.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        size = lc.build_corpus(tmp_path)
        assert size == 2  # Only accepted entries

        corpus = lc.load_corpus()
        assert len(corpus) == 2
        assert corpus[0]["query"] == "fix error"
        assert corpus[0]["task_type"] == "debugging"
        assert corpus[1]["query"] == "review code"
        assert corpus[1]["task_type"] == "code_review"

    def test_build_corpus_no_outcomes_dir_returns_zero(self, tmp_path):
        """build_corpus returns 0 when outcomes directory does not exist."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        size = lc.build_corpus(tmp_path)
        assert size == 0

    def test_build_corpus_empty_dir_returns_zero(self, tmp_path):
        """build_corpus returns 0 when outcomes directory is empty."""
        outcomes_dir = tmp_path / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True)
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        size = lc.build_corpus(tmp_path)
        assert size == 0

    def test_build_corpus_skips_malformed_json_lines(self, tmp_path):
        """build_corpus skips lines with invalid JSON."""
        outcomes_dir = tmp_path / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True)
        with open(outcomes_dir / "test.jsonl", "w") as f:
            f.write('{"query": "fix error", "task_type": "debugging", "outcome": "accepted"}\n')
            f.write("{invalid json}\n")
            f.write('{"query": "review code", "task_type": "code_review", "outcome": "accepted"}\n')

        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        size = lc.build_corpus(tmp_path)
        assert size == 2

    def test_build_corpus_skips_entries_without_query(self, tmp_path):
        """build_corpus skips accepted entries missing query field."""
        outcomes_dir = tmp_path / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True)
        with open(outcomes_dir / "test.jsonl", "w") as f:
            f.write('{"task_type": "debugging", "outcome": "accepted"}\n')
            f.write('{"query": "", "task_type": "debugging", "outcome": "accepted"}\n')
            f.write('{"query": "valid", "task_type": "debugging", "outcome": "accepted"}\n')

        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        size = lc.build_corpus(tmp_path)
        assert size == 1  # Only the one with non-empty query

    def test_fit_returns_false_when_corpus_too_small(self, tmp_path):
        """fit returns False when corpus has fewer than MIN_CORPUS_SIZE entries."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        # Save a small corpus
        lc.save_corpus([{"query": f"q{i}", "task_type": "debugging"} for i in range(10)])
        assert lc.fit() is False
        assert lc.is_fitted is False

    def test_fit_returns_true_with_sufficient_corpus(self, tmp_path):
        """fit returns True when corpus has >= MIN_CORPUS_SIZE entries."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)

        # Generate 60 entries across multiple task types
        entries = []
        entries.extend(_make_outcome_entries("debugging", 20))
        entries.extend(_make_outcome_entries("code_review", 20))
        entries.extend(_make_outcome_entries("explanation", 20))
        lc.save_corpus(entries)

        assert lc.fit() is True
        assert lc.is_fitted is True

    def test_classify_returns_correct_task_type(self, tmp_path):
        """classify returns the correct task_type for a matching query."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)

        # Build corpus with distinct categories
        entries = []
        entries.extend(_make_outcome_entries("debugging", 25))
        entries.extend(_make_outcome_entries("code_review", 25))
        entries.extend(_make_outcome_entries("testing", 25))
        lc.save_corpus(entries)
        lc.fit()

        # Query that strongly matches debugging vocabulary
        task_type, confidence = lc.classify("fix the critical error in the database connection")
        assert task_type == "debugging"
        assert confidence > 0.0

    def test_classify_returns_general_when_not_fitted(self, tmp_path):
        """classify returns ('general', 0.0) when not fitted."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        assert lc.is_fitted is False

        task_type, confidence = lc.classify("fix error")
        assert task_type == "general"
        assert confidence == 0.0

    def test_classify_confidence_above_zero_for_match(self, tmp_path):
        """classify returns confidence > 0 for a query matching the corpus."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)

        entries = []
        entries.extend(_make_outcome_entries("debugging", 30))
        entries.extend(_make_outcome_entries("testing", 30))
        lc.save_corpus(entries)
        lc.fit()

        _, confidence = lc.classify("write unit tests for the service")
        assert confidence > 0.0

    def test_corpus_persistence_roundtrip(self, tmp_path):
        """save_corpus + load_corpus preserves data correctly."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)

        original = [
            {"query": "fix the bug", "task_type": "debugging"},
            {"query": "review the code", "task_type": "code_review"},
            {"query": "explain async", "task_type": "explanation"},
        ]
        lc.save_corpus(original)
        loaded = lc.load_corpus()
        assert loaded == original

    def test_load_corpus_returns_empty_when_no_file(self, tmp_path):
        """load_corpus returns [] when corpus file does not exist."""
        corpus_path = tmp_path / "nonexistent.json"
        lc = LearnedClassifier(corpus_path)
        assert lc.load_corpus() == []

    def test_load_corpus_returns_empty_on_malformed_json(self, tmp_path):
        """load_corpus returns [] when corpus file has invalid JSON."""
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text("{invalid", encoding="utf-8")
        lc = LearnedClassifier(corpus_path)
        assert lc.load_corpus() == []

    def test_load_corpus_returns_empty_on_non_list_json(self, tmp_path):
        """load_corpus returns [] when corpus is valid JSON but not a list."""
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text('{"key": "value"}', encoding="utf-8")
        lc = LearnedClassifier(corpus_path)
        assert lc.load_corpus() == []

    @patch("src.core.task_router._SKLEARN_AVAILABLE", False)
    def test_sklearn_unavailable_fit_returns_false(self, tmp_path):
        """fit returns False when sklearn is not available."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        entries = _make_outcome_entries("debugging", 60)
        lc.save_corpus(entries)
        assert lc.fit() is False

    @patch("src.core.task_router._SKLEARN_AVAILABLE", False)
    def test_sklearn_unavailable_classify_returns_general(self, tmp_path):
        """classify returns ('general', 0.0) when sklearn is unavailable."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        # Force _fitted to True to test the sklearn guard inside classify
        lc._fitted = True
        task_type, confidence = lc.classify("fix error")
        assert task_type == "general"
        assert confidence == 0.0

    def test_build_corpus_multiple_jsonl_files(self, tmp_path):
        """build_corpus reads from multiple JSONL files in outcomes directory."""
        outcomes_dir = tmp_path / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True)

        for day in ["2026-02-25", "2026-02-26"]:
            with open(outcomes_dir / f"{day}_outcomes.jsonl", "w") as f:
                f.write(
                    json.dumps(
                        {"query": f"q from {day}", "task_type": "debugging", "outcome": "accepted"}
                    )
                    + "\n"
                )

        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        size = lc.build_corpus(tmp_path)
        assert size == 2

    def test_classify_distinguishes_multiple_types(self, tmp_path):
        """Classifier can distinguish between several task types."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)

        entries = []
        entries.extend(_make_outcome_entries("debugging", 20))
        entries.extend(_make_outcome_entries("testing", 20))
        entries.extend(_make_outcome_entries("architecture", 20))
        lc.save_corpus(entries)
        lc.fit()

        # Testing query
        task_type, _ = lc.classify("write pytest unit tests with mock fixtures")
        assert task_type == "testing"

        # Architecture query
        task_type, _ = lc.classify("design the hexagonal architecture for this service")
        assert task_type == "architecture"

    def test_min_corpus_size_boundary(self, tmp_path):
        """fit returns False at 49 entries, True at 50."""
        corpus_path = tmp_path / "corpus.json"

        # 49 entries -> False
        lc = LearnedClassifier(corpus_path)
        entries = _make_outcome_entries("debugging", 49)
        lc.save_corpus(entries)
        assert lc.fit() is False

        # 50 entries -> True
        entries = _make_outcome_entries("debugging", 50)
        lc.save_corpus(entries)
        assert lc.fit() is True


# ---------------------------------------------------------------------------
# Task 12: Learned classifier integration with TaskRouter (FC-47)
# ---------------------------------------------------------------------------


class TestLearnedClassifierIntegration:
    """Integration tests: TaskRouter with LearnedClassifier."""

    def _build_fitted_classifier(self, tmp_path) -> LearnedClassifier:
        """Build and fit a LearnedClassifier with sufficient corpus."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        entries = []
        entries.extend(_make_outcome_entries("debugging", 25))
        entries.extend(_make_outcome_entries("code_review", 25))
        entries.extend(_make_outcome_entries("testing", 25))
        lc.save_corpus(entries)
        lc.fit()
        return lc

    def _make_router(self, tmp_path, learned=None) -> TaskRouter:
        """Build a TaskRouter with optional LearnedClassifier."""
        profile = PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
            patterns=["Use Python with FastAPI"],
        )
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
                CompetenceEntry(topic="docker", score=0.45, level="Competent"),
            ]
        )
        mock_settings = MagicMock()
        mock_settings.default_model = "qwen2.5-coder:14b"
        mock_settings.fallback_model = "qwen2.5-coder:32b"
        mock_settings.data_dir = str(tmp_path)
        return TaskRouter(cmap, profile, mock_settings, learned_classifier=learned)

    def test_router_uses_learned_method_when_confident(self, tmp_path):
        """TaskRouter uses 'learned' classification_method when classifier is confident."""
        learned = self._build_fitted_classifier(tmp_path)
        router = self._make_router(tmp_path, learned=learned)

        # Query with multiple corpus-matching terms to exceed CONFIDENCE_THRESHOLD
        decision = asyncio.run(router.route("review refactor clean improve quality readable"))
        assert decision.classification_method == "learned"
        assert decision.task_type == "code_review"

    def test_router_falls_back_to_keyword_when_no_classifier(self, tmp_path):
        """TaskRouter without learned_classifier uses keyword method (backward compat)."""
        router = self._make_router(tmp_path, learned=None)
        decision = asyncio.run(router.route("fix the error in my code"))
        assert decision.classification_method == "keyword"
        assert decision.task_type == "debugging"

    def test_router_falls_back_to_keyword_when_not_fitted(self, tmp_path):
        """Unfitted LearnedClassifier causes fallback to keyword classification."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        # Not fitted -> is_fitted is False
        router = self._make_router(tmp_path, learned=lc)
        decision = asyncio.run(router.route("fix the error in my code"))
        assert decision.classification_method == "keyword"

    def test_routing_decision_learned_method_in_dataclass(self, tmp_path):
        """RoutingDecision correctly stores 'learned' in classification_method."""
        learned = self._build_fitted_classifier(tmp_path)
        router = self._make_router(tmp_path, learned=learned)
        decision = asyncio.run(router.route("review refactor clean improve quality readable"))
        assert isinstance(decision, RoutingDecision)
        assert decision.classification_method == "learned"

    def test_router_falls_back_when_learned_low_confidence(self, tmp_path):
        """When learned classifier confidence is below threshold, falls back to keywords."""
        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)

        # Build corpus with very homogeneous data (all debugging)
        entries = _make_outcome_entries("debugging", 60)
        lc.save_corpus(entries)
        lc.fit()

        # Mock classify to return low confidence
        def low_confidence_classify(query):
            return ("debugging", 0.1)  # Below CONFIDENCE_THRESHOLD (0.3)

        lc.classify = low_confidence_classify

        router = self._make_router(tmp_path, learned=lc)
        decision = asyncio.run(router.route("fix the error now"))
        # Should fall back to keyword since learned confidence < 0.3
        assert decision.classification_method == "keyword"

    @patch("src.core.task_router.classify_by_llm", new_callable=AsyncMock)
    def test_full_fallback_chain_learned_to_keyword_to_llm(self, mock_llm, tmp_path):
        """Full chain: learned (unfitted) -> keywords (no match) -> LLM."""
        mock_llm.return_value = "architecture"

        corpus_path = tmp_path / "corpus.json"
        lc = LearnedClassifier(corpus_path)
        # Not fitted -> skip
        router = self._make_router(tmp_path, learned=lc)

        # Query with no keyword matches -> LLM fallback
        decision = asyncio.run(router.route("make it better"))
        assert decision.classification_method == "llm"
        assert decision.task_type == "architecture"
        mock_llm.assert_called_once()

    def test_router_backward_compat_no_learned_no_mab(self, tmp_path):
        """TaskRouter works exactly as before when no learned_classifier and no mab."""
        router = self._make_router(tmp_path, learned=None)
        decision = asyncio.run(router.route("fix the error in postgresql"))
        assert decision.task_type == "debugging"
        assert decision.topic == "postgresql"
        assert decision.classification_method == "keyword"
        assert decision.competence_level == "Expert"


# ---------------------------------------------------------------------------
# Task 13: load_learned_classifier helper + build_corpus guard (FC-47)
# ---------------------------------------------------------------------------


class TestLoadLearnedClassifier:
    """Tests for the load_learned_classifier helper function."""

    def test_returns_fitted_classifier_when_corpus_exists(self, tmp_path):
        """Returns a fitted LearnedClassifier when corpus file exists with enough data."""
        corpus_path = tmp_path / "profile" / "router_corpus.json"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        entries = _make_outcome_entries("debugging", 30)
        entries.extend(_make_outcome_entries("code_review", 30))
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(entries, f)

        settings = SimpleNamespace(data_dir=tmp_path)
        result = load_learned_classifier(settings)
        assert result is not None
        assert result.is_fitted

    def test_returns_none_when_no_corpus(self, tmp_path):
        """Returns None when corpus file doesn't exist."""
        settings = SimpleNamespace(data_dir=tmp_path)
        result = load_learned_classifier(settings)
        assert result is None

    def test_returns_none_when_data_dir_none(self):
        """Returns None when settings.data_dir is None."""
        settings = SimpleNamespace(data_dir=None)
        result = load_learned_classifier(settings)
        assert result is None

    def test_returns_none_when_corpus_too_small(self, tmp_path):
        """Returns None when corpus exists but has fewer than 50 entries."""
        corpus_path = tmp_path / "profile" / "router_corpus.json"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        entries = _make_outcome_entries("debugging", 10)
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(entries, f)

        settings = SimpleNamespace(data_dir=tmp_path)
        result = load_learned_classifier(settings)
        assert result is None


class TestBuildCorpusGuard:
    """Tests for build_corpus not overwriting valid corpus with empty list."""

    def test_build_corpus_does_not_overwrite_with_empty(self, tmp_path):
        """build_corpus with no accepted outcomes does not overwrite existing corpus."""
        corpus_path = tmp_path / "corpus.json"
        # Pre-populate corpus with valid data
        existing = [{"query": "fix error", "task_type": "debugging"}]
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(existing, f)

        # Create outcomes dir with only rejected outcomes
        datalake = tmp_path / "datalake"
        outcomes_dir = datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True)
        with open(outcomes_dir / "2026-01-01_outcomes.jsonl", "w") as f:
            f.write(json.dumps({"query": "q", "task_type": "debugging", "outcome": "rejected"}))

        lc = LearnedClassifier(corpus_path)
        result = lc.build_corpus(datalake)
        assert result == 0

        # Verify existing corpus was NOT overwritten
        with open(corpus_path, encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 1
        assert saved[0]["query"] == "fix error"
