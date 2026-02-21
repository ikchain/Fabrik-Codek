"""Tests for the Adaptive Task Router."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.competence_model import CompetenceEntry, CompetenceMap
from src.core.personal_profile import PersonalProfile, StyleProfile
from src.core.task_router import (
    KEYWORD_CONFIDENCE_THRESHOLD,
    TASK_INSTRUCTIONS,
    TASK_KEYWORDS,
    TASK_STRATEGIES,
    RetrievalStrategy,
    RoutingDecision,
    TaskRouter,
    build_system_prompt,
    classify_by_keywords,
    classify_by_llm,
    detect_topic,
    get_model,
    get_strategy,
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
        task_type, confidence = classify_by_keywords("what design pattern should I use for this module")
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
        return CompetenceMap(topics=[
            CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
            CompetenceEntry(topic="typescript", score=0.6, level="Competent"),
            CompetenceEntry(topic="docker", score=0.45, level="Competent"),
            CompetenceEntry(topic="kubernetes", score=0.05, level="Unknown"),
            CompetenceEntry(topic="angular", score=0.3, level="Novice"),
        ])

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
            "debugging", "code_review", "architecture",
            "explanation", "testing", "devops", "ml_engineering", "general",
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
        return CompetenceMap(topics=[
            CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
            CompetenceEntry(topic="docker", score=0.45, level="Competent"),
        ])

    def test_includes_profile(self):
        prompt = build_system_prompt(
            self._make_profile(), self._make_competence_map(), "debugging",
        )
        assert "software development" in prompt.lower()

    def test_includes_competence(self):
        prompt = build_system_prompt(
            self._make_profile(), self._make_competence_map(), "debugging",
        )
        assert "Expert in: postgresql" in prompt

    def test_includes_task_instruction(self):
        prompt = build_system_prompt(
            self._make_profile(), self._make_competence_map(), "debugging",
        )
        assert "root cause" in prompt.lower()

    def test_general_no_task_instruction(self):
        prompt = build_system_prompt(
            self._make_profile(), self._make_competence_map(), "general",
        )
        # Should still have profile and competence, but no task instruction
        assert "software development" in prompt.lower()
        assert "Expert in: postgresql" in prompt

    def test_empty_profile(self):
        prompt = build_system_prompt(
            PersonalProfile(), self._make_competence_map(), "debugging",
        )
        assert "root cause" in prompt.lower()

    def test_empty_competence(self):
        prompt = build_system_prompt(
            self._make_profile(), CompetenceMap(), "code_review",
        )
        assert "software development" in prompt.lower()
        assert "specific about issues" in prompt.lower()

    def test_all_three_layers_present(self):
        prompt = build_system_prompt(
            self._make_profile(), self._make_competence_map(), "architecture",
        )
        # Layer 1: profile
        assert "software development" in prompt.lower()
        # Layer 2: competence
        assert "Expert in:" in prompt
        # Layer 3: task instruction
        assert "trade-offs" in prompt.lower()

    def test_unknown_task_type_no_crash(self):
        prompt = build_system_prompt(
            self._make_profile(), self._make_competence_map(), "nonexistent",
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
        cmap = CompetenceMap(topics=[
            CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
            CompetenceEntry(topic="docker", score=0.45, level="Competent"),
            CompetenceEntry(topic="kubernetes", score=0.05, level="Unknown"),
        ])
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
