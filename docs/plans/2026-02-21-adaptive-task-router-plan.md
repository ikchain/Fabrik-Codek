# Adaptive Task Router — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route user queries through intelligent classification (hybrid keyword + LLM) to select task-appropriate retrieval strategies, models, and system prompts based on learned competence.

**Architecture:** Standalone `TaskRouter` in `src/core/task_router.py` receives a query + CompetenceMap + PersonalProfile + Settings and returns a `RoutingDecision` with task_type, topic, model, strategy, and adapted system prompt. Integrates at CLI (`ask`, `chat`), API (`/ask`), MCP (`fabrik_ask`), and adds a CLI debug command (`fabrik router test`).

**Tech Stack:** Python 3.11+, dataclasses, structlog, Typer/Rich (CLI), existing LLMClient for fallback classification.

---

### Task 1: Data model — RetrievalStrategy + RoutingDecision

**Files:**
- Create: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing test**

In `tests/test_task_router.py`:

```python
"""Tests for the Adaptive Task Router."""

from src.core.task_router import RetrievalStrategy, RoutingDecision


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestDataModel -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.task_router'`

**Step 3: Write minimal implementation**

Create `src/core/task_router.py`:

```python
"""Adaptive Task Router for Fabrik-Codek.

Classifies user queries and produces routing decisions that adapt
retrieval strategy, model selection, and system prompt based on
task type and user competence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class RetrievalStrategy:
    """Retrieval parameters adapted per task type."""

    use_rag: bool = True
    use_graph: bool = True
    graph_depth: int = 2
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    fulltext_weight: float = 0.0


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestDataModel -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add TaskRouter data model"
```

---

### Task 2: Keyword classification

**Files:**
- Modify: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing tests**

Append to `tests/test_task_router.py`:

```python
from src.core.task_router import classify_by_keywords, TASK_KEYWORDS


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestKeywordClassification -v`
Expected: FAIL with `ImportError: cannot import name 'classify_by_keywords'`

**Step 3: Write minimal implementation**

Add to `src/core/task_router.py` after the dataclasses:

```python
TASK_KEYWORDS: dict[str, list[str]] = {
    "debugging": ["error", "bug", "fix", "crash", "traceback", "exception", "fails", "broken", "issue", "wrong"],
    "code_review": ["review", "refactor", "clean", "improve", "quality", "smell", "readable"],
    "architecture": ["design", "pattern", "structure", "ddd", "hexagonal", "module", "architecture", "component"],
    "explanation": ["explain", "how", "why", "what is", "difference", "compare", "understand", "meaning"],
    "testing": ["test", "assert", "mock", "coverage", "pytest", "spec", "unit test", "integration test"],
    "devops": ["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "terraform", "nginx", "container"],
    "ml_engineering": ["model", "training", "fine-tune", "embedding", "rag", "llm", "vector", "dataset"],
}

# Threshold: if top keyword score is below this, fall back to LLM classification.
KEYWORD_CONFIDENCE_THRESHOLD = 0.3


def classify_by_keywords(query: str) -> tuple[str, float]:
    """Classify a query by keyword matching against TASK_KEYWORDS.

    Returns (task_type, confidence). Returns ("general", 0.0) if
    no keywords match or the query is empty.
    """
    if not query or not query.strip():
        return ("general", 0.0)

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scores: dict[str, float] = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        matches = 0
        for kw in keywords:
            if " " in kw:
                # Multi-word keyword: check substring
                if kw in query_lower:
                    matches += 1
            else:
                if kw in query_words:
                    matches += 1
        if matches > 0:
            scores[task_type] = matches / len(keywords)

    if not scores:
        return ("general", 0.0)

    best_type = max(scores, key=scores.get)
    return (best_type, scores[best_type])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestKeywordClassification -v`
Expected: 12 PASS

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add keyword classification for task routing"
```

---

### Task 3: Topic detection from CompetenceMap

**Files:**
- Modify: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing tests**

Append to `tests/test_task_router.py`:

```python
from src.core.competence_model import CompetenceEntry, CompetenceMap
from src.core.task_router import detect_topic


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
        # "docker-compose" contains "docker"
        topic = detect_topic("docker-compose up", cmap)
        assert topic == "docker"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestTopicDetection -v`
Expected: FAIL with `ImportError: cannot import name 'detect_topic'`

**Step 3: Write minimal implementation**

Add to `src/core/task_router.py`:

```python
from src.core.competence_model import CompetenceMap


def detect_topic(query: str, competence_map: CompetenceMap) -> str | None:
    """Detect the most relevant topic from a query using CompetenceMap.

    Matches query tokens against topic names (case-insensitive).
    Returns the first matching topic by competence score order (highest first),
    or None if no topic matches.
    """
    if not query or not query.strip() or not competence_map.topics:
        return None

    query_lower = query.lower()
    # Split on whitespace and hyphens for compound words
    query_tokens = set(query_lower.replace("-", " ").split())

    # Topics are already sorted by score descending in CompetenceMap
    for entry in competence_map.topics:
        topic_lower = entry.topic.lower()
        # Check exact token match or substring in compound words
        if topic_lower in query_tokens or topic_lower in query_lower.replace(" ", "-").split("-"):
            return entry.topic

    return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestTopicDetection -v`
Expected: 8 PASS

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add topic detection from CompetenceMap"
```

---

### Task 4: Strategy selection + escalation logic

**Files:**
- Modify: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing tests**

Append to `tests/test_task_router.py`:

```python
from src.core.task_router import get_strategy, get_model, TASK_STRATEGIES, TASK_INSTRUCTIONS


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
        for task_type in ["debugging", "code_review", "architecture",
                          "explanation", "testing", "devops", "ml_engineering", "general"]:
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestStrategySelection tests/test_task_router.py::TestEscalationLogic -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/core/task_router.py`:

```python
TASK_STRATEGIES: dict[str, dict] = {
    "debugging":      {"graph_depth": 2, "vector_weight": 0.5, "graph_weight": 0.5},
    "code_review":    {"graph_depth": 1, "vector_weight": 0.7, "graph_weight": 0.3},
    "architecture":   {"graph_depth": 3, "vector_weight": 0.4, "graph_weight": 0.6},
    "explanation":    {"graph_depth": 2, "vector_weight": 0.6, "graph_weight": 0.4},
    "testing":        {"graph_depth": 2, "vector_weight": 0.6, "graph_weight": 0.4},
    "devops":         {"graph_depth": 1, "vector_weight": 0.7, "graph_weight": 0.3},
    "ml_engineering": {"graph_depth": 2, "vector_weight": 0.5, "graph_weight": 0.5},
    "general":        {"graph_depth": 2, "vector_weight": 0.6, "graph_weight": 0.4},
}

TASK_INSTRUCTIONS: dict[str, str] = {
    "debugging": "Focus on root cause analysis. Be direct about the fix.",
    "code_review": "Be specific about issues. Reference patterns and best practices.",
    "architecture": "Explain trade-offs. Consider scalability and maintainability.",
    "explanation": "Be clear and structured. Use examples when helpful.",
    "testing": "Focus on edge cases and coverage. Suggest test strategies.",
    "devops": "Be precise with commands and configs. Warn about destructive ops.",
    "ml_engineering": "Reference specific techniques. Distinguish theory from practice.",
    "general": "",
}

_DEFAULT_STRATEGY = TASK_STRATEGIES["general"]


def get_strategy(task_type: str) -> RetrievalStrategy:
    """Return the retrieval strategy for a task type."""
    params = TASK_STRATEGIES.get(task_type, _DEFAULT_STRATEGY)
    return RetrievalStrategy(
        graph_depth=params["graph_depth"],
        vector_weight=params["vector_weight"],
        graph_weight=params["graph_weight"],
    )


def get_model(
    competence_level: str,
    default_model: str,
    fallback_model: str,
) -> str:
    """Select model based on competence level. Escalate to fallback for Novice/Unknown."""
    if competence_level in ("Expert", "Competent"):
        return default_model
    return fallback_model
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestStrategySelection tests/test_task_router.py::TestEscalationLogic -v`
Expected: 18 PASS

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add strategy selection and escalation logic"
```

---

### Task 5: System prompt construction (3 layers)

**Files:**
- Modify: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing tests**

Append to `tests/test_task_router.py`:

```python
from src.core.personal_profile import PersonalProfile, StyleProfile
from src.core.task_router import build_system_prompt


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestSystemPromptConstruction -v`
Expected: FAIL with `ImportError: cannot import name 'build_system_prompt'`

**Step 3: Write minimal implementation**

Add to `src/core/task_router.py`:

```python
from src.core.personal_profile import PersonalProfile


def build_system_prompt(
    profile: PersonalProfile,
    competence_map: CompetenceMap,
    task_type: str,
) -> str:
    """Build a 3-layer system prompt: profile + competence + task instruction."""
    parts: list[str] = []

    # Layer 1: Personal profile
    profile_prompt = profile.to_system_prompt()
    if profile_prompt:
        parts.append(profile_prompt)

    # Layer 2: Competence fragment
    competence_fragment = competence_map.to_system_prompt_fragment()
    if competence_fragment:
        parts.append(competence_fragment)

    # Layer 3: Task-specific instruction
    instruction = TASK_INSTRUCTIONS.get(task_type, "")
    if instruction:
        parts.append(f"Task: {task_type}. {instruction}")

    return " ".join(parts)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestSystemPromptConstruction -v`
Expected: 8 PASS

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add 3-layer system prompt construction"
```

---

### Task 6: LLM fallback classification

**Files:**
- Modify: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing tests**

Append to `tests/test_task_router.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
from src.core.task_router import classify_by_llm, parse_llm_classification


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
        mock_get_client.return_value = mock_client

        import asyncio
        result = asyncio.run(classify_by_llm("why is this crashing"))
        assert result == "debugging"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestLLMFallback -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/core/task_router.py`:

```python
_VALID_TASK_TYPES = set(TASK_STRATEGIES.keys())

_CLASSIFICATION_PROMPT = (
    "Classify this query into ONE task type: debugging, code_review, architecture, "
    "explanation, testing, devops, ml_engineering, general.\n"
    "Query: \"{query}\"\n"
    "Answer with just the task type."
)


def _get_llm_client():
    """Get an LLMClient instance. Separated for testability."""
    from src.core import LLMClient
    return LLMClient()


def parse_llm_classification(response: str) -> str:
    """Extract a valid task type from LLM response text."""
    if not response:
        return "general"
    text = response.strip().lower()
    # Direct match
    if text in _VALID_TASK_TYPES:
        return text
    # Search for a valid type embedded in the response
    for task_type in _VALID_TASK_TYPES:
        if task_type in text:
            return task_type
    return "general"


async def classify_by_llm(query: str) -> str:
    """Classify a query using LLM as fallback. Returns task type string."""
    client = _get_llm_client()
    try:
        response = await client.generate(
            _CLASSIFICATION_PROMPT.format(query=query),
            temperature=0.0,
        )
        return parse_llm_classification(response.content)
    except Exception as exc:
        logger.warning("llm_classification_failed", error=str(exc))
        return "general"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestLLMFallback -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add LLM fallback classification"
```

---

### Task 7: TaskRouter.route() — full integration

**Files:**
- Modify: `src/core/task_router.py`
- Test: `tests/test_task_router.py`

**Step 1: Write the failing tests**

Append to `tests/test_task_router.py`:

```python
from src.core.task_router import TaskRouter


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
        import asyncio
        decision = asyncio.run(router.route("fix the error in my postgresql query"))
        assert decision.task_type == "debugging"
        assert decision.topic == "postgresql"
        assert decision.competence_level == "Expert"
        assert decision.model == "qwen2.5-coder:14b"
        assert decision.classification_method == "keyword"

    def test_route_unknown_topic_escalates(self):
        router = self._make_router()
        import asyncio
        decision = asyncio.run(router.route("deploy kubernetes cluster"))
        assert decision.topic == "kubernetes"
        assert decision.competence_level == "Unknown"
        assert decision.model == "qwen2.5-coder:32b"

    def test_route_no_topic_escalates(self):
        router = self._make_router()
        import asyncio
        decision = asyncio.run(router.route("fix this random error"))
        assert decision.topic is None
        assert decision.competence_level == "Unknown"
        assert decision.model == "qwen2.5-coder:32b"

    def test_route_system_prompt_has_three_layers(self):
        router = self._make_router()
        import asyncio
        decision = asyncio.run(router.route("fix the error in postgresql"))
        assert "software development" in decision.system_prompt.lower()
        assert "Expert in:" in decision.system_prompt
        assert "root cause" in decision.system_prompt.lower()

    def test_route_strategy_matches_task_type(self):
        router = self._make_router()
        import asyncio
        decision = asyncio.run(router.route("explain how docker works"))
        assert decision.task_type == "explanation"
        assert decision.topic == "docker"
        assert decision.strategy.graph_depth == 2

    @patch("src.core.task_router.classify_by_llm", new_callable=AsyncMock)
    def test_route_falls_back_to_llm(self, mock_llm):
        mock_llm.return_value = "architecture"
        router = self._make_router()
        import asyncio
        decision = asyncio.run(router.route("make it better"))
        # "make it better" has no keyword matches -> LLM fallback
        assert decision.task_type == "architecture"
        assert decision.classification_method == "llm"
        mock_llm.assert_called_once()

    def test_route_keyword_match_skips_llm(self):
        router = self._make_router()
        import asyncio
        with patch("src.core.task_router.classify_by_llm") as mock_llm:
            decision = asyncio.run(router.route("fix the error now"))
            mock_llm.assert_not_called()
            assert decision.classification_method == "keyword"

    def test_route_returns_routing_decision(self):
        router = self._make_router()
        import asyncio
        decision = asyncio.run(router.route("test with pytest mock"))
        assert isinstance(decision, RoutingDecision)
        assert isinstance(decision.strategy, RetrievalStrategy)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_task_router.py::TestTaskRouterIntegration -v`
Expected: FAIL with `ImportError: cannot import name 'TaskRouter'`

**Step 3: Write minimal implementation**

Add to `src/core/task_router.py`:

```python
class TaskRouter:
    """Adaptive Task Router — classifies queries and produces routing decisions."""

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

    async def route(self, query: str) -> RoutingDecision:
        """Classify query and produce a full routing decision."""
        # 1. Classify task type (hybrid: keywords first, LLM fallback)
        task_type, confidence = classify_by_keywords(query)
        classification_method = "keyword"

        if confidence < KEYWORD_CONFIDENCE_THRESHOLD:
            task_type = await classify_by_llm(query)
            classification_method = "llm"

        # 2. Detect topic
        topic = detect_topic(query, self.competence_map)

        # 3. Get competence level
        competence_level = (
            self.competence_map.get_level(topic) if topic else "Unknown"
        )

        # 4. Select model (escalate if Novice/Unknown)
        model = get_model(competence_level, self.default_model, self.fallback_model)

        # 5. Get retrieval strategy
        strategy = get_strategy(task_type)

        # 6. Build adapted system prompt
        system_prompt = build_system_prompt(
            self.profile, self.competence_map, task_type,
        )

        decision = RoutingDecision(
            task_type=task_type,
            topic=topic,
            competence_level=competence_level,
            model=model,
            strategy=strategy,
            system_prompt=system_prompt,
            classification_method=classification_method,
        )

        logger.info(
            "task_routed",
            task_type=task_type,
            topic=topic,
            competence=competence_level,
            model=model,
            method=classification_method,
        )

        return decision
```

Also add `from typing import Any` to the imports at the top.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_task_router.py::TestTaskRouterIntegration -v`
Expected: 8 PASS

**Step 5: Run ALL task_router tests**

Run: `pytest tests/test_task_router.py -v`
Expected: ALL PASS (~54 tests)

**Step 6: Commit**

```bash
git add src/core/task_router.py tests/test_task_router.py
git commit -m "FEAT: Add TaskRouter.route() with hybrid classification"
```

---

### Task 8: Integrate router into CLI — ask() and chat()

**Files:**
- Modify: `src/interfaces/cli.py` (lines 56-77 for chat, lines 150-209 for ask)

**Step 1: Write the failing test**

Append to `tests/test_task_router.py`:

```python
class TestCLIIntegrationSmoke:
    """Verify TaskRouter can be constructed with real settings (no Ollama needed)."""

    def test_router_with_real_settings(self):
        from src.config import settings
        profile = PersonalProfile()
        cmap = CompetenceMap()
        router = TaskRouter(cmap, profile, settings)
        assert router.default_model == settings.default_model
        assert router.fallback_model == settings.fallback_model
```

Run: `pytest tests/test_task_router.py::TestCLIIntegrationSmoke -v`
Expected: PASS (already works with current code)

**Step 2: Modify `cli.py` — `ask()` command (lines 150-225)**

Replace the current profile/competence injection and RAG logic in `ask()` with router integration. The key changes in the `run()` inner function:

```python
    async def run():
        nonlocal final_prompt, context

        # --- Router: classify and adapt ---
        from src.core.personal_profile import get_active_profile
        from src.core.competence_model import get_active_competence_map
        from src.core.task_router import TaskRouter

        active_profile = get_active_profile()
        competence_map = get_active_competence_map()
        router = TaskRouter(competence_map, active_profile, settings)
        decision = await router.route(prompt)

        console.print(
            f"[dim]Router: {decision.task_type} "
            f"({decision.classification_method}) "
            f"| topic={decision.topic or '—'} "
            f"| competence={decision.competence_level} "
            f"| model={decision.model}[/dim]"
        )

        # Inject hybrid RAG context with adapted strategy
        if use_graph:
            from src.knowledge.hybrid_rag import HybridRAGEngine
            async with HybridRAGEngine() as hybrid:
                results = await hybrid.retrieve(
                    prompt, limit=5,
                    graph_depth=decision.strategy.graph_depth,
                )
                if results:
                    final_prompt = await hybrid.query_with_context(
                        prompt, limit=5,
                        graph_depth=decision.strategy.graph_depth,
                    )
                    context = final_prompt
                    origins = {r.get("origin", "?") for r in results}
                    console.print(
                        f"[dim]Hybrid RAG: {len(results)} docs "
                        f"(origins: {', '.join(origins)})[/dim]\n"
                    )

        # Inject RAG context if requested (vector only)
        elif use_rag:
            from src.knowledge.rag import RAGEngine
            async with RAGEngine() as rag:
                rag_results = await rag.retrieve(prompt, limit=3)
                if rag_results:
                    rag_context = "\n---\n".join([
                        f"[{r['category']}] {r['text'][:500]}"
                        for r in rag_results
                    ])
                    final_prompt = f"""Context from knowledge base:
{rag_context}

---
Question: {prompt}

Answer using the context when relevant."""
                    context = rag_context
                    console.print(f"[dim]RAG: {len(rag_results)} relevant documents found[/dim]\n")

        async with LLMClient(model=model) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Processing...", total=None)
                response = await client.generate(
                    final_prompt,
                    system=decision.system_prompt,
                    model=decision.model,
                )

            console.print(Markdown(response.content))
            console.print(f"\n[dim]({response.tokens_used} tokens, {response.latency_ms:.0f}ms)[/dim]")

            collector = get_collector()
            await collector.capture_prompt_response(
                prompt=prompt,
                response=response.content,
                model=response.model,
                tokens=response.tokens_used,
                latency_ms=response.latency_ms,
                context=context,
            )
            await collector.close()
```

**Step 3: Modify `cli.py` — `chat()` command (lines 56-77)**

Replace the profile/competence injection block in `chat()`:

```python
            # --- Router: classify and adapt per message ---
            from src.core.personal_profile import get_active_profile
            from src.core.competence_model import get_active_competence_map
            from src.core.task_router import TaskRouter

            active_profile = get_active_profile()
            competence_map = get_active_competence_map()
            router = TaskRouter(competence_map, active_profile, settings)

            if not system:
                # Use router's system prompt for first message
                initial_decision = await router.route("general conversation")
                messages.insert(0, {"role": "system", "content": initial_decision.system_prompt})
```

Note: In chat, the system prompt is set once at start. For a more advanced integration (re-routing per message), that would be Outcome Trackingterritory.

**Step 4: Add `from src.config import settings` import in `ask()` inner function**

The `ask()` command currently does `from src.config import settings` only inside `graph` — add it at the top of `run()`.

**Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -x`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/interfaces/cli.py
git commit -m "FEAT: Integrate TaskRouter into CLI ask() and chat()"
```

---

### Task 9: Integrate router into API — /ask endpoint

**Files:**
- Modify: `src/interfaces/api.py` (lines 338-389)

**Step 1: Modify the `/ask` endpoint**

Add router initialization in the `lifespan` function (after hybrid RAG init, ~line 215):

```python
    # 6. Task Router
    from src.core.personal_profile import get_active_profile
    from src.core.competence_model import get_active_competence_map
    from src.core.task_router import TaskRouter

    profile = get_active_profile()
    competence_map = get_active_competence_map()
    app.state.router = TaskRouter(competence_map, profile, settings)
```

Modify the `/ask` endpoint (lines 338-389) to use the router:

```python
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    """Ask a question with optional RAG/graph context."""
    state = request.app.state
    await _ensure_ollama(state)

    prompt = req.prompt
    sources: list[dict] = []

    # Route the query
    router = getattr(state, "router", None)
    decision = await router.route(req.prompt) if router else None

    graph_depth = decision.strategy.graph_depth if decision else req.graph_depth

    # Hybrid RAG (vector + graph)
    if req.use_graph and getattr(state, "hybrid", None):
        results = await state.hybrid.retrieve(
            req.prompt, limit=5, graph_depth=graph_depth,
        )
        if results:
            context = "\n---\n".join(
                f"[{r.get('category', '?')}] {r['text'][:500]}" for r in results
            )
            prompt = (
                f"Context from knowledge base:\n{context}\n\n---\n"
                f"Question: {req.prompt}\n\nAnswer using the context when relevant."
            )
            sources = [
                {"source": r.get("source", ""), "category": r.get("category", ""), "origin": r.get("origin", "")}
                for r in results
            ]

    # Vector RAG only
    elif req.use_rag and getattr(state, "rag", None):
        rag_results = await state.rag.retrieve(req.prompt, limit=5)
        if rag_results:
            context = "\n---\n".join(
                f"[{r['category']}] {r['text'][:500]}" for r in rag_results
            )
            prompt = (
                f"Context from knowledge base:\n{context}\n\n---\n"
                f"Question: {req.prompt}\n\nAnswer using the context when relevant."
            )
            sources = [
                {"source": r.get("source", ""), "category": r.get("category", "")}
                for r in rag_results
            ]

    response = await state.llm.generate(
        prompt,
        model=decision.model if decision else req.model,
        system=decision.system_prompt if decision else None,
    )

    return AskResponse(
        answer=response.content,
        model=response.model,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
        sources=sources,
    )
```

**Step 2: Run full test suite**

Run: `pytest tests/ -x`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/interfaces/api.py
git commit -m "FEAT: Integrate TaskRouter into API /ask endpoint"
```

---

### Task 10: Integrate router into MCP — fabrik_ask tool

**Files:**
- Modify: `src/interfaces/mcp_server.py` (lines 252-337)

**Step 1: Add router initialization in MCP lifespan**

After the hybrid RAG init (~line 97), add:

```python
    # 6. Task Router
    from src.core.personal_profile import get_active_profile
    from src.core.competence_model import get_active_competence_map
    from src.core.task_router import TaskRouter

    profile = get_active_profile()
    competence_map = get_active_competence_map()
    _state["router"] = TaskRouter(competence_map, profile, settings)
```

**Step 2: Modify `fabrik_ask` tool**

In `fabrik_ask` (lines 252-337), add routing after the Ollama health check:

```python
    # Route the query
    router = _state.get("router")
    decision = await router.route(prompt) if router else None

    graph_depth = decision.strategy.graph_depth if decision else graph_depth
```

And modify the `generate` call at line 326:

```python
    try:
        response = await llm.generate(
            effective_prompt,
            model=decision.model if decision else model,
            system=decision.system_prompt if decision else None,
        )
    except Exception as exc:
        return json.dumps({"error": f"LLM generation failed: {exc}"})
```

Add routing info to the result dict:

```python
    result = {
        "answer": response.content,
        "model": response.model,
        "tokens_used": response.tokens_used,
        "latency_ms": response.latency_ms,
        "sources": sources,
        "routing": {
            "task_type": decision.task_type,
            "topic": decision.topic,
            "competence": decision.competence_level,
            "method": decision.classification_method,
        } if decision else None,
    }
```

**Step 3: Run full test suite**

Run: `pytest tests/ -x`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/interfaces/mcp_server.py
git commit -m "FEAT: Integrate TaskRouter into MCP fabrik_ask tool"
```

---

### Task 11: CLI debug command — `fabrik router test`

**Files:**
- Modify: `src/interfaces/cli.py`

**Step 1: Add the `router` command**

After the `competence` command (~line 1295), add:

```python
@app.command()
def router(
    action: str = typer.Argument("test", help="Action: test"),
    query: str = typer.Option(..., "--query", "-q", help="Query to classify"),
):
    """Test the adaptive task router — inspect classification without executing."""
    from src.config import settings
    from src.core.competence_model import get_active_competence_map
    from src.core.personal_profile import get_active_profile
    from src.core.task_router import TaskRouter

    active_profile = get_active_profile()
    competence_map = get_active_competence_map()
    task_router = TaskRouter(competence_map, active_profile, settings)

    async def run():
        decision = await task_router.route(query)

        table = Table(title="Routing Decision")
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="green")
        table.add_row("Query", query)
        table.add_row("Task Type", f"{decision.task_type} ({decision.classification_method})")
        table.add_row(
            "Topic",
            f"{decision.topic} ({decision.competence_level}, score={competence_map.get_score(decision.topic):.2f})"
            if decision.topic else "— (Unknown)",
        )
        table.add_row("Model", decision.model)
        table.add_row(
            "Strategy",
            f"graph_depth={decision.strategy.graph_depth}, "
            f"vector={decision.strategy.vector_weight}, "
            f"graph={decision.strategy.graph_weight}",
        )
        console.print(table)

        console.print(f"\n[dim]System prompt:[/dim]")
        console.print(f"[italic]{decision.system_prompt}[/italic]")

    if action == "test":
        async_run(run())
    else:
        console.print("[yellow]Usage:[/yellow] fabrik router test -q 'your query'")
```

**Step 2: Run full test suite**

Run: `pytest tests/ -x`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/interfaces/cli.py
git commit -m "FEAT: Add 'fabrik router test' CLI debug command"
```

---

### Task 12: Final verification + CHANGELOG

**Step 1: Run ALL tests**

Run: `pytest tests/ -v`
Expected: ALL PASS (~700+ tests: 648 existing + ~60 new)

**Step 2: Update CHANGELOG.md**

Add under `## [Unreleased]` → `### Added`:

```markdown
- **Adaptive Task Router** — Intelligent query classification and routing  - Hybrid classification: keyword matching + LLM fallback
  - Topic detection from CompetenceMap with auto-escalation
  - Per-task retrieval strategies (graph depth, vector/graph weights)
  - 3-layer system prompt: profile + competence + task-specific instructions
  - Model escalation: Novice/Unknown topics use fallback model automatically
  - CLI: `fabrik router test -q "query"` for classification debugging
  - Integrated into CLI (`ask`, `chat`), API (`/ask`), and MCP (`fabrik_ask`)
  - ~60 new tests
```

Update `### Changed` test count.

**Step 3: Commit**

```bash
git add -A
git commit -m "DOCS: Update CHANGELOG with Adaptive Task Router"
```

**Step 4: Run one more full suite to be safe**

Run: `pytest tests/ -x`
Expected: ALL PASS
