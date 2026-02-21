"""Adaptive Task Router for Fabrik-Codek.

Classifies user queries and produces routing decisions that adapt
retrieval strategy, model selection, and system prompt based on
task type and user competence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.core.competence_model import CompetenceMap
from src.core.personal_profile import PersonalProfile

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Keyword classification
# ---------------------------------------------------------------------------

TASK_KEYWORDS: dict[str, list[str]] = {
    "debugging": [
        "error", "bug", "fix", "crash", "traceback",
        "exception", "fails", "broken", "issue", "wrong",
    ],
    "code_review": [
        "review", "refactor", "clean", "improve",
        "quality", "smell", "readable",
    ],
    "architecture": [
        "design", "pattern", "structure", "ddd",
        "hexagonal", "module", "architecture", "component",
    ],
    "explanation": [
        "explain", "how", "why", "what is",
        "difference", "compare", "understand", "meaning",
    ],
    "testing": [
        "test", "assert", "mock", "coverage",
        "pytest", "spec", "unit test", "integration test",
    ],
    "devops": [
        "deploy", "docker", "kubernetes", "ci/cd",
        "pipeline", "terraform", "nginx", "container",
    ],
    "ml_engineering": [
        "model", "training", "fine-tune", "embedding",
        "rag", "llm", "vector", "dataset",
    ],
}

# Threshold: if top keyword score is below this, fall back to LLM classification.
# Set low enough that any single keyword match avoids LLM fallback.
KEYWORD_CONFIDENCE_THRESHOLD = 0.1


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

    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    return (best_type, scores[best_type])


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------


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
        # Check exact token match or presence in hyphen-split tokens
        if topic_lower in query_tokens:
            return entry.topic

    return None


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Escalation logic
# ---------------------------------------------------------------------------


def get_model(
    competence_level: str,
    default_model: str,
    fallback_model: str,
) -> str:
    """Select model based on competence level.

    Expert/Competent use the default (smaller) model.
    Novice/Unknown/empty escalate to the fallback (larger) model.
    """
    if competence_level in ("Expert", "Competent"):
        return default_model
    return fallback_model


# ---------------------------------------------------------------------------
# System prompt construction (3 layers)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM fallback classification
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = set(TASK_STRATEGIES.keys())

_CLASSIFICATION_PROMPT = (
    "Classify this query into ONE task type: debugging, code_review, architecture, "
    "explanation, testing, devops, ml_engineering, general.\n"
    'Query: "{query}"\n'
    "Answer with just the task type."
)


def _get_llm_client():
    """Get an LLMClient instance. Separated for testability.

    Imports from the concrete module (not the package __init__) so that
    patches on ``src.core.LLMClient`` in CLI tests don't accidentally
    intercept the router's internal classification client.
    """
    from src.core.llm_client import LLMClient
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
        async with client:
            response = await client.generate(
                _CLASSIFICATION_PROMPT.format(query=query),
                temperature=0.0,
            )
        return parse_llm_classification(response.content)
    except Exception as exc:
        logger.warning("llm_classification_failed", error=str(exc))
        return "general"


# ---------------------------------------------------------------------------
# TaskRouter — full integration
# ---------------------------------------------------------------------------


class TaskRouter:
    """Adaptive Task Router -- classifies queries and produces routing decisions."""

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
