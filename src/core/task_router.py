"""Adaptive Task Router for Fabrik-Codek (FC-37).

Classifies user queries and produces routing decisions that adapt
retrieval strategy, model selection, and system prompt based on
task type and user competence level.

Includes a learned TF-IDF classifier (FC-47) that trains on accepted
outcomes to improve classification accuracy over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.core.competence_model import CompetenceMap
from src.core.personal_profile import PersonalProfile

# Import guard for sklearn — optional dependency for LearnedClassifier
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Learned classifier (FC-47)
# ---------------------------------------------------------------------------


class LearnedClassifier:
    """TF-IDF based query classifier learned from accepted outcomes.

    Trains on (query, task_type) pairs extracted from outcome files where
    outcome=accepted, confirming the classification was correct.
    Falls back gracefully when sklearn is unavailable or corpus is too small.
    """

    MIN_CORPUS_SIZE: int = 50
    CONFIDENCE_THRESHOLD: float = 0.3

    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._vectorizer: Any | None = None  # TfidfVectorizer when fitted
        self._centroids: dict[str, Any] | None = None  # ndarray per task_type
        self._corpus_size: int = 0
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the classifier has been fitted with sufficient data."""
        return self._fitted

    def build_corpus(self, datalake_path: Path) -> int:
        """Read outcomes from datalake, extract accepted (query, task_type) pairs.

        Scans all JSONL files in ``datalake_path/01-raw/outcomes/``,
        extracts entries where outcome=accepted, and saves the corpus.

        Returns the corpus size (number of entries saved).
        """
        outcomes_dir = datalake_path / "01-raw" / "outcomes"
        if not outcomes_dir.exists():
            logger.info("learned_classifier_no_outcomes_dir", path=str(outcomes_dir))
            return 0

        entries: list[dict[str, str]] = []
        for jsonl_file in sorted(outcomes_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            record.get("outcome") == "accepted"
                            and record.get("query")
                            and record.get("task_type")
                        ):
                            entries.append(
                                {
                                    "query": record["query"],
                                    "task_type": record["task_type"],
                                }
                            )
            except OSError:
                continue

        if entries:
            self.save_corpus(entries)
        self._corpus_size = len(entries)
        logger.info("learned_classifier_corpus_built", size=len(entries))
        return len(entries)

    def save_corpus(self, entries: list[dict]) -> None:
        """Persist corpus entries to JSON file."""
        self._corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._corpus_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def load_corpus(self) -> list[dict]:
        """Load corpus entries from JSON file."""
        if not self._corpus_path.exists():
            return []
        try:
            with open(self._corpus_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def fit(self) -> bool:
        """Fit TF-IDF vectorizer and compute centroids per task_type.

        Returns True if enough data (>= MIN_CORPUS_SIZE) and sklearn
        is available. Returns False otherwise (graceful degradation).
        """
        if not _SKLEARN_AVAILABLE:
            logger.info("learned_classifier_sklearn_unavailable")
            return False

        corpus = self.load_corpus()
        self._corpus_size = len(corpus)

        if self._corpus_size < self.MIN_CORPUS_SIZE:
            logger.info(
                "learned_classifier_insufficient_data",
                corpus_size=self._corpus_size,
                min_required=self.MIN_CORPUS_SIZE,
            )
            self._fitted = False
            return False

        queries = [entry["query"] for entry in corpus]
        task_types = [entry["task_type"] for entry in corpus]

        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = self._vectorizer.fit_transform(queries)

        # Compute centroid per task_type (mean of TF-IDF vectors)
        self._centroids = {}
        unique_types = set(task_types)
        for tt in unique_types:
            indices = [i for i, t in enumerate(task_types) if t == tt]
            subset = tfidf_matrix[indices]
            centroid = np.asarray(subset.mean(axis=0)).flatten()
            self._centroids[tt] = centroid

        self._fitted = True
        logger.info(
            "learned_classifier_fitted",
            corpus_size=self._corpus_size,
            task_types=list(self._centroids.keys()),
        )
        return True

    def classify(self, query: str) -> tuple[str, float]:
        """Classify a query against learned centroids.

        Returns (task_type, confidence). Returns ("general", 0.0) if
        not fitted or sklearn unavailable.
        """
        if not self._fitted or not _SKLEARN_AVAILABLE:
            return ("general", 0.0)

        if not self._vectorizer or not self._centroids:
            return ("general", 0.0)

        query_vec = self._vectorizer.transform([query])
        query_arr = np.asarray(query_vec.toarray()).flatten()

        best_type = "general"
        best_score = 0.0

        for tt, centroid in self._centroids.items():
            # Cosine similarity between query vector and centroid
            sim = sklearn_cosine_similarity(
                query_arr.reshape(1, -1),
                centroid.reshape(1, -1),
            )[0, 0]
            if sim > best_score:
                best_score = float(sim)
                best_type = tt

        return (best_type, best_score)


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
    confidence_threshold: float = 0.7
    min_k: int = 1
    max_k: int = 8


@dataclass
class RoutingDecision:
    """Complete routing decision for a user query."""

    task_type: str
    topic: str | None
    competence_level: str
    model: str
    strategy: RetrievalStrategy
    system_prompt: str
    classification_method: str  # "learned", "keyword", or "llm"
    arm_id: str | None = None  # MAB arm ID (FC-42), None if no MAB


# ---------------------------------------------------------------------------
# Keyword classification
# ---------------------------------------------------------------------------

TASK_KEYWORDS: dict[str, list[str]] = {
    "debugging": [
        "error",
        "bug",
        "fix",
        "crash",
        "traceback",
        "exception",
        "fails",
        "broken",
        "issue",
        "wrong",
    ],
    "code_review": [
        "review",
        "refactor",
        "clean",
        "improve",
        "quality",
        "smell",
        "readable",
    ],
    "architecture": [
        "design",
        "pattern",
        "structure",
        "ddd",
        "hexagonal",
        "module",
        "architecture",
        "component",
    ],
    "explanation": [
        "explain",
        "how",
        "why",
        "what is",
        "difference",
        "compare",
        "understand",
        "meaning",
    ],
    "testing": [
        "test",
        "assert",
        "mock",
        "coverage",
        "pytest",
        "spec",
        "unit test",
        "integration test",
    ],
    "devops": [
        "deploy",
        "docker",
        "kubernetes",
        "ci/cd",
        "pipeline",
        "terraform",
        "nginx",
        "container",
    ],
    "ml_engineering": [
        "model",
        "training",
        "fine-tune",
        "embedding",
        "rag",
        "llm",
        "vector",
        "dataset",
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
    "debugging": {
        "graph_depth": 2,
        "vector_weight": 0.5,
        "graph_weight": 0.5,
        "confidence_threshold": 0.6,
        "min_k": 2,
        "max_k": 8,
    },
    "code_review": {
        "graph_depth": 1,
        "vector_weight": 0.7,
        "graph_weight": 0.3,
        "confidence_threshold": 0.7,
        "min_k": 1,
        "max_k": 5,
    },
    "architecture": {
        "graph_depth": 3,
        "vector_weight": 0.4,
        "graph_weight": 0.6,
        "confidence_threshold": 0.5,
        "min_k": 2,
        "max_k": 8,
    },
    "explanation": {
        "graph_depth": 2,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "confidence_threshold": 0.8,
        "min_k": 1,
        "max_k": 4,
    },
    "testing": {
        "graph_depth": 2,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "confidence_threshold": 0.7,
        "min_k": 1,
        "max_k": 5,
    },
    "devops": {
        "graph_depth": 1,
        "vector_weight": 0.7,
        "graph_weight": 0.3,
        "confidence_threshold": 0.6,
        "min_k": 1,
        "max_k": 6,
    },
    "ml_engineering": {
        "graph_depth": 2,
        "vector_weight": 0.5,
        "graph_weight": 0.5,
        "confidence_threshold": 0.6,
        "min_k": 2,
        "max_k": 8,
    },
    "general": {
        "graph_depth": 2,
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "confidence_threshold": 0.7,
        "min_k": 1,
        "max_k": 5,
    },
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
        confidence_threshold=params.get("confidence_threshold", 0.7),
        min_k=params.get("min_k", 1),
        max_k=params.get("max_k", 8),
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
# Helper: load learned classifier
# ---------------------------------------------------------------------------


def load_learned_classifier(settings: Any) -> LearnedClassifier | None:
    """Load a LearnedClassifier from data/profile/router_corpus.json if available.

    Returns a fitted classifier, or None if corpus doesn't exist or fit fails.
    """
    data_dir = getattr(settings, "data_dir", None)
    if data_dir is None:
        return None
    corpus_path = Path(data_dir) / "profile" / "router_corpus.json"
    if not corpus_path.exists():
        return None
    classifier = LearnedClassifier(corpus_path)
    if classifier.fit():
        return classifier
    return None


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
        mab: Any = None,  # Optional MABStrategyOptimizer
        learned_classifier: LearnedClassifier | None = None,  # FC-47
    ) -> None:
        self.competence_map = competence_map
        self.profile = profile
        self.default_model: str = getattr(settings, "default_model", "")
        self.fallback_model: str = getattr(settings, "fallback_model", "")
        self._strategy_overrides: dict = self._load_overrides(settings)
        self._mab = mab
        self._learned = learned_classifier

    @staticmethod
    def _load_overrides(settings: Any) -> dict:
        """Load strategy overrides from data/profile/strategy_overrides.json."""
        data_dir = getattr(settings, "data_dir", None)
        if data_dir is None:
            return {}
        override_path = Path(data_dir) / "profile" / "strategy_overrides.json"
        if not override_path.exists():
            return {}
        try:
            with open(override_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    async def route(self, query: str) -> RoutingDecision:
        """Classify query and produce a full routing decision.

        Classification chain (3 levels):
        1. Learned classifier (TF-IDF, if fitted and confidence >= threshold)
        2. Keyword matching (if learned unavailable or low confidence)
        3. LLM fallback (if keywords also low confidence)
        """
        # 1. Classify task type — 3-level chain: learned → keywords → LLM
        task_type: str = "general"
        classification_method: str = "keyword"
        classified = False

        # Level 1: Learned classifier (FC-47)
        if self._learned is not None and self._learned.is_fitted:
            learned_type, learned_conf = self._learned.classify(query)
            if learned_conf >= LearnedClassifier.CONFIDENCE_THRESHOLD:
                task_type = learned_type
                classification_method = "learned"
                classified = True
                logger.debug(
                    "learned_classifier_used",
                    task_type=learned_type,
                    confidence=learned_conf,
                )

        # Level 2: Keyword matching
        if not classified:
            task_type, confidence = classify_by_keywords(query)
            classification_method = "keyword"

            # Level 3: LLM fallback
            if confidence < KEYWORD_CONFIDENCE_THRESHOLD:
                task_type = await classify_by_llm(query)
                classification_method = "llm"

        # 2. Detect topic
        topic = detect_topic(query, self.competence_map)

        # 3. Get competence level
        competence_level = self.competence_map.get_level(topic) if topic else "Unknown"

        # 4. Select model (escalate if Novice/Unknown)
        model = get_model(competence_level, self.default_model, self.fallback_model)

        # 5. Get retrieval strategy
        arm_id: str | None = None

        if self._mab is not None:
            # MAB Thompson Sampling (FC-42)
            arm_id, strategy = self._mab.select_arm(task_type, topic)
        else:
            strategy = get_strategy(task_type)

            # Apply static strategy override if available (FC-38 fallback)
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
                    confidence_threshold=override.get(
                        "confidence_threshold", strategy.confidence_threshold
                    ),
                    min_k=override.get("min_k", strategy.min_k),
                    max_k=override.get("max_k", strategy.max_k),
                )

        # 6. Build adapted system prompt
        system_prompt = build_system_prompt(
            self.profile,
            self.competence_map,
            task_type,
        )

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

        logger.info(
            "task_routed",
            task_type=task_type,
            topic=topic,
            competence=competence_level,
            model=model,
            method=classification_method,
        )

        return decision
