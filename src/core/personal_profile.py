"""Personal Profile data model for hyper-personalization.

Domain-agnostic profile built from datalake analysis.
Works for any profession: developer, lawyer, doctor, etc.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".java",
    ".go", ".rs", ".rb", ".cs", ".cpp", ".c", ".h",
    ".vue", ".svelte", ".kt", ".swift", ".sh", ".sql",
}

# Maps file extensions to human-readable language names for pattern generation.
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".kt": "Kotlin", ".swift": "Swift",
}

# Categories that indicate *what* the user does, not *what* they know.
# Filtered from topic weights (noise) but used for task type detection.
META_CATEGORIES = {
    "general", "error_fix", "explanation", "code_generation",
    "refactor", "decision",
}

# Task type consolidation: many datalake subcategories → canonical types.
# Value None means the category is filtered out entirely.
TASK_TYPE_CONSOLIDATION: dict[str, str | None] = {
    # Debugging
    "debugging": "debugging", "debugging-real": "debugging",
    "error_fix": "debugging",
    # Code review
    "code_review": "code_review", "code-review": "code_review",
    # Architecture
    "ddd": "architecture", "hexagonal": "architecture",
    "clean-architecture": "architecture",
    "architecture-decisions": "architecture", "api-design": "architecture",
    # ML / AI
    "ml": "ml_engineering", "ml-finetuning": "ml_engineering",
    "ml-rag": "ml_engineering", "ml-embeddings": "ml_engineering",
    "ml-agents": "ml_engineering", "ml-prompting": "ml_engineering",
    "ml-quantization": "ml_engineering", "ml-deployment": "ml_engineering",
    "ml-vectordb": "ml_engineering", "ml-evaluation": "ml_engineering",
    "agents": "ml_engineering", "langgraph": "ml_engineering",
    # DevOps / Infra
    "docker": "devops", "kubernetes": "devops", "terraform": "devops",
    "cicd": "devops", "nginx": "devops", "git": "devops",
    # Frontend
    "angular": "frontend", "react": "frontend", "nextjs": "frontend",
    "typescript": "frontend",
    # Backend
    "postgresql": "backend", "fastapi": "backend", "python": "backend",
    # Security
    "security": "security", "keycloak": "security",
    # Testing
    "testing": "testing",
    # Refactoring (all sub-variants)
    "refactoring": "refactoring",
    # Generation / Explanation
    "code_generation": "generation", "explanation": "generation",
    # Noise — filtered
    "general": None, "decision": None, "refactor": None,
}


@dataclass
class TopicWeight:
    """A topic with its relevance weight."""

    topic: str
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"topic": self.topic, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicWeight":
        """Deserialize from dictionary."""
        return cls(topic=data["topic"], weight=data.get("weight", 0.0))


@dataclass
class StyleProfile:
    """Communication style preferences."""

    formality: float = 0.5  # 0=casual, 1=formal
    verbosity: float = 0.5  # 0=concise, 1=verbose
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "formality": self.formality,
            "verbosity": self.verbosity,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StyleProfile":
        """Deserialize from dictionary."""
        return cls(
            formality=data.get("formality", 0.5),
            verbosity=data.get("verbosity", 0.5),
            language=data.get("language", "en"),
        )


@dataclass
class PersonalProfile:
    """Domain-agnostic personal profile built from datalake analysis.

    Captures the user's domain expertise, preferred topics,
    communication style, and detected task patterns.
    """

    domain: str = "unknown"
    domain_confidence: float = 0.0
    top_topics: list[TopicWeight] = field(default_factory=list)
    style: StyleProfile = field(default_factory=StyleProfile)
    patterns: list[str] = field(default_factory=list)
    task_types_detected: list[str] = field(default_factory=list)
    total_entries: int = 0
    built_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "top_topics": [t.to_dict() for t in self.top_topics],
            "style": self.style.to_dict(),
            "patterns": self.patterns,
            "task_types_detected": self.task_types_detected,
            "total_entries": self.total_entries,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonalProfile":
        """Deserialize profile from dictionary."""
        return cls(
            domain=data.get("domain", "unknown"),
            domain_confidence=data.get("domain_confidence", 0.0),
            top_topics=[
                TopicWeight.from_dict(t) for t in data.get("top_topics", [])
            ],
            style=StyleProfile.from_dict(data.get("style", {})),
            patterns=data.get("patterns", []),
            task_types_detected=data.get("task_types_detected", []),
            total_entries=data.get("total_entries", 0),
            built_at=data.get("built_at", datetime.now().isoformat()),
        )

    def to_system_prompt(self) -> str:
        """Generate a concise, behavioral system prompt from the profile.

        Produces direct instructions the model can follow (e.g.
        "Use Python with FastAPI") rather than descriptions about the
        user.  Keeps total length short — long prompts dilute a small
        model's attention.
        """
        if self.domain == "unknown" or self.domain_confidence < 0.1:
            return (
                "You are a general-purpose assistant. "
                "Adapt your responses to the user's needs."
            )

        domain_label = self.domain.replace("_", " ")
        parts = [f"You are assisting a {domain_label} professional."]

        # Patterns are already actionable instructions
        for pattern in self.patterns[:5]:
            parts.append(f"{pattern}.")

        if self.style.language != "en":
            parts.append(f"Respond in {self.style.language}.")

        return " ".join(parts)


def save_profile(profile: PersonalProfile, path: Path) -> None:
    """Save a profile to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
    logger.info("profile_saved", path=str(path), domain=profile.domain)


def load_profile(path: Path) -> PersonalProfile:
    """Load a profile from a JSON file.

    Returns an empty default profile if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        logger.debug("profile_not_found", path=str(path))
        return PersonalProfile()

    try:
        data = json.loads(path.read_text())
        profile = PersonalProfile.from_dict(data)
        logger.info("profile_loaded", path=str(path), domain=profile.domain)
        return profile
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("profile_load_error", path=str(path), error=str(exc))
        return PersonalProfile()


class DatalakeAnalyzer:
    """Analyze datalake contents to extract profile signals.

    Reads training pairs and auto-captures from the datalake directory
    to build a statistical picture of the user's domain expertise,
    preferred tools, active projects, and technology focus areas.
    """

    def __init__(self, datalake_path: Path) -> None:
        self.datalake_path = Path(datalake_path)

    def _read_jsonl(self, filepath: Path) -> list[dict]:
        """Read a JSONL file, handling both line-delimited and pretty-printed JSON.

        First tries standard line-by-line JSONL parsing.  If the first
        non-empty line does not parse to a ``dict`` (e.g. pretty-printed
        JSON objects spanning multiple lines), falls back to
        ``json.JSONDecoder.raw_decode`` which can handle concatenated
        JSON documents.
        """
        records: list[dict] = []
        try:
            text = filepath.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("read_jsonl_error", file=str(filepath), error=str(exc))
            return records

        if not text.strip():
            return records

        # Fast path: try first non-empty line as standard JSONL
        first_line = ""
        for raw_line in text.split("\n"):
            stripped = raw_line.strip()
            if stripped:
                first_line = stripped
                break

        use_raw_decode = False
        if first_line:
            try:
                obj = json.loads(first_line)
                if not isinstance(obj, dict):
                    use_raw_decode = True
            except json.JSONDecodeError:
                use_raw_decode = True

        if use_raw_decode:
            # Pretty-printed / concatenated JSON: use raw_decode
            decoder = json.JSONDecoder()
            idx = 0
            length = len(text)
            while idx < length:
                # Skip whitespace
                while idx < length and text[idx] in " \t\n\r":
                    idx += 1
                if idx >= length:
                    break
                try:
                    obj, end_idx = decoder.raw_decode(text, idx)
                    if isinstance(obj, dict):
                        records.append(obj)
                    idx = end_idx
                except json.JSONDecodeError:
                    # Skip to next line on error
                    next_nl = text.find("\n", idx)
                    if next_nl == -1:
                        break
                    idx = next_nl + 1
        else:
            # Standard JSONL: one dict per line
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    logger.debug("skipped_bad_line", file=str(filepath))

        return records

    def analyze_training_pairs(self) -> dict:
        """Scan training pairs and return category/tag statistics.

        Reads all ``*.jsonl`` files under ``02-processed/training-pairs/``
        and aggregates counts by category and tag.

        Returns:
            dict with keys: total_pairs, categories (Counter), tags (Counter)
        """
        tp_dir = self.datalake_path / "02-processed" / "training-pairs"
        categories: Counter = Counter()
        tags: Counter = Counter()
        total_pairs = 0

        if not tp_dir.exists():
            return {"total_pairs": 0, "categories": categories, "tags": tags}

        for jsonl_file in sorted(tp_dir.glob("*.jsonl")):
            records = self._read_jsonl(jsonl_file)
            total_pairs += len(records)
            for record in records:
                cat = record.get("category")
                if cat:
                    categories[cat] += 1
                for tag in record.get("tags", []):
                    tags[tag] += 1

        logger.info(
            "training_pairs_analyzed",
            total=total_pairs,
            categories=len(categories),
        )
        return {
            "total_pairs": total_pairs,
            "categories": categories,
            "tags": tags,
        }

    def analyze_auto_captures(self) -> dict:
        """Scan auto-captures and return project/tool/extension statistics.

        Reads all ``*auto-captures*.jsonl`` files under
        ``01-raw/code-changes/`` and aggregates counts by project,
        file extension, and tool used.

        Returns:
            dict with keys: total_captures, projects (Counter),
            file_extensions (Counter), tools (Counter)
        """
        ac_dir = self.datalake_path / "01-raw" / "code-changes"
        projects: Counter = Counter()
        file_extensions: Counter = Counter()
        tools: Counter = Counter()
        total_captures = 0

        if not ac_dir.exists():
            return {
                "total_captures": 0,
                "projects": projects,
                "file_extensions": file_extensions,
                "tools": tools,
            }

        for jsonl_file in sorted(ac_dir.glob("*auto-captures*.jsonl")):
            records = self._read_jsonl(jsonl_file)
            total_captures += len(records)
            for record in records:
                project = record.get("project")
                if project:
                    projects[project] += 1
                file_modified = record.get("file_modified", "")
                if file_modified:
                    ext = Path(file_modified).suffix
                    if ext:
                        file_extensions[ext] += 1
                tool = record.get("tool")
                if tool:
                    tools[tool] += 1

        logger.info(
            "auto_captures_analyzed",
            total=total_captures,
            projects=len(projects),
        )
        return {
            "total_captures": total_captures,
            "projects": projects,
            "file_extensions": file_extensions,
            "tools": tools,
        }


class ProfileBuilder:
    """Build a PersonalProfile from datalake analysis results.

    Combines DatalakeAnalyzer output to detect the user's domain,
    compute topic weights, identify task types and patterns.
    Domain-agnostic: software development is one heuristic among many.
    """

    def __init__(
        self, datalake_path: Path, graph_stats: dict | None = None
    ) -> None:
        self.datalake_path = Path(datalake_path)
        self.graph_stats = graph_stats or {}

    def build(self, output_path: Path | None = None) -> PersonalProfile:
        """Analyze the datalake and build a PersonalProfile.

        Args:
            output_path: If provided, save the profile JSON to this path.

        Returns:
            A fully populated PersonalProfile.
        """
        analyzer = DatalakeAnalyzer(self.datalake_path)
        tp_data = analyzer.analyze_training_pairs()
        ac_data = analyzer.analyze_auto_captures()

        total_entries = tp_data["total_pairs"] + ac_data["total_captures"]

        if total_entries == 0:
            profile = PersonalProfile(
                domain="unknown",
                domain_confidence=0.0,
                total_entries=0,
            )
            if output_path:
                save_profile(profile, output_path)
            return profile

        domain, confidence = self._detect_domain(tp_data, ac_data)
        categories = tp_data["categories"]
        top_topics = self._compute_topic_weights(categories)
        task_types = self._detect_task_types(categories)
        style = self._detect_style(tp_data)
        patterns = self._detect_patterns(ac_data, tp_data, domain)

        profile = PersonalProfile(
            domain=domain,
            domain_confidence=confidence,
            top_topics=top_topics,
            style=style,
            patterns=patterns,
            task_types_detected=task_types,
            total_entries=total_entries,
        )

        if output_path:
            save_profile(profile, output_path)

        logger.info(
            "profile_built",
            domain=domain,
            confidence=f"{confidence:.2f}",
            topics=len(top_topics),
            total_entries=total_entries,
        )
        return profile

    def _detect_domain(
        self, tp_data: dict, ac_data: dict
    ) -> tuple[str, float]:
        """Detect the user's primary domain from datalake signals.

        PRIMARY signal: file extensions from auto-captures (unambiguous).
        SECONDARY signal: category names mapped through TASK_TYPE_CONSOLIDATION
        (any category that maps to a known type is code-related).

        If code signals dominate, returns ``"software_development"``.
        Otherwise derives from the top category name.
        """
        categories: Counter = tp_data["categories"]
        file_extensions: Counter = ac_data.get("file_extensions", Counter())

        total_ext_count = sum(file_extensions.values()) if file_extensions else 0
        total_cat_count = sum(categories.values()) if categories else 0

        # PRIMARY: file extensions — most unambiguous signal
        code_ext_count = sum(
            count for ext, count in file_extensions.items()
            if ext in CODE_EXTENSIONS
        )
        ext_ratio = code_ext_count / total_ext_count if total_ext_count else 0.0

        # SECONDARY: categories that map through consolidation are code-related
        code_cat_count = sum(
            count for cat, count in categories.items()
            if cat in TASK_TYPE_CONSOLIDATION
        )
        cat_ratio = code_cat_count / total_cat_count if total_cat_count else 0.0

        # Combine: extensions are stronger signal (0.6 weight)
        if total_ext_count and total_cat_count:
            code_ratio = 0.6 * ext_ratio + 0.4 * cat_ratio
        elif total_ext_count:
            code_ratio = ext_ratio
        elif total_cat_count:
            code_ratio = cat_ratio
        else:
            return ("unknown", 0.0)

        if code_ratio > 0.3:
            return ("software_development", min(code_ratio, 1.0))

        # Non-code: derive domain from top category
        if categories:
            top_cat = categories.most_common(1)[0][0]
            domain = top_cat.replace("-", "_")
            confidence = categories[top_cat] / total_cat_count
            return (domain, confidence)

        return ("unknown", 0.0)

    def _compute_topic_weights(
        self, categories: Counter
    ) -> list[TopicWeight]:
        """Normalize category counts into TopicWeight list summing to ~1.0.

        Filters out META_CATEGORIES (noise like "general", "error_fix")
        before computing weights. Returns at most the top 10 real expertise
        topics sorted by weight descending.
        """
        if not categories:
            return []

        # Filter meta-categories — they describe *what* the user does,
        # not *what* they know about.
        filtered = Counter({
            cat: count for cat, count in categories.items()
            if cat not in META_CATEGORIES
        })

        total = sum(filtered.values())
        if total == 0:
            return []

        top_items = filtered.most_common(10)
        weights = [
            TopicWeight(topic=cat, weight=count / total)
            for cat, count in top_items
        ]
        return weights

    def _detect_task_types(self, categories: Counter) -> list[str]:
        """Consolidate raw datalake categories into canonical task types.

        Uses ``TASK_TYPE_CONSOLIDATION`` to map many sub-categories
        (e.g. ``ml-rag``, ``ml-embeddings``) to a single canonical type
        (``ml_engineering``).  Unknown categories are passed through as-is.
        Categories mapped to ``None`` are filtered out (noise).

        Returns a deduplicated, count-ordered list capped at 10 types.
        """
        if not categories:
            return []

        type_counts: Counter = Counter()
        for cat, count in categories.items():
            canonical = TASK_TYPE_CONSOLIDATION.get(cat, cat)
            if canonical is not None:
                type_counts[canonical] += count

        return [t for t, _ in type_counts.most_common(10)]

    def _detect_style(self, tp_data: dict) -> StyleProfile:
        """Heuristic style detection from training pair data.

        Uses average output length to estimate verbosity and a simple
        formality guess based on categories present.
        """
        categories: Counter = tp_data["categories"]
        total_pairs = tp_data["total_pairs"]

        if total_pairs == 0:
            return StyleProfile()

        # Formal categories hint at higher formality
        formal_cats = {"ddd", "terraform", "kubernetes", "civil_law", "labor_law"}
        formal_count = sum(
            count for cat, count in categories.items()
            if cat in formal_cats
        )
        formality = min(0.4 + (formal_count / total_pairs) * 0.6, 1.0)

        return StyleProfile(formality=formality, verbosity=0.5, language="en")

    def _detect_patterns(
        self, ac_data: dict, tp_data: dict, domain: str = "unknown"
    ) -> list[str]:
        """Extract actionable preferences from datalake signals.

        For software_development, generates behavioral instructions
        (e.g. "Use Python with FastAPI") instead of metadata
        (e.g. "Primary file types: .py").  For other domains, falls
        back to top-category descriptions.
        """
        categories: Counter = tp_data["categories"]
        file_extensions: Counter = ac_data.get("file_extensions", Counter())

        if domain == "software_development":
            return self._detect_code_patterns(categories, file_extensions)

        # Non-code domain: generic patterns from top categories
        patterns: list[str] = []
        if categories:
            real_cats = [
                cat for cat, _ in categories.most_common(5)
                if cat not in META_CATEGORIES
            ]
            if real_cats:
                patterns.append(
                    f"Key areas: {', '.join(real_cats[:3])}"
                )
        return patterns

    def _detect_code_patterns(
        self, categories: Counter, file_extensions: Counter
    ) -> list[str]:
        """Generate actionable coding preferences from datalake signals.

        Produces short imperative instructions that a model can follow
        directly (e.g. "Use Python with FastAPI and async/await").
        """
        patterns: list[str] = []

        # 1. Primary language from the single most common code extension
        if file_extensions:
            top_ext = file_extensions.most_common(1)[0][0]
            primary_lang = _EXT_TO_LANGUAGE.get(top_ext)
            if primary_lang:
                patterns.append(f"Use {primary_lang} for code examples")

        # 2. Web framework preference
        framework_map = {
            "fastapi": "FastAPI with async/await and Pydantic",
            "django": "Django",
            "flask": "Flask",
            "angular": "Angular",
            "react": "React with TypeScript",
            "nextjs": "Next.js",
        }
        for cat, framework in framework_map.items():
            if cat in categories:
                patterns.append(f"Prefer {framework}")
                break  # only the top framework

        # 3. Architecture methodology
        if "ddd" in categories or "hexagonal" in categories:
            patterns.append("Follow DDD and hexagonal architecture")
        elif "clean-architecture" in categories:
            patterns.append("Follow clean architecture patterns")

        # 4. Infrastructure / deployment
        infra = []
        if "docker" in categories:
            infra.append("Docker")
        if "kubernetes" in categories:
            infra.append("Kubernetes")
        if "terraform" in categories:
            infra.append("Terraform")
        if infra:
            patterns.append(f"Deploy with {', '.join(infra)}")

        # 5. AI/ML tooling
        ml_cats = {"langgraph", "agents", "ml", "ml-rag", "ml-agents"}
        if any(c in categories for c in ml_cats):
            patterns.append("Build AI agents and RAG pipelines")

        return patterns


# Simple cache to avoid re-reading profile on every LLM call
_profile_cache: dict[str, PersonalProfile] = {}


def get_active_profile(profile_path: Path | None = None) -> PersonalProfile:
    """Get the active profile, with simple caching.

    Loads and caches the profile so repeated LLM calls don't re-read
    from disk. Pass a specific path or use the default location.
    """
    from src.config import settings

    path = Path(profile_path) if profile_path else settings.data_dir / "profile" / "personal_profile.json"
    cache_key = str(path)

    if cache_key in _profile_cache:
        return _profile_cache[cache_key]

    profile = load_profile(path)
    _profile_cache[cache_key] = profile
    return profile
