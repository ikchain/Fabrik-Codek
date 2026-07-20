"""Competence Model data structures for hyper-personalization.

Measures HOW DEEP a user's knowledge is per topic, complementing
the Personal Profile that identifies WHAT topics they work with.
Measures depth of knowledge to complement the Personal Profile.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# Score thresholds for competence level classification.
EXPERT_THRESHOLD = 0.8
COMPETENT_THRESHOLD = 0.4
NOVICE_THRESHOLD = 0.1

# Scoring ceilings and decay constants.
ENTRY_CEILING = 100
MAX_EDGES_CEILING = 100
RECENCY_HALF_LIFE_DAYS = 30

# Weight sets for composite score calculation (with outcome signal).
WEIGHTS_ALL: dict[str, float] = {"entry": 0.30, "density": 0.25, "recency": 0.20, "outcome": 0.25}
WEIGHTS_NO_GRAPH: dict[str, float] = {"entry": 0.40, "recency": 0.30, "outcome": 0.30}
WEIGHTS_NO_RECENCY: dict[str, float] = {"entry": 0.40, "density": 0.30, "outcome": 0.30}
WEIGHTS_ENTRY_ONLY: dict[str, float] = {"entry": 0.60, "outcome": 0.40}

# Fallback weight sets when no outcome data exists (original values).
WEIGHTS_ALL_NO_OUTCOME: dict[str, float] = {"entry": 0.5, "density": 0.3, "recency": 0.2}
WEIGHTS_NO_GRAPH_NO_OUTCOME: dict[str, float] = {"entry": 0.7, "recency": 0.3}
WEIGHTS_NO_RECENCY_NO_OUTCOME: dict[str, float] = {"entry": 0.6, "density": 0.4}
WEIGHTS_ENTRY_ONLY_NO_OUTCOME: dict[str, float] = {"entry": 1.0}


def _classify_level(score: float) -> str:
    """Classify a competence score into a human-readable level.

    Thresholds (inclusive lower bound):
        >= 0.8  -> Expert
        >= 0.4  -> Competent
        >= 0.1  -> Novice
        <  0.1  -> Unknown
    """
    if score >= EXPERT_THRESHOLD:
        return "Expert"
    if score >= COMPETENT_THRESHOLD:
        return "Competent"
    if score >= NOVICE_THRESHOLD:
        return "Novice"
    return "Unknown"


@dataclass
class CompetenceEntry:
    """A single topic competence measurement.

    Captures the score, classification level, and supporting metrics
    for one knowledge area.
    """

    topic: str
    score: float = 0.0
    level: str = "Unknown"
    entries: int = 0
    entity_density: float = 0.0
    recency_weight: float = 0.0
    last_activity: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary. Rounds float fields to 4 decimals."""
        return {
            "topic": self.topic,
            "score": round(self.score, 4),
            "level": self.level,
            "entries": self.entries,
            "entity_density": round(self.entity_density, 4),
            "recency_weight": round(self.recency_weight, 4),
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompetenceEntry:
        """Deserialize from dictionary."""
        return cls(
            topic=data["topic"],
            score=data.get("score", 0.0),
            level=data.get("level", "Unknown"),
            entries=data.get("entries", 0),
            entity_density=data.get("entity_density", 0.0),
            recency_weight=data.get("recency_weight", 0.0),
            last_activity=data.get("last_activity", ""),
        )


@dataclass
class CompetenceMap:
    """Aggregated competence map across all topics.

    Provides lookup methods for individual topic scores/levels
    and serialization for persistence.
    """

    topics: list[CompetenceEntry] = field(default_factory=list)
    built_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_topics: int = 0

    def __post_init__(self) -> None:
        if self.total_topics == 0 and self.topics:
            self.total_topics = len(self.topics)

    def get_entry(self, topic: str) -> CompetenceEntry | None:
        """Return the CompetenceEntry for a topic, or None if not found."""
        for entry in self.topics:
            if entry.topic == topic:
                return entry
        return None

    def get_level(self, topic: str) -> str:
        """Return the competence level for a topic, or 'Unknown' if not found."""
        entry = self.get_entry(topic)
        if entry is None:
            return "Unknown"
        return entry.level

    def get_score(self, topic: str) -> float:
        """Return the competence score for a topic, or 0.0 if not found."""
        entry = self.get_entry(topic)
        if entry is None:
            return 0.0
        return entry.score

    def experts(self) -> list[CompetenceEntry]:
        """Return all topics where level is Expert."""
        return [e for e in self.topics if e.level == "Expert"]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "topics": [t.to_dict() for t in self.topics],
            "built_at": self.built_at,
            "total_topics": self.total_topics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompetenceMap:
        """Deserialize from dictionary."""
        return cls(
            topics=[CompetenceEntry.from_dict(t) for t in data.get("topics", [])],
            built_at=data.get("built_at", datetime.now().isoformat()),
            total_topics=data.get("total_topics", 0),
        )

    def to_system_prompt_fragment(self) -> str:
        """Generate a concise competence fragment for injection into system prompts.

        Format: "Expert in: X, Y. Competent in: A, B."
        Only includes Expert and Competent levels, max 5 each.
        Returns empty string if no Expert or Competent topics exist.
        """
        expert_topics = [e.topic for e in self.topics if e.level == "Expert"][:5]
        competent_topics = [e.topic for e in self.topics if e.level == "Competent"][:5]

        if not expert_topics and not competent_topics:
            return ""

        parts: list[str] = []
        if expert_topics:
            parts.append(f"Expert in: {', '.join(expert_topics)}")
        if competent_topics:
            parts.append(f"Competent in: {', '.join(competent_topics)}")

        return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def compute_entry_score(entries: int) -> float:
    """Compute a logarithmic score from the number of training entries.

    Uses ``log(entries+1) / log(ENTRY_CEILING+1)`` capped at 1.0 so that
    any count >= ENTRY_CEILING yields the maximum score.

    Returns 0.0 for non-positive inputs.
    """
    if entries <= 0:
        return 0.0
    return min(math.log(entries + 1) / math.log(ENTRY_CEILING + 1), 1.0)


def compute_entity_density(edge_count: int, max_ceiling: int = MAX_EDGES_CEILING) -> float:
    """Compute a linear density score from graph edge count.

    Simply ``edge_count / max_ceiling`` capped at 1.0.

    Returns 0.0 for non-positive inputs.
    """
    if edge_count <= 0 or max_ceiling <= 0:
        return 0.0
    return min(edge_count / max_ceiling, 1.0)


def compute_recency_weight(
    last_activity_iso: str,
    reference_time: datetime | None = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Compute an exponential-decay recency weight.

    Formula: ``0.5 ** (days_elapsed / half_life_days)``

    Handles both timezone-naive and timezone-aware datetimes by
    normalising to UTC when needed.

    Returns 0.0 for empty or unparseable timestamps.
    Returns 1.0 for future timestamps (negative elapsed days).
    """
    if not last_activity_iso or not last_activity_iso.strip():
        return 0.0

    try:
        last_dt = datetime.fromisoformat(last_activity_iso)
    except (ValueError, TypeError):
        return 0.0

    if reference_time is None:
        reference_time = datetime.now(tz=UTC)

    # Normalise timezone awareness so subtraction works.
    if last_dt.tzinfo is None and reference_time.tzinfo is not None:
        last_dt = last_dt.replace(tzinfo=UTC)
    elif last_dt.tzinfo is not None and reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)

    days_elapsed = (reference_time - last_dt).total_seconds() / 86400

    if days_elapsed < 0:
        return 1.0

    return 0.5 ** (days_elapsed / half_life_days)


def compute_competence_score(
    entries: int,
    edge_count: int | None,
    last_activity_iso: str,
    reference_time: datetime | None = None,
    outcome_rate: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute a weighted competence score from available signals.

    Selects the weight set based on which signals are present.  When
    ``outcome_rate`` is provided the weight sets that include the
    outcome dimension are used; otherwise the NO_OUTCOME fallback sets
    (identical to the original weights) are selected for graceful
    degradation.

    Returns ``(final_score, entry_s, density_s, recency_s)``.
    """
    entry_s = compute_entry_score(entries)

    has_graph = edge_count is not None
    has_recency = bool(last_activity_iso and last_activity_iso.strip())
    has_outcome = outcome_rate is not None

    density_s = compute_entity_density(edge_count, MAX_EDGES_CEILING) if has_graph else 0.0
    recency_s = compute_recency_weight(last_activity_iso, reference_time) if has_recency else 0.0

    if has_graph and has_recency:
        weights = WEIGHTS_ALL if has_outcome else WEIGHTS_ALL_NO_OUTCOME
    elif not has_graph and has_recency:
        weights = WEIGHTS_NO_GRAPH if has_outcome else WEIGHTS_NO_GRAPH_NO_OUTCOME
    elif has_graph and not has_recency:
        weights = WEIGHTS_NO_RECENCY if has_outcome else WEIGHTS_NO_RECENCY_NO_OUTCOME
    else:
        weights = WEIGHTS_ENTRY_ONLY if has_outcome else WEIGHTS_ENTRY_ONLY_NO_OUTCOME

    final_score = (
        weights.get("entry", 0.0) * entry_s
        + weights.get("density", 0.0) * density_s
        + weights.get("recency", 0.0) * (recency_s or 0.0)
        + weights.get("outcome", 0.0) * (outcome_rate or 0.0)
    )

    return (final_score, entry_s, density_s, recency_s)


# ---------------------------------------------------------------------------
# Extension-to-topic mapping for auto-capture classification
# ---------------------------------------------------------------------------

# NOTE: .tsx -> "angular" is specific to this user's datalake.
# Override via datalake category metadata for other frameworks.
_EXT_TO_TOPIC: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "angular",
    ".sql": "postgresql",
    ".tf": "terraform",
}


# ---------------------------------------------------------------------------
# CompetenceBuilder — data gathering from datalake + graph
# ---------------------------------------------------------------------------


class CompetenceBuilder:
    """Gather raw competence signals from datalake and knowledge graph.

    Provides private methods that collect entry counts, graph density,
    and recency timestamps per topic.  These are consumed by the
    ``build()`` method (added in a later task) to produce a
    :class:`CompetenceMap`.
    """

    def __init__(
        self,
        datalake_path: Path,
        graph_engine: Any | None = None,
    ) -> None:
        # Lazy import to avoid circular dependency at module level.
        from src.core.personal_profile import DatalakeAnalyzer

        self.datalake_path = Path(datalake_path)
        self.analyzer = DatalakeAnalyzer(self.datalake_path)
        self.graph = graph_engine

    # -- entry counts -------------------------------------------------------

    def _get_entry_counts(self) -> Counter:
        """Return per-topic entry counts from training pairs.

        Delegates to :pymethod:`DatalakeAnalyzer.analyze_training_pairs` and
        filters out categories listed in ``META_CATEGORIES`` (task-type
        labels that do not represent knowledge topics).
        """
        from src.core.personal_profile import META_CATEGORIES

        stats = self.analyzer.analyze_training_pairs()
        categories: Counter = stats.get("categories", Counter())
        return Counter(
            {cat: count for cat, count in categories.items() if cat not in META_CATEGORIES}
        )

    # -- graph density ------------------------------------------------------

    def _get_topic_edge_counts(self, topics: list[str]) -> dict[str, int]:
        """Return the number of non-self-loop edges for each topic entity.

        Looks up each topic in the knowledge graph via
        ``find_entity_by_name`` and counts relations (both directions)
        excluding self-loops (``source_id == target_id``).

        Returns an empty dict when no graph engine is available.
        """
        if self.graph is None:
            return {}

        result: dict[str, int] = {}
        for topic in topics:
            entity = self.graph.find_entity_by_name(topic)
            if entity is None:
                continue
            relations = self.graph.get_relations(entity.id, direction="both")
            count = sum(1 for r in relations if r.source_id != r.target_id)
            result[topic] = count
        return result

    # -- recency ------------------------------------------------------------

    def _get_topic_recency(self, topics: list[str]) -> dict[str, str]:
        """Return the latest ISO timestamp per topic.

        Scans auto-capture JSONL files for timestamps and maps each
        record to topics via :pymethod:`_map_capture_to_topics`.  Also
        checks training-pair filenames for date prefixes.

        Returns a dict mapping topic -> latest ISO timestamp string.
        """
        latest: dict[str, str] = {}
        topic_set = set(topics)

        # 1. Scan auto-captures for timestamps
        ac_dir = self.datalake_path / "01-raw" / "code-changes"
        if ac_dir.exists():
            for jsonl_file in sorted(ac_dir.glob("*auto-captures*.jsonl")):
                records = self.analyzer._read_jsonl(jsonl_file)
                for record in records:
                    ts = record.get("timestamp", "")
                    if not ts:
                        continue
                    matched = self._map_capture_to_topics(record, topics)
                    for topic in matched:
                        if topic not in latest or ts > latest[topic]:
                            latest[topic] = ts

        # 2. Check training-pair filenames for date prefixes
        tp_dir = self.datalake_path / "02-processed" / "training-pairs"
        if tp_dir.exists():
            for jsonl_file in sorted(tp_dir.glob("*.jsonl")):
                date_str = self._extract_date_from_filename(jsonl_file.stem)
                if not date_str:
                    continue
                # Convert date to ISO timestamp for comparison
                ts = f"{date_str}T00:00:00"
                stem_lower = jsonl_file.stem.lower()
                for topic in topic_set:
                    if topic.lower() in stem_lower:
                        if topic not in latest or ts > latest[topic]:
                            latest[topic] = ts

        return latest

    # -- outcome rates ------------------------------------------------------

    def _get_outcome_rates(self, topics: list[str]) -> dict[str, float]:
        """Read outcomes and compute acceptance rate per topic.

        Scans all ``*_outcomes.jsonl`` files from
        ``datalake_path / "01-raw" / "outcomes"``.  Each record is
        expected to have ``topic`` and ``outcome`` fields where outcome
        is one of ``"accepted"``, ``"rejected"``, or ``"neutral"``.

        Neutral outcomes are ignored.  Returns the acceptance rate
        (accepted / non-neutral total) only for topics with at least 5
        non-neutral outcomes.
        """
        outcomes_dir = self.datalake_path / "01-raw" / "outcomes"
        if not outcomes_dir.exists():
            return {}

        topic_set = {t.lower() for t in topics}
        # Map lowercase topic -> {"accepted": int, "rejected": int}
        counts: dict[str, dict[str, int]] = {}

        for jsonl_file in sorted(outcomes_dir.glob("*_outcomes.jsonl")):
            records = self.analyzer._read_jsonl(jsonl_file)
            for record in records:
                topic_raw = record.get("topic", "")
                outcome = record.get("outcome", "")
                if not topic_raw or not outcome:
                    continue
                topic_lower = topic_raw.lower()
                if topic_lower not in topic_set:
                    continue
                if outcome == "neutral":
                    continue

                if topic_lower not in counts:
                    counts[topic_lower] = {"accepted": 0, "rejected": 0}

                if outcome == "accepted":
                    counts[topic_lower]["accepted"] += 1
                elif outcome == "rejected":
                    counts[topic_lower]["rejected"] += 1

        min_outcomes = 5
        result: dict[str, float] = {}
        for topic_lower, c in counts.items():
            total = c["accepted"] + c["rejected"]
            if total >= min_outcomes:
                result[topic_lower] = c["accepted"] / total

        return result

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _map_capture_to_topics(record: dict, topics: list[str]) -> list[str]:
        """Map an auto-capture record to matching topics.

        Matching strategies (in order):
        1. Record ``tags`` list contains the topic name.
        2. Record ``project`` name contains the topic name.
        3. File extension of ``file_modified`` maps via ``_EXT_TO_TOPIC``.
        """
        matched: list[str] = []
        record_tags = {t.lower() for t in record.get("tags", [])}
        project = (record.get("project") or "").lower()
        file_modified = record.get("file_modified", "")
        ext = Path(file_modified).suffix if file_modified else ""
        ext_topic = _EXT_TO_TOPIC.get(ext, "")

        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower in record_tags:
                matched.append(topic)
            elif topic_lower in project:
                matched.append(topic)
            elif ext_topic and ext_topic == topic_lower:
                matched.append(topic)

        return matched

    @staticmethod
    def _extract_date_from_filename(stem: str) -> str | None:
        """Extract an ISO date (YYYY-MM-DD) from a filename stem.

        Returns the first match or ``None`` if no date pattern is found.
        """
        m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
        return m.group(1) if m else None

    # -- build --------------------------------------------------------------

    def build(
        self,
        topics: list[str] | None = None,
        output_path: Path | None = None,
    ) -> CompetenceMap:
        """Build a complete competence map from datalake and graph signals.

        Steps:
        1. Gather entry counts from training pairs.
        2. Determine topic list (explicit or from entry_counts).
        3. Gather graph edge counts (if graph available).
        4. Gather recency timestamps.
        5. Score each topic and classify into levels.
        6. Sort by score descending.
        7. Optionally save to disk.

        Returns an empty :class:`CompetenceMap` when no entry counts exist.
        """
        entry_counts = self._get_entry_counts()
        if not entry_counts:
            logger.info("competence_build_empty", reason="no_entry_counts")
            return CompetenceMap()

        # Determine topic list
        if topics is None:
            topic_list = [cat for cat, _ in entry_counts.most_common()]
        else:
            topic_list = topics

        # Gather signals
        edge_counts = self._get_topic_edge_counts(topic_list) if self.graph is not None else {}
        recency = self._get_topic_recency(topic_list)
        outcome_rates = self._get_outcome_rates(topic_list)

        # Score each topic
        entries_list: list[CompetenceEntry] = []
        for topic in topic_list:
            count = entry_counts.get(topic, 0)
            edge_count = edge_counts.get(topic)  # None if not in graph
            last_activity = recency.get(topic, "")
            topic_outcome = outcome_rates.get(topic.lower())

            final_score, entry_s, density_s, recency_s = compute_competence_score(
                entries=count,
                edge_count=edge_count,
                last_activity_iso=last_activity,
                outcome_rate=topic_outcome,
            )

            level = _classify_level(final_score)
            entries_list.append(
                CompetenceEntry(
                    topic=topic,
                    score=final_score,
                    level=level,
                    entries=count,
                    entity_density=density_s,
                    recency_weight=recency_s,
                    last_activity=last_activity,
                )
            )

        # Sort by score descending
        entries_list.sort(key=lambda e: e.score, reverse=True)

        cmap = CompetenceMap(
            topics=entries_list,
            total_topics=len(entries_list),
        )

        if output_path:
            save_competence_map(cmap, Path(output_path))

        logger.info(
            "competence_map_built",
            topics=len(entries_list),
            experts=len(cmap.experts()),
        )
        return cmap


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_competence_map(cmap: CompetenceMap, path: Path) -> None:
    """Save a competence map to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cmap.to_dict(), indent=2, ensure_ascii=False))
    logger.info("competence_map_saved", path=str(path), topics=cmap.total_topics)


def load_competence_map(path: Path) -> CompetenceMap:
    """Load a competence map from a JSON file.

    Returns an empty default map if the file does not exist or is corrupt.
    """
    path = Path(path)
    if not path.exists():
        logger.debug("competence_map_not_found", path=str(path))
        return CompetenceMap()

    try:
        data = json.loads(path.read_text())
        cmap = CompetenceMap.from_dict(data)
        logger.info("competence_map_loaded", path=str(path), topics=cmap.total_topics)
        return cmap
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("competence_map_load_error", path=str(path), error=str(exc))
        return CompetenceMap()


# Simple cache to avoid re-reading competence map on every LLM call
# Cache keyed by path → (mtime, value). Re-reads only when the file changes on
# disk (FC-93), so `fabrik competence build` is picked up without a restart.
_competence_cache: dict[str, tuple[float, CompetenceMap]] = {}


def get_active_competence_map(map_path: Path | None = None) -> CompetenceMap:
    """Get the active competence map, cached and invalidated by file mtime.

    Repeated calls don't re-read from disk, but a rebuilt competence_map.json
    is reloaded automatically (mtime change). Pass a path or use the default.
    """
    from src.config import settings

    path = Path(map_path) if map_path else settings.data_dir / "profile" / "competence_map.json"
    cache_key = str(path)

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0  # missing/unreadable — reload when it appears

    cached = _competence_cache.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    cmap = load_competence_map(path)
    _competence_cache[cache_key] = (mtime, cmap)
    return cmap
