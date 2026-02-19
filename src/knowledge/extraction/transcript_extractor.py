"""Extract knowledge triples from Claude session transcript thinking blocks."""

import json
from pathlib import Path

import structlog

from src.knowledge.extraction.heuristic import HeuristicExtractor
from src.knowledge.graph_schema import EntityType, Triple

logger = structlog.get_logger(__name__)

MIN_THINKING_LENGTH = 100
TRANSCRIPT_CONFIDENCE = 0.65


class TranscriptExtractor:
    """Extract entities and relations from Claude session transcript thinking blocks.

    Parses JSONL transcript files, finds assistant messages with thinking blocks,
    and uses HeuristicExtractor to extract technologies, patterns, strategies,
    and errors from the reasoning text.
    """

    def __init__(self) -> None:
        self._heuristic = HeuristicExtractor()

    def extract_from_transcript(self, transcript_path: Path, source_doc: str = "") -> list[Triple]:
        """Extract triples from a single transcript file.

        Args:
            transcript_path: Path to the JSONL transcript file.
            source_doc: Optional source document identifier. Defaults to
                        ``transcript:{filename}``.

        Returns:
            List of extracted triples.
        """
        if not source_doc:
            source_doc = f"transcript:{transcript_path.name}"

        if not transcript_path.exists():
            return []

        try:
            _, _, triples = self._scan_transcript_with_stats(
                transcript_path, source_doc=source_doc,
            )
            return triples
        except Exception as exc:
            logger.warning("failed_to_read_transcript", path=str(transcript_path), error=str(exc))
            return []

    def scan_all_transcripts(
        self, transcripts_dir: Path, project_filter: str | None = None
    ) -> tuple[list[Triple], dict]:
        """Batch scan transcript directories for thinking blocks.

        Args:
            transcripts_dir: Root directory containing project subdirectories.
            project_filter: If set, only process subdirs whose name contains this string.
                            Defaults to FABRIK_PROJECT_FILTER env var, or None (all projects).

        Returns:
            Tuple of (triples, stats) where stats contains:
                - transcripts_scanned
                - thinking_blocks_found
                - thinking_blocks_processed
                - triples_extracted
                - errors
        """
        import os

        if project_filter is None:
            project_filter = os.environ.get("FABRIK_PROJECT_FILTER", "")

        stats = {
            "transcripts_scanned": 0,
            "thinking_blocks_found": 0,
            "thinking_blocks_processed": 0,
            "triples_extracted": 0,
            "errors": 0,
        }
        all_triples: list[Triple] = []

        if not transcripts_dir.exists():
            return all_triples, stats

        for subdir in sorted(transcripts_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if project_filter and project_filter not in subdir.name:
                continue

            for transcript_file in sorted(subdir.glob("*.jsonl")):
                try:
                    found, processed, triples = self._scan_transcript_with_stats(transcript_file)
                    stats["transcripts_scanned"] += 1
                    stats["thinking_blocks_found"] += found
                    stats["thinking_blocks_processed"] += processed
                    stats["triples_extracted"] += len(triples)
                    all_triples.extend(triples)
                except Exception as exc:
                    stats["errors"] += 1
                    logger.warning(
                        "transcript_scan_error",
                        path=str(transcript_file),
                        error=str(exc),
                    )

        return all_triples, stats

    def _scan_transcript_with_stats(
        self, transcript_path: Path, source_doc: str = "",
    ) -> tuple[int, int, list[Triple]]:
        """Scan a single transcript file and return detailed stats.

        Uses streaming I/O (line-by-line) to avoid loading entire files into
        memory. Individual transcripts can be up to ~50 MB.

        Args:
            transcript_path: Path to the JSONL transcript file.
            source_doc: Source identifier. Defaults to ``transcript:{filename}``.

        Returns:
            Tuple of (thinking_blocks_found, thinking_blocks_processed, triples).
        """
        if not source_doc:
            source_doc = f"transcript:{transcript_path.name}"
        found = 0
        processed = 0
        triples: list[Triple] = []

        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict):
                    continue

                if record.get("type") != "assistant":
                    continue

                message = record.get("message", {})
                content_blocks = message.get("content", [])
                if not isinstance(content_blocks, list):
                    continue

                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "thinking":
                        continue

                    thinking_text = block.get("thinking", "")
                    found += 1

                    if len(thinking_text) < MIN_THINKING_LENGTH:
                        continue

                    processed += 1
                    extracted = self._extract_from_thinking(thinking_text, source_doc)
                    triples.extend(extracted)

        return found, processed, triples

    def _extract_from_thinking(self, thinking_text: str, source_doc: str) -> list[Triple]:
        """Extract triples from a single thinking block using heuristic internals.

        All returned triples get confidence = TRANSCRIPT_CONFIDENCE (0.65).

        Args:
            thinking_text: The raw thinking/reasoning text.
            source_doc: Source document identifier.

        Returns:
            List of triples with confidence set to TRANSCRIPT_CONFIDENCE.
        """
        triples: list[Triple] = []

        # Extract technologies
        tech_triples = self._heuristic._extract_technologies(thinking_text, source_doc)
        triples.extend(tech_triples)

        # Extract patterns
        pattern_triples = self._heuristic._extract_patterns(thinking_text, source_doc)
        triples.extend(pattern_triples)

        # Extract strategies
        strategy_triples = self._heuristic._extract_strategies(thinking_text, source_doc)
        triples.extend(strategy_triples)

        # Extract errors
        error_triples = self._heuristic._extract_errors(thinking_text, source_doc)
        triples.extend(error_triples)

        # Co-occurrence relations between technologies
        tech_names = [t.subject_name for t in tech_triples if t.subject_type == EntityType.TECHNOLOGY]
        cooccurrence_triples = self._heuristic._create_cooccurrence_relations(
            tech_names, EntityType.TECHNOLOGY, source_doc,
        )
        triples.extend(cooccurrence_triples)

        # Override confidence for ALL triples to TRANSCRIPT_CONFIDENCE
        for t in triples:
            t.confidence = TRANSCRIPT_CONFIDENCE

        return triples
