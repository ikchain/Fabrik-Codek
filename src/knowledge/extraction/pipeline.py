"""Extraction pipeline - orchestrates heuristic and LLM extractors."""

import json
from datetime import datetime
from pathlib import Path

import structlog

from src.config import settings
from src.core.llm_client import LLMClient
from src.knowledge.graph_engine import GraphEngine
from src.knowledge.graph_schema import Triple
from src.knowledge.extraction.heuristic import HeuristicExtractor
from src.knowledge.extraction.llm_extractor import LLMExtractor
from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

logger = structlog.get_logger()


def _parse_multiline_json(file_path: Path) -> list[dict]:
    """Parse a file containing JSON records in JSONL or pretty-printed format.

    Tries line-by-line JSONL first. If that fails, uses json.JSONDecoder.raw_decode
    to incrementally parse concatenated JSON objects (handles pretty-printed output
    from jq, including strings with unbalanced braces).
    """
    records: list[dict] = []
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        return records

    # Try JSONL first (fast path for single-line-per-record files)
    lines = text.split("\n")
    jsonl_ok = 0
    jsonl_fail = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                records.append(obj)
                jsonl_ok += 1
            else:
                jsonl_fail += 1
        except json.JSONDecodeError:
            jsonl_fail += 1

    if jsonl_ok > 0 and jsonl_ok >= jsonl_fail:
        return records

    # Fallback: incremental raw_decode for pretty-printed / concatenated JSON
    records = []
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
            # Skip to next '{' to try recovering
            next_brace = text.find("{", idx + 1)
            if next_brace == -1:
                break
            idx = next_brace

    return records


class ExtractionPipeline:
    """Orchestrates extraction from datalake into the knowledge graph."""

    def __init__(
        self,
        engine: GraphEngine | None = None,
        use_llm: bool = False,
        include_transcripts: bool = False,
    ):
        self.engine = engine or GraphEngine()
        self.heuristic = HeuristicExtractor()
        self.llm_extractor = LLMExtractor() if use_llm else None
        self.transcript_extractor = TranscriptExtractor() if include_transcripts else None
        self._llm_available = False
        self._stats = {
            "files_processed": 0,
            "pairs_processed": 0,
            "triples_extracted": 0,
            "llm_triples_extracted": 0,
            "transcript_triples_extracted": 0,
            "inferred_triples": 0,
            "errors": 0,
        }

    async def build(self, force: bool = False, transcripts_dir: Path | None = None) -> dict:
        """Build the knowledge graph from datalake data.

        Args:
            force: If True, rebuild from scratch. Otherwise, incremental.
            transcripts_dir: Optional path to session transcripts directory.

        Returns:
            Build statistics dict.
        """
        self._stats = {
            "files_processed": 0,
            "pairs_processed": 0,
            "triples_extracted": 0,
            "llm_triples_extracted": 0,
            "transcript_triples_extracted": 0,
            "inferred_triples": 0,
            "errors": 0,
        }

        # Load existing graph (for incremental builds)
        if not force:
            self.engine.load()

        state = {} if force else self.engine.load_extraction_state()
        processed_files = state.get("processed_files", {})

        # Check LLM availability if requested
        self._llm_available = False
        if self.llm_extractor:
            try:
                async with LLMClient(model=self.llm_extractor.model) as client:
                    self._llm_available = await client.health_check()
                if self._llm_available:
                    logger.info("llm_extractor_available", model=self.llm_extractor.model)
                else:
                    logger.warning("llm_extractor_unavailable", model=self.llm_extractor.model)
            except (OSError, ConnectionError):
                logger.warning("llm_extractor_check_failed")

        datalake = settings.datalake_path
        if not datalake.exists():
            logger.warning("datalake_not_found", path=str(datalake))
            return self._stats

        # 1. Process training pairs
        training_dir = datalake / "02-processed" / "training-pairs"
        if training_dir.exists():
            await self._process_training_pairs(training_dir, processed_files)

        # 2. Process decisions
        decisions_dir = datalake / "03-metadata" / "decisions"
        if decisions_dir.exists():
            await self._process_decisions(decisions_dir, processed_files)

        # 3. Process learnings
        learnings_dir = datalake / "03-metadata" / "learnings"
        if learnings_dir.exists():
            await self._process_learnings(learnings_dir, processed_files)

        # 4. Process auto-captures
        auto_dir = datalake / "01-raw" / "code-changes"
        if auto_dir.exists():
            await self._process_auto_captures(auto_dir, processed_files)

        # 5. Process enriched captures (with reasoning)
        enriched_dir = auto_dir / "enriched" if auto_dir.exists() else None
        if enriched_dir and enriched_dir.exists():
            await self._process_enriched_captures(enriched_dir, processed_files)

        # 6. Process session transcripts (reasoning from thinking blocks)
        if self.transcript_extractor:
            if transcripts_dir is None:
                transcripts_dir = Path.home() / ".claude" / "projects"
            if transcripts_dir.exists():
                await self._process_transcripts(transcripts_dir, processed_files)

        # 7. Graph completion (transitive inference)
        completion_stats = self.engine.complete()
        self._stats["inferred_triples"] = completion_stats["inferred_count"]

        # Save graph and state
        self.engine.save()
        state["processed_files"] = processed_files
        self.engine.save_extraction_state(state)

        logger.info("extraction_complete", **self._stats)
        return self._stats

    async def _process_training_pairs(
        self, training_dir: Path, processed_files: dict,
    ) -> None:
        """Process JSONL training pair files."""
        for jsonl_file in sorted(training_dir.glob("*.jsonl")):
            file_key = str(jsonl_file.relative_to(settings.datalake_path))

            # Skip already processed (incremental)
            file_mtime = jsonl_file.stat().st_mtime
            if file_key in processed_files:
                if processed_files[file_key] >= file_mtime:
                    continue

            try:
                await self._extract_from_jsonl(jsonl_file)
                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
            except Exception as e:
                logger.error("extraction_error", file=str(jsonl_file), error=str(e))
                self._stats["errors"] += 1

    async def _extract_from_jsonl(self, file_path: Path) -> None:
        """Extract triples from a JSONL file using heuristic + optional LLM."""
        source_prefix = file_path.stem

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError:
                    continue

                source_doc = f"{source_prefix}:{line_num}"

                # Heuristic extraction (always)
                triples = self.heuristic.extract_from_pair(pair, source_doc=source_doc)
                for triple in triples:
                    self.engine.ingest_triple(triple)
                self._stats["triples_extracted"] += len(triples)

                # LLM extraction (if available)
                if self.llm_extractor and self._llm_available:
                    try:
                        llm_triples = await self.llm_extractor.extract_from_pair(
                            pair, source_doc=f"llm:{source_doc}",
                        )
                        for triple in llm_triples:
                            self.engine.ingest_triple(triple)
                        self._stats["llm_triples_extracted"] += len(llm_triples)
                    except Exception as e:
                        logger.debug("llm_pair_extraction_failed", source_doc=source_doc, error=str(e))

                self._stats["pairs_processed"] += 1

    async def _process_decisions(
        self, decisions_dir: Path, processed_files: dict,
    ) -> None:
        """Process decision markdown/JSON files."""
        for dec_file in sorted(decisions_dir.glob("**/*.md")):
            file_key = str(dec_file.relative_to(settings.datalake_path))
            file_mtime = dec_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                content = dec_file.read_text(encoding="utf-8", errors="ignore")
                decision = self._parse_markdown_decision(content, dec_file.stem)
                triples = self.heuristic.extract_from_decision(decision, source_doc=file_key)

                for triple in triples:
                    self.engine.ingest_triple(triple)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += len(triples)
            except Exception as e:
                logger.error("decision_extraction_error", file=str(dec_file), error=str(e))
                self._stats["errors"] += 1

        # Also check for JSON decisions
        for dec_file in sorted(decisions_dir.glob("**/*.json")):
            file_key = str(dec_file.relative_to(settings.datalake_path))
            file_mtime = dec_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                decision = json.loads(dec_file.read_text(encoding="utf-8"))
                triples = self.heuristic.extract_from_decision(decision, source_doc=file_key)

                for triple in triples:
                    self.engine.ingest_triple(triple)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += len(triples)
            except Exception as e:
                logger.error("decision_extraction_error", file=str(dec_file), error=str(e))
                self._stats["errors"] += 1

        # Also check for JSONL decisions (from logger)
        for dec_file in sorted(decisions_dir.glob("**/*.jsonl")):
            file_key = str(dec_file.relative_to(settings.datalake_path))
            file_mtime = dec_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                records = _parse_multiline_json(dec_file)
                total_triples = 0
                for record in records:
                    triples = self.heuristic.extract_from_decision(record, source_doc=file_key)
                    for triple in triples:
                        self.engine.ingest_triple(triple)
                    total_triples += len(triples)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += total_triples
            except Exception as e:
                logger.error("decision_jsonl_error", file=str(dec_file), error=str(e))
                self._stats["errors"] += 1

    async def _process_learnings(
        self, learnings_dir: Path, processed_files: dict,
    ) -> None:
        """Process learning documents."""
        for learn_file in sorted(learnings_dir.glob("**/*.md")):
            file_key = str(learn_file.relative_to(settings.datalake_path))
            file_mtime = learn_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                content = learn_file.read_text(encoding="utf-8", errors="ignore")
                learning = {"topic": learn_file.stem, "applicable_to": content}
                triples = self.heuristic.extract_from_learning(learning, source_doc=file_key)

                for triple in triples:
                    self.engine.ingest_triple(triple)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += len(triples)
            except Exception as e:
                logger.error("learning_extraction_error", file=str(learn_file), error=str(e))
                self._stats["errors"] += 1

        # Also check for JSONL learnings (from logger)
        for learn_file in sorted(learnings_dir.glob("**/*.jsonl")):
            file_key = str(learn_file.relative_to(settings.datalake_path))
            file_mtime = learn_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                records = _parse_multiline_json(learn_file)
                total_triples = 0
                for record in records:
                    triples = self.heuristic.extract_from_learning(record, source_doc=file_key)
                    for triple in triples:
                        self.engine.ingest_triple(triple)
                    total_triples += len(triples)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += total_triples
            except Exception as e:
                logger.error("learning_jsonl_error", file=str(learn_file), error=str(e))
                self._stats["errors"] += 1

    async def _process_auto_captures(
        self, auto_dir: Path, processed_files: dict,
    ) -> None:
        """Process auto-capture JSONL files from code-changes."""
        for cap_file in sorted(auto_dir.glob("*auto-captures*.jsonl")):
            file_key = str(cap_file.relative_to(settings.datalake_path))
            file_mtime = cap_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                records = _parse_multiline_json(cap_file)
                total_triples = 0
                for record in records:
                    if record.get("type") != "auto_capture":
                        continue
                    triples = self.heuristic.extract_from_auto_capture(
                        record, source_doc=file_key,
                    )
                    for triple in triples:
                        self.engine.ingest_triple(triple)
                    total_triples += len(triples)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += total_triples
            except Exception as e:
                logger.error("auto_capture_error", file=str(cap_file), error=str(e))
                self._stats["errors"] += 1

    async def _process_enriched_captures(
        self, enriched_dir: Path, processed_files: dict,
    ) -> None:
        """Process enriched auto-capture JSONL files (with reasoning)."""
        for enr_file in sorted(enriched_dir.glob("*enriched*.jsonl")):
            file_key = f"enriched/{enr_file.name}"
            file_mtime = enr_file.stat().st_mtime

            if file_key in processed_files and processed_files[file_key] >= file_mtime:
                continue

            try:
                records = _parse_multiline_json(enr_file)
                total_triples = 0
                for record in records:
                    if record.get("type") != "enriched_capture":
                        continue
                    triples = self.heuristic.extract_from_auto_capture(
                        record, source_doc=file_key,
                    )
                    for triple in triples:
                        self.engine.ingest_triple(triple)
                    total_triples += len(triples)

                processed_files[file_key] = file_mtime
                self._stats["files_processed"] += 1
                self._stats["triples_extracted"] += total_triples
            except Exception as e:
                logger.error("enriched_capture_error", file=str(enr_file), error=str(e))
                self._stats["errors"] += 1

    async def _process_transcripts(
        self, transcripts_dir: Path, processed_files: dict,
    ) -> None:
        """Process session transcripts for thinking block reasoning."""
        if not transcripts_dir.exists():
            return

        import os
        project_filter = os.environ.get("FABRIK_PROJECT_FILTER", "")

        for project_dir in sorted(transcripts_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            if project_filter and project_filter not in project_dir.name:
                continue

            for transcript_file in sorted(project_dir.glob("*.jsonl")):
                file_key = f"transcript:{project_dir.name}/{transcript_file.name}"
                file_mtime = transcript_file.stat().st_mtime

                if file_key in processed_files and processed_files[file_key] >= file_mtime:
                    continue

                try:
                    triples = self.transcript_extractor.extract_from_transcript(
                        transcript_file, source_doc=file_key,
                    )
                    for triple in triples:
                        self.engine.ingest_triple(triple)

                    processed_files[file_key] = file_mtime
                    self._stats["files_processed"] += 1
                    self._stats["transcript_triples_extracted"] += len(triples)
                except Exception as e:
                    logger.error(
                        "transcript_extraction_error",
                        file=str(transcript_file),
                        error=str(e),
                    )
                    self._stats["errors"] += 1

    def _parse_markdown_decision(self, content: str, title: str) -> dict:
        """Extract structured data from markdown decision document."""
        decision: dict = {"topic": title}

        # Simple heuristic: look for common patterns
        lines = content.split("\n")
        for i, line in enumerate(lines):
            lower = line.lower().strip()
            if lower.startswith("# "):
                decision["topic"] = line[2:].strip()
            elif "decision:" in lower or "chosen:" in lower or "option:" in lower:
                # Next non-empty line is the decision
                for next_line in lines[i + 1:]:
                    if next_line.strip():
                        decision["chosen_option"] = next_line.strip()
                        break
            elif "lesson" in lower or "learning" in lower or "takeaway" in lower:
                for next_line in lines[i + 1:]:
                    if next_line.strip():
                        decision["lesson_learned"] = next_line.strip()
                        break

        return decision
