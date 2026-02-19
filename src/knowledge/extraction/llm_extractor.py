"""LLM-assisted entity and relation extraction using Qwen.

Uses LLMClient to send training pair instructions to the model
and parses structured JSON responses into knowledge graph triples.
"""

import asyncio
import json
import structlog

from src.core.llm_client import LLMClient
from src.knowledge.graph_schema import EntityType, RelationType, Triple

logger = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are a technical knowledge extractor. Given a technical instruction, "
    "extract entities (technologies, patterns, concepts, strategies) and their relationships. "
    "Return ONLY valid JSON. No explanation."
)

EXTRACTION_PROMPT = """Extract entities and relationships from this technical text.

Category: {category}
Topic: {topic}
Text: {instruction}

Entity types: technology, pattern, concept, strategy, error_type
Relation types: uses, depends_on, part_of, alternative_to, related_to, fixes, learned_from

Return JSON:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"source": "...", "target": "...", "type": "..."}}]}}"""

TYPE_MAP = {
    "concept": EntityType.CONCEPT,
    "technology": EntityType.TECHNOLOGY,
    "pattern": EntityType.PATTERN,
    "error_type": EntityType.ERROR_TYPE,
    "strategy": EntityType.STRATEGY,
}

ENTITY_TYPE_FALLBACK: dict[str, EntityType] = {
    "library": EntityType.TECHNOLOGY,
    "framework": EntityType.TECHNOLOGY,
    "tool": EntityType.TECHNOLOGY,
    "language": EntityType.TECHNOLOGY,
    "database": EntityType.TECHNOLOGY,
    "service": EntityType.TECHNOLOGY,
    "design_pattern": EntityType.PATTERN,
    "architecture": EntityType.PATTERN,
    "method": EntityType.STRATEGY,
    "technique": EntityType.STRATEGY,
    "approach": EntityType.STRATEGY,
    "bug": EntityType.ERROR_TYPE,
    "issue": EntityType.ERROR_TYPE,
}

REL_TYPE_MAP = {
    "uses": RelationType.USES,
    "related_to": RelationType.RELATED_TO,
    "fixes": RelationType.FIXES,
    "part_of": RelationType.PART_OF,
    "depends_on": RelationType.DEPENDS_ON,
    "alternative_to": RelationType.ALTERNATIVE_TO,
    "learned_from": RelationType.LEARNED_FROM,
}

MAX_CONSECUTIVE_ERRORS = 5


class LLMExtractor:
    """Extract entities and relations using LLM (Qwen via Ollama).

    Sends instruction + category + topic to the model and parses
    structured JSON responses into knowledge graph triples (confidence 0.6).
    """

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model
        self._available = False

    async def check_availability(self) -> bool:
        """Check if the LLM model is available."""
        try:
            async with LLMClient(model=self.model) as client:
                self._available = await client.health_check()
                return self._available
        except (OSError, ConnectionError):
            self._available = False
            return False

    async def extract_from_pair(self, pair: dict, source_doc: str = "") -> list[Triple]:
        """Extract triples from a training pair using LLM.

        Sends only instruction + category + topic (no output field).
        Returns triples with confidence 0.6 (lower than heuristic 0.7-0.8).
        """
        instruction = pair.get("instruction", "").strip()
        if not instruction:
            return []

        category = pair.get("category", "")
        topic = pair.get("topic", "")

        prompt = EXTRACTION_PROMPT.format(
            category=category,
            topic=topic,
            instruction=instruction,
        )

        try:
            async with LLMClient(model=self.model, timeout=30.0) as client:
                response = await client.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=0.1,
                )
            triples = self._parse_llm_response(response.content, source_doc)
            logger.debug(
                "llm_extraction_done",
                source_doc=source_doc,
                triples_count=len(triples),
                latency_ms=round(response.latency_ms, 1),
            )
            return triples
        except Exception as e:
            logger.warning("llm_extraction_error", source_doc=source_doc, error=str(e))
            raise

    async def extract_batch(
        self,
        pairs: list[dict],
        source_docs: list[str],
        batch_size: int = 50,
        delay: float = 0.5,
    ) -> list[Triple]:
        """Extract triples from a batch of pairs sequentially.

        Uses a circuit breaker: stops after MAX_CONSECUTIVE_ERRORS consecutive
        failures to avoid hammering a down Ollama instance.

        Args:
            pairs: List of training pair dicts.
            source_docs: Corresponding source document identifiers.
            batch_size: Log progress every N pairs.
            delay: Seconds to wait between requests.

        Returns:
            Combined list of all extracted triples.
        """
        all_triples: list[Triple] = []
        consecutive_errors = 0
        processed = 0
        errors = 0

        for i, (pair, source_doc) in enumerate(zip(pairs, source_docs)):
            try:
                triples = await self.extract_from_pair(pair, source_doc)
                all_triples.extend(triples)
                consecutive_errors = 0
                processed += 1
            except Exception as e:
                consecutive_errors += 1
                errors += 1
                logger.debug("llm_batch_pair_error", index=i, error=str(e))

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "llm_circuit_breaker_open",
                        consecutive_errors=consecutive_errors,
                        processed=processed,
                        total=len(pairs),
                    )
                    break

            if (i + 1) % batch_size == 0:
                logger.info(
                    "llm_batch_progress",
                    processed=processed,
                    errors=errors,
                    total=len(pairs),
                    triples_so_far=len(all_triples),
                )

            if delay > 0 and i < len(pairs) - 1:
                await asyncio.sleep(delay)

        logger.info(
            "llm_batch_complete",
            processed=processed,
            errors=errors,
            total=len(pairs),
            triples_extracted=len(all_triples),
        )

        return all_triples

    def _parse_llm_response(self, response: str, source_doc: str) -> list[Triple]:
        """Parse LLM JSON response into triples.

        Handles: markdown fences, unknown types (fallback), empty entities,
        truncated JSON, extra fields.
        """
        if not response or not response.strip():
            return []

        cleaned = self._strip_markdown_fences(response)
        data = self._try_parse_json(cleaned)
        if data is None:
            return []

        triples = []
        entities = data.get("entities", [])
        relations = data.get("relations", [])

        # Build entity type lookup
        entity_types: dict[str, EntityType] = {}
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name", "").strip().lower()
            raw_type = ent.get("type", "").strip().lower()
            etype = TYPE_MAP.get(raw_type) or ENTITY_TYPE_FALLBACK.get(raw_type, EntityType.CONCEPT)
            if name:
                entity_types[name] = etype

        # Create relation triples
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            src = rel.get("source", "").strip().lower()
            tgt = rel.get("target", "").strip().lower()
            raw_rtype = rel.get("type", "").strip().lower()
            rtype = REL_TYPE_MAP.get(raw_rtype, RelationType.RELATED_TO)

            if not src or not tgt:
                continue

            triples.append(Triple(
                subject_name=src,
                subject_type=entity_types.get(src, EntityType.CONCEPT),
                relation_type=rtype,
                object_name=tgt,
                object_type=entity_types.get(tgt, EntityType.CONCEPT),
                source_doc=source_doc,
                confidence=0.6,
            ))

        return triples

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences wrapping JSON."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
        return stripped

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Try to parse JSON, with recovery for truncated responses."""
        start = text.find("{")
        if start == -1:
            return None
        end = text.rfind("}") + 1
        if end <= start:
            fragment = text[start:]
            for closer in ['}]}', ']}', '}']:
                try:
                    return json.loads(fragment + closer)
                except json.JSONDecodeError:
                    continue
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
