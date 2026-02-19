# LLM Extractor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Activate the LLM extractor stub to extract entities and relations from training pairs via qwen2.5-coder:7b (Ollama), complementing the heuristic extractor.

**Architecture:** The `LLMExtractor` uses `LLMClient.generate()` to send training pair instructions to qwen2.5-coder:7b with a structured extraction prompt. Results are parsed into `Triple` objects and ingested alongside heuristic triples. Sequential processing with circuit breaker for robustness.

**Tech Stack:** Python 3.12, asyncio, httpx (via LLMClient), structlog, pytest, unittest.mock

**Design doc:** `docs/plans/2026-02-15-llm-extractor-design.md`

---

### Task 1: Reforzar _parse_llm_response() - tests

**Files:**
- Modify: `tests/test_extraction.py` (add tests after existing `TestLLMExtractor` class, line ~205)

**Step 1: Write failing tests for parse robustness**

Add these tests inside the existing `TestLLMExtractor` class in `tests/test_extraction.py`:

```python
def test_parse_markdown_fences(self):
    """LLM sometimes wraps JSON in markdown code fences."""
    extractor = LLMExtractor()
    response = '```json\n{"entities": [{"name": "Docker", "type": "technology"}], "relations": []}\n```'
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert isinstance(triples, list)
    # Should parse successfully despite fences (currently no entities = 0 triples from relations)
    # But entities are parsed into entity_types dict, so we verify no crash

def test_parse_markdown_fences_with_relations(self):
    """Markdown fences with actual relations should produce triples."""
    extractor = LLMExtractor()
    response = '```json\n{"entities": [{"name": "Docker", "type": "technology"}, {"name": "Kubernetes", "type": "technology"}], "relations": [{"source": "Kubernetes", "target": "Docker", "type": "uses"}]}\n```'
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert len(triples) == 1
    assert triples[0].subject_name == "kubernetes"
    assert triples[0].object_name == "docker"
    assert triples[0].relation_type == RelationType.USES

def test_parse_unknown_entity_type_fallback(self):
    """Unknown entity types should fallback to CONCEPT, not crash."""
    extractor = LLMExtractor()
    response = json.dumps({
        "entities": [
            {"name": "Redis", "type": "library"},
            {"name": "caching", "type": "unknown_type"},
        ],
        "relations": [
            {"source": "Redis", "target": "caching", "type": "uses"},
        ],
    })
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert len(triples) == 1
    assert triples[0].subject_type == EntityType.TECHNOLOGY  # "library" -> TECHNOLOGY fallback
    assert triples[0].object_type == EntityType.CONCEPT  # "unknown_type" -> CONCEPT default

def test_parse_unknown_relation_type_fallback(self):
    """Unknown relation types should fallback to RELATED_TO."""
    extractor = LLMExtractor()
    response = json.dumps({
        "entities": [
            {"name": "A", "type": "concept"},
            {"name": "B", "type": "concept"},
        ],
        "relations": [
            {"source": "A", "target": "B", "type": "implements"},
        ],
    })
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert len(triples) == 1
    assert triples[0].relation_type == RelationType.RELATED_TO

def test_parse_empty_source_target_discarded(self):
    """Relations with empty source or target should be silently discarded."""
    extractor = LLMExtractor()
    response = json.dumps({
        "entities": [{"name": "FastAPI", "type": "technology"}],
        "relations": [
            {"source": "", "target": "FastAPI", "type": "uses"},
            {"source": "FastAPI", "target": "  ", "type": "uses"},
            {"source": "FastAPI", "target": "Pydantic", "type": "uses"},
        ],
    })
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert len(triples) == 1  # Only the valid one
    assert triples[0].subject_name == "fastapi"
    assert triples[0].object_name == "pydantic"

def test_parse_truncated_json_recovery(self):
    """Truncated JSON should attempt recovery or return []."""
    extractor = LLMExtractor()
    response = '{"entities": [{"name": "FastAPI", "type": "technology"}], "relations": [{"source": "FastAPI", "target": "Py'
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert isinstance(triples, list)
    # May recover entities-only or return [] - both acceptable

def test_parse_extra_fields_ignored(self):
    """Extra fields in entities/relations should be ignored without error."""
    extractor = LLMExtractor()
    response = json.dumps({
        "entities": [
            {"name": "FastAPI", "type": "technology", "description": "web framework", "extra": True},
        ],
        "relations": [
            {"source": "FastAPI", "target": "Python", "type": "uses", "confidence": 0.9, "note": "obvious"},
        ],
    })
    triples = extractor._parse_llm_response(response, source_doc="test")
    assert len(triples) == 1

def test_parse_empty_string(self):
    """Empty string response returns []."""
    extractor = LLMExtractor()
    triples = extractor._parse_llm_response("", source_doc="test")
    assert triples == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_extraction.py::TestLLMExtractor -v 2>&1 | tail -20`

Expected: Several FAIL - markdown fences test fails because current code can't find `{` inside fences (it finds the one after ````json\n`), unknown type tests fail because `TYPE_MAP` doesn't have "library" and defaults to CONCEPT not TECHNOLOGY, empty source/target test fails because current code doesn't filter empties.

**Step 3: Commit test-only**

```bash
git add tests/test_extraction.py
git commit -m "TEST: Tests de robustez para _parse_llm_response()"
```

---

### Task 2: Reforzar _parse_llm_response() - implementacion

**Files:**
- Modify: `src/knowledge/extraction/llm_extractor.py` (method `_parse_llm_response`, lines 82-126)

**Step 1: Implement the reinforced parser**

Replace `_parse_llm_response` and add helper constants in `llm_extractor.py`:

```python
# Add after existing TYPE_MAP (line 29)
# Fallback map for unknown entity types - common LLM inventions
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
```

Replace `_parse_llm_response`:

```python
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
        # Remove opening fence (```json or ```)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped

@staticmethod
def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON, with recovery for truncated responses."""
    # Direct parse
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}") + 1
    if end <= start:
        # Truncated - try to close it
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
```

**Step 2: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_extraction.py::TestLLMExtractor -v`

Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/knowledge/extraction/llm_extractor.py
git commit -m "FEAT: Reforzar _parse_llm_response() con markdown fences, fallback tipos, recovery"
```

---

### Task 3: Implementar extract_from_pair() - tests

**Files:**
- Modify: `tests/test_extraction.py` (new test class after `TestLLMExtractor`)

**Step 1: Write failing tests**

Add a new test class:

```python
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.llm_client import LLMResponse


class TestLLMExtractFromPair:
    """Tests for LLMExtractor.extract_from_pair() with mocked LLMClient."""

    @pytest.fixture
    def llm_extractor(self):
        return LLMExtractor(model="qwen2.5-coder:7b")

    @pytest.fixture
    def sample_pair(self):
        return {
            "instruction": "How to create a REST API with FastAPI and Pydantic?",
            "output": "Use FastAPI with Pydantic models... (long code omitted)",
            "category": "api",
            "topic": "web-development",
        }

    @pytest.fixture
    def llm_response_valid(self):
        """Fixture simulating a real qwen2.5-coder:7b response."""
        return LLMResponse(
            content=json.dumps({
                "entities": [
                    {"name": "FastAPI", "type": "technology"},
                    {"name": "Pydantic", "type": "technology"},
                    {"name": "REST API", "type": "concept"},
                ],
                "relations": [
                    {"source": "FastAPI", "target": "Pydantic", "type": "uses"},
                    {"source": "REST API", "target": "FastAPI", "type": "depends_on"},
                ],
            }),
            model="qwen2.5-coder:7b",
            tokens_used=150,
            latency_ms=2500.0,
        )

    @pytest.mark.asyncio
    async def test_extract_from_pair_returns_triples(self, llm_extractor, sample_pair, llm_response_valid):
        """extract_from_pair should return parsed triples from LLM response."""
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(return_value=llm_response_valid)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            triples = await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")

            assert len(triples) == 2
            assert all(t.confidence == 0.6 for t in triples)
            names = {(t.subject_name, t.object_name) for t in triples}
            assert ("fastapi", "pydantic") in names

    @pytest.mark.asyncio
    async def test_extract_from_pair_prompt_uses_instruction_not_output(self, llm_extractor, sample_pair, llm_response_valid):
        """Prompt should contain instruction, category, topic but NOT output."""
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(return_value=llm_response_valid)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")

            call_args = mock_instance.generate.call_args
            prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
            assert "REST API with FastAPI" in prompt  # instruction present
            assert "api" in prompt.lower()  # category present
            assert "(long code omitted)" not in prompt  # output NOT present

    @pytest.mark.asyncio
    async def test_extract_from_pair_uses_low_temperature(self, llm_extractor, sample_pair, llm_response_valid):
        """Should use low temperature for deterministic extraction."""
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(return_value=llm_response_valid)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")

            call_args = mock_instance.generate.call_args
            temperature = call_args.kwargs.get("temperature")
            assert temperature is not None
            assert temperature <= 0.2

    @pytest.mark.asyncio
    async def test_extract_from_pair_llm_returns_garbage(self, llm_extractor, sample_pair):
        """If LLM returns unparseable text, return empty list without error."""
        garbage_response = LLMResponse(
            content="I cannot extract entities from this text.",
            model="qwen2.5-coder:7b",
            tokens_used=20,
            latency_ms=1000.0,
        )
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(return_value=garbage_response)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            triples = await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")
            assert triples == []

    @pytest.mark.asyncio
    async def test_extract_from_pair_empty_instruction(self, llm_extractor):
        """Pair with empty instruction should return [] without calling LLM."""
        pair = {"instruction": "", "output": "something", "category": "test"}
        triples = await llm_extractor.extract_from_pair(pair, source_doc="test:1")
        assert triples == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_extraction.py::TestLLMExtractFromPair -v 2>&1 | tail -20`

Expected: FAIL - `extract_from_pair` is a stub returning `[]`

**Step 3: Commit tests**

```bash
git add tests/test_extraction.py
git commit -m "TEST: Tests para extract_from_pair() con LLMClient mockeado"
```

---

### Task 4: Implementar extract_from_pair() - codigo

**Files:**
- Modify: `src/knowledge/extraction/llm_extractor.py`

**Step 1: Implement extract_from_pair()**

Update imports at top of file:

```python
import asyncio
import json
import structlog

from src.core.llm_client import LLMClient
from src.knowledge.graph_schema import EntityType, RelationType, Triple
```

Replace `EXTRACTION_PROMPT` (line 14-21) with the new prompts:

```python
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
```

Replace `extract_from_pair` stub (lines 64-70):

```python
async def extract_from_pair(self, pair: dict, source_doc: str = "") -> list[Triple]:
    """Extract triples from a training pair using LLM.

    Sends only instruction + category + topic to the LLM (no output field).
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
```

**Step 2: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_extraction.py::TestLLMExtractFromPair -v`

Expected: ALL PASS

**Step 3: Run ALL existing tests to verify no regression**

Run: `python3 -m pytest tests/test_extraction.py -v 2>&1 | tail -30`

Expected: ALL PASS (including old TestLLMExtractor and TestHeuristicExtractor)

**Step 4: Commit**

```bash
git add src/knowledge/extraction/llm_extractor.py
git commit -m "FEAT: Implementar extract_from_pair() con LLMClient"
```

---

### Task 5: Implementar extract_batch() - tests

**Files:**
- Modify: `tests/test_extraction.py`

**Step 1: Write failing tests**

Add new test class:

```python
class TestLLMExtractBatch:
    """Tests for LLMExtractor.extract_batch() - sequential processing + circuit breaker."""

    @pytest.fixture
    def llm_extractor(self):
        return LLMExtractor(model="qwen2.5-coder:7b")

    @pytest.fixture
    def sample_pairs(self):
        return [
            {"instruction": f"How to use tool {i}?", "category": "test", "topic": "tools"}
            for i in range(5)
        ]

    @pytest.fixture
    def valid_response(self):
        return LLMResponse(
            content=json.dumps({
                "entities": [{"name": "tool", "type": "technology"}],
                "relations": [],
            }),
            model="qwen2.5-coder:7b",
            tokens_used=50,
            latency_ms=1000.0,
        )

    @pytest.mark.asyncio
    async def test_batch_processes_all_pairs(self, llm_extractor, sample_pairs, valid_response):
        """Batch should process all pairs and return combined triples."""
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(return_value=valid_response)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            source_docs = [f"test:{i}" for i in range(5)]
            triples = await llm_extractor.extract_batch(sample_pairs, source_docs, delay=0.0)

            assert mock_instance.generate.call_count == 5
            assert isinstance(triples, list)

    @pytest.mark.asyncio
    async def test_batch_circuit_breaker_opens(self, llm_extractor, sample_pairs):
        """5 consecutive connection errors should stop the batch."""
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(side_effect=Exception("Connection refused"))
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            pairs = [{"instruction": f"test {i}", "category": "t", "topic": "t"} for i in range(10)]
            source_docs = [f"test:{i}" for i in range(10)]
            triples = await llm_extractor.extract_batch(pairs, source_docs, delay=0.0)

            # Should have stopped after 5 consecutive errors, not processed all 10
            assert mock_instance.generate.call_count == 5
            assert triples == []

    @pytest.mark.asyncio
    async def test_batch_circuit_breaker_resets_on_success(self, llm_extractor, valid_response):
        """A successful extraction should reset the consecutive error counter."""
        responses = [
            Exception("timeout"),
            Exception("timeout"),
            valid_response,  # Success - resets counter
            Exception("timeout"),
            Exception("timeout"),
            valid_response,  # Success - resets counter
        ]

        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(side_effect=responses)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            pairs = [{"instruction": f"test {i}", "category": "t", "topic": "t"} for i in range(6)]
            source_docs = [f"test:{i}" for i in range(6)]
            triples = await llm_extractor.extract_batch(pairs, source_docs, delay=0.0)

            # All 6 should be attempted (circuit breaker never reached 5 consecutive)
            assert mock_instance.generate.call_count == 6

    @pytest.mark.asyncio
    async def test_batch_returns_stats(self, llm_extractor, sample_pairs, valid_response):
        """extract_batch should return triples list and stats can be checked externally."""
        with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.generate = AsyncMock(return_value=valid_response)
            mock_instance.health_check = AsyncMock(return_value=True)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            source_docs = [f"test:{i}" for i in range(5)]
            triples = await llm_extractor.extract_batch(sample_pairs, source_docs, delay=0.0)
            assert isinstance(triples, list)
```

**Step 2: Run to verify failures**

Run: `python3 -m pytest tests/test_extraction.py::TestLLMExtractBatch -v 2>&1 | tail -20`

Expected: FAIL - `extract_batch` is a stub returning `[]` and doesn't use circuit breaker

**Step 3: Commit tests**

```bash
git add tests/test_extraction.py
git commit -m "TEST: Tests para extract_batch() con circuit breaker"
```

---

### Task 6: Implementar extract_batch() - codigo

**Files:**
- Modify: `src/knowledge/extraction/llm_extractor.py`

**Step 1: Implement extract_batch()**

Replace the `extract_batch` stub (lines 72-79):

```python
MAX_CONSECUTIVE_ERRORS = 5

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

        # Progress logging
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
```

**Step 2: Run tests**

Run: `python3 -m pytest tests/test_extraction.py::TestLLMExtractBatch -v`

Expected: ALL PASS

**Step 3: Run all LLM tests**

Run: `python3 -m pytest tests/test_extraction.py -k "LLM" -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/knowledge/extraction/llm_extractor.py
git commit -m "FEAT: Implementar extract_batch() con circuit breaker"
```

---

### Task 7: Integrar LLM en pipeline - tests

**Files:**
- Modify: `tests/test_extraction.py`

**Step 1: Write failing integration tests**

```python
class TestPipelineWithLLM:
    """Integration tests for ExtractionPipeline with LLM extractor."""

    @pytest.fixture
    def tmp_datalake(self, tmp_dir):
        """Create a minimal datalake structure with training pairs."""
        training_dir = tmp_dir / "02-processed" / "training-pairs"
        training_dir.mkdir(parents=True)

        pairs = [
            {
                "instruction": "How to create REST API with FastAPI?",
                "output": "Use FastAPI with Pydantic for validation.",
                "category": "api",
                "topic": "web",
            },
            {
                "instruction": "Implement retry with exponential backoff",
                "output": "Use tenacity library for retries.",
                "category": "patterns",
                "topic": "resilience",
            },
        ]
        jsonl_file = training_dir / "test-pairs.jsonl"
        with open(jsonl_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        return tmp_dir

    @pytest.mark.asyncio
    async def test_pipeline_llm_adds_triples(self, tmp_dir, tmp_datalake):
        """Pipeline with use_llm=True should extract more triples than heuristic alone."""
        llm_response = LLMResponse(
            content=json.dumps({
                "entities": [
                    {"name": "FastAPI", "type": "technology"},
                    {"name": "REST", "type": "concept"},
                ],
                "relations": [
                    {"source": "REST", "target": "FastAPI", "type": "depends_on"},
                ],
            }),
            model="qwen2.5-coder:7b",
            tokens_used=100,
            latency_ms=2000.0,
        )

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_datalake

        try:
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=llm_response)
                mock_instance.health_check = AsyncMock(return_value=True)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                pipeline = ExtractionPipeline(engine=engine, use_llm=True)
                stats = await pipeline.build(force=True)

                assert stats["llm_triples_extracted"] > 0
                assert stats["triples_extracted"] > 0
        finally:
            settings.datalake_path = original_path

    @pytest.mark.asyncio
    async def test_pipeline_graceful_degradation(self, tmp_dir, tmp_datalake):
        """If Ollama is unavailable, pipeline should continue with heuristic only."""
        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_datalake

        try:
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.health_check = AsyncMock(return_value=False)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                pipeline = ExtractionPipeline(engine=engine, use_llm=True)
                stats = await pipeline.build(force=True)

                # Should still work with heuristic
                assert stats["triples_extracted"] > 0
                assert stats.get("llm_triples_extracted", 0) == 0
        finally:
            settings.datalake_path = original_path

    @pytest.mark.asyncio
    async def test_pipeline_stats_separate_llm_count(self, tmp_dir, tmp_datalake):
        """Stats should separately track heuristic vs LLM triples."""
        llm_response = LLMResponse(
            content=json.dumps({
                "entities": [{"name": "X", "type": "concept"}],
                "relations": [],
            }),
            model="qwen2.5-coder:7b",
            tokens_used=50,
            latency_ms=1000.0,
        )

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_datalake

        try:
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=llm_response)
                mock_instance.health_check = AsyncMock(return_value=True)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                pipeline = ExtractionPipeline(engine=engine, use_llm=True)
                stats = await pipeline.build(force=True)

                assert "llm_triples_extracted" in stats
                assert "triples_extracted" in stats
        finally:
            settings.datalake_path = original_path
```

**Step 2: Run to verify failures**

Run: `python3 -m pytest tests/test_extraction.py::TestPipelineWithLLM -v 2>&1 | tail -20`

Expected: FAIL - pipeline doesn't call LLM, no `llm_triples_extracted` key in stats

**Step 3: Commit tests**

```bash
git add tests/test_extraction.py
git commit -m "TEST: Tests de integracion pipeline + LLM extractor"
```

---

### Task 8: Integrar LLM en pipeline - codigo

**Files:**
- Modify: `src/knowledge/extraction/pipeline.py`

**Step 1: Add availability check in build()**

In `build()` method, after line 116 (`state = ...`), add LLM availability check:

```python
# Check LLM availability if requested
llm_available = False
if self.llm_extractor:
    try:
        from src.core.llm_client import LLMClient
        async with LLMClient(model=self.llm_extractor.model) as client:
            llm_available = await client.health_check()
        if llm_available:
            logger.info("llm_extractor_available", model=self.llm_extractor.model)
        else:
            logger.warning("llm_extractor_unavailable", model=self.llm_extractor.model)
    except Exception:
        logger.warning("llm_extractor_check_failed")
self._llm_available = llm_available
```

Add `llm_triples_extracted` to the `_stats` dict initialization in `build()`:

```python
self._stats = {
    "files_processed": 0,
    "pairs_processed": 0,
    "triples_extracted": 0,
    "llm_triples_extracted": 0,
    "errors": 0,
}
```

**Step 2: Modify _extract_from_jsonl() to call LLM**

Replace `_extract_from_jsonl` method:

```python
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
```

Also add `self._llm_available = False` in `__init__`:

```python
def __init__(
    self,
    engine: GraphEngine | None = None,
    use_llm: bool = False,
):
    self.engine = engine or GraphEngine()
    self.heuristic = HeuristicExtractor()
    self.llm_extractor = LLMExtractor() if use_llm else None
    self._llm_available = False
    self._stats = {
        "files_processed": 0,
        "pairs_processed": 0,
        "triples_extracted": 0,
        "llm_triples_extracted": 0,
        "errors": 0,
    }
```

**Step 3: Run integration tests**

Run: `python3 -m pytest tests/test_extraction.py::TestPipelineWithLLM -v`

Expected: ALL PASS

**Step 4: Run ALL tests to verify no regression**

Run: `python3 -m pytest tests/test_extraction.py -v 2>&1 | tail -30`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/knowledge/extraction/pipeline.py
git commit -m "FEAT: Integrar LLM extractor en pipeline con availability check y stats"
```

---

### Task 9: Verificacion final

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v`

Expected: ALL PASS

**Step 2: Run linting**

Run: `python3 -m ruff check src/knowledge/extraction/ tests/test_extraction.py`

Expected: No errors (fix any that appear)

**Step 3: Verify CLI flag works**

Run: `python3 -m src.interfaces.cli graph build --help`

Expected: Shows `--no-llm` flag (already existed, just verify it still works)

**Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "CHORE: Cleanup y verificacion LLM extractor"
```
