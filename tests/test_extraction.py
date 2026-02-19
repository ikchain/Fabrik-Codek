"""Tests for the extraction pipeline."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.core.llm_client import LLMResponse
from src.knowledge.extraction.heuristic import HeuristicExtractor
from src.knowledge.extraction.llm_extractor import LLMExtractor
from src.knowledge.extraction.pipeline import ExtractionPipeline, _parse_multiline_json
from src.knowledge.graph_engine import GraphEngine
from src.knowledge.graph_schema import EntityType, RelationType

from scripts.enrich_auto_captures import (
    load_transcript,
    find_tool_use_message,
    walk_up_for_context,
    determine_confidence,
    enrich_record,
)
from scripts import extract_reasoning as er


@pytest.fixture
def extractor():
    return HeuristicExtractor()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# --- Heuristic Extractor Tests ---


class TestHeuristicExtractor:
    def test_extract_technology(self, extractor):
        pair = {
            "instruction": "How to create a REST API with FastAPI?",
            "output": "Use FastAPI with Pydantic models for validation.",
            "category": "api",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:1")
        tech_names = {t.subject_name for t in triples if t.subject_type == EntityType.TECHNOLOGY}
        assert "fastapi" in tech_names
        assert "pydantic" in tech_names

    def test_extract_pattern(self, extractor):
        pair = {
            "instruction": "Explain hexagonal architecture",
            "output": "Hexagonal architecture uses ports and adapters pattern.",
            "category": "architecture",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:2")
        pattern_names = {t.subject_name for t in triples if t.subject_type == EntityType.PATTERN}
        assert "hexagonal architecture" in pattern_names

    def test_extract_error_type(self, extractor):
        pair = {
            "instruction": "How to handle connection timeout?",
            "output": "Use retry with exponential backoff for connection timeout errors.",
            "category": "debugging",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:3")
        error_names = {t.subject_name for t in triples if t.subject_type == EntityType.ERROR_TYPE}
        assert "timeout error" in error_names or "connection error" in error_names

    def test_extract_strategy(self, extractor):
        pair = {
            "instruction": "How to improve API resilience?",
            "output": "Implement retry with exponential backoff and circuit breaker pattern.",
            "category": "patterns",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:4")
        strategy_names = {t.subject_name for t in triples if t.subject_type == EntityType.STRATEGY}
        assert "retry with backoff" in strategy_names
        assert "circuit breaker" in strategy_names

    def test_extract_fixes_relation(self, extractor):
        pair = {
            "instruction": "Fix connection timeout error",
            "output": "Add retry with exponential backoff to handle timeout errors.",
            "category": "debugging",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:5")
        fixes = [t for t in triples if t.relation_type == RelationType.FIXES]
        assert len(fixes) > 0
        assert any(t.subject_type == EntityType.STRATEGY for t in fixes)
        assert any(t.object_type == EntityType.ERROR_TYPE for t in fixes)

    def test_cooccurrence_relations(self, extractor):
        pair = {
            "instruction": "How to use FastAPI with PostgreSQL?",
            "output": "Connect FastAPI to PostgreSQL using SQLAlchemy ORM.",
            "category": "database",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:6")
        related = [t for t in triples
                   if t.relation_type == RelationType.RELATED_TO
                   and t.subject_type == EntityType.TECHNOLOGY
                   and t.object_type == EntityType.TECHNOLOGY
                   and t.subject_name != t.object_name]
        assert len(related) > 0

    def test_empty_pair(self, extractor):
        pair = {"instruction": "", "output": "", "category": ""}
        triples = extractor.extract_from_pair(pair)
        assert isinstance(triples, list)

    def test_category_linking(self, extractor):
        pair = {
            "instruction": "How to test Angular components?",
            "output": "Use Jest for unit testing Angular components.",
            "category": "testing",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:7")
        category_rels = [
            t for t in triples
            if t.object_type == EntityType.CATEGORY
        ]
        assert len(category_rels) > 0

    def test_extract_from_decision(self, extractor):
        decision = {
            "topic": "Choose database for the project",
            "chosen_option": "PostgreSQL with connection pooling",
            "lesson_learned": "Always use connection pooling for production",
        }
        triples = extractor.extract_from_decision(decision, source_doc="decision:1")
        assert isinstance(triples, list)

    def test_extract_from_learning(self, extractor):
        learning = {
            "topic": "API performance optimization",
            "applicable_to": ["FastAPI", "Django"],
        }
        triples = extractor.extract_from_learning(learning, source_doc="learning:1")
        assert isinstance(triples, list)

    def test_technology_word_boundary(self, extractor):
        """Ensure partial matches are avoided (e.g., 'react' in 'reactive')."""
        pair = {
            "instruction": "What is reactive programming?",
            "output": "Reactive programming uses observables and streams.",
            "category": "patterns",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:8")
        tech_names = {t.subject_name for t in triples if t.subject_type == EntityType.TECHNOLOGY}
        # "react" should NOT match because "reactive" != "react" with word boundaries
        assert "react" not in tech_names

    def test_multiple_patterns(self, extractor):
        pair = {
            "instruction": "DDD with dependency injection",
            "output": "Domain-Driven Design works well with DI containers.",
            "category": "architecture",
        }
        triples = extractor.extract_from_pair(pair, source_doc="test:9")
        pattern_names = {t.subject_name for t in triples if t.subject_type == EntityType.PATTERN}
        assert "domain-driven design" in pattern_names
        assert "dependency injection" in pattern_names


# --- LLM Extractor Tests ---


class TestLLMExtractor:
    def test_stub_returns_empty(self):
        extractor = LLMExtractor()
        # parse a valid response
        triples = extractor._parse_llm_response(
            '{"entities": [{"name": "FastAPI", "type": "technology"}], "relations": []}',
            source_doc="test",
        )
        assert isinstance(triples, list)

    def test_parse_valid_json(self):
        extractor = LLMExtractor()
        response = json.dumps({
            "entities": [
                {"name": "FastAPI", "type": "technology"},
                {"name": "Pydantic", "type": "technology"},
            ],
            "relations": [
                {"source": "FastAPI", "target": "Pydantic", "type": "uses"},
            ],
        })
        triples = extractor._parse_llm_response(response, source_doc="test")
        assert len(triples) == 1
        assert triples[0].relation_type == RelationType.USES

    def test_parse_malformed_json(self):
        extractor = LLMExtractor()
        triples = extractor._parse_llm_response("not json at all", source_doc="test")
        assert triples == []

    def test_parse_partial_json(self):
        extractor = LLMExtractor()
        response = 'Here is the result: {"entities": [], "relations": []} end'
        triples = extractor._parse_llm_response(response, source_doc="test")
        assert isinstance(triples, list)

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
        """Unknown entity types should fallback to TECHNOLOGY for common aliases, CONCEPT for unknown."""
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
        assert triples[0].subject_type == EntityType.TECHNOLOGY
        assert triples[0].object_type == EntityType.CONCEPT

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
            "entities": [
                {"name": "FastAPI", "type": "technology"},
                {"name": "Pydantic", "type": "technology"},
            ],
            "relations": [
                {"source": "", "target": "FastAPI", "type": "uses"},
                {"source": "FastAPI", "target": "  ", "type": "uses"},
                {"source": "FastAPI", "target": "Pydantic", "type": "uses"},
            ],
        })
        triples = extractor._parse_llm_response(response, source_doc="test")
        assert len(triples) == 1
        assert triples[0].subject_name == "fastapi"
        assert triples[0].object_name == "pydantic"

    def test_parse_truncated_json_recovery(self):
        """Truncated JSON should attempt recovery or return []."""
        extractor = LLMExtractor()
        response = '{"entities": [{"name": "FastAPI", "type": "technology"}], "relations": [{"source": "FastAPI", "target": "Py'
        triples = extractor._parse_llm_response(response, source_doc="test")
        assert isinstance(triples, list)

    def test_parse_extra_fields_ignored(self):
        """Extra fields in entities/relations should be ignored without error."""
        extractor = LLMExtractor()
        response = json.dumps({
            "entities": [
                {"name": "FastAPI", "type": "technology", "description": "web framework", "extra": True},
                {"name": "Python", "type": "technology"},
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


# --- LLM extract_from_pair Tests ---


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

    def test_extract_returns_triples(self, llm_extractor, sample_pair, llm_response_valid):
        async def _test():
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=llm_response_valid)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                triples = await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")
                assert len(triples) == 2
                assert all(t.confidence == 0.6 for t in triples)
                names = {(t.subject_name, t.object_name) for t in triples}
                assert ("fastapi", "pydantic") in names

        asyncio.run(_test())

    def test_prompt_uses_instruction_not_output(self, llm_extractor, sample_pair, llm_response_valid):
        async def _test():
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=llm_response_valid)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")
                call_args = mock_instance.generate.call_args
                prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
                assert "REST API with FastAPI" in prompt
                assert "(long code omitted)" not in prompt

        asyncio.run(_test())

    def test_uses_low_temperature(self, llm_extractor, sample_pair, llm_response_valid):
        async def _test():
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=llm_response_valid)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")
                call_args = mock_instance.generate.call_args
                temperature = call_args.kwargs.get("temperature")
                assert temperature is not None
                assert temperature <= 0.2

        asyncio.run(_test())

    def test_llm_returns_garbage(self, llm_extractor, sample_pair):
        async def _test():
            garbage_response = LLMResponse(
                content="I cannot extract entities.",
                model="qwen2.5-coder:7b",
                tokens_used=20,
                latency_ms=1000.0,
            )
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=garbage_response)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                triples = await llm_extractor.extract_from_pair(sample_pair, source_doc="test:1")
                assert triples == []

        asyncio.run(_test())

    def test_empty_instruction_skips_llm(self, llm_extractor):
        async def _test():
            pair = {"instruction": "", "output": "something", "category": "test"}
            triples = await llm_extractor.extract_from_pair(pair, source_doc="test:1")
            assert triples == []

        asyncio.run(_test())


# --- LLM extract_batch Tests ---


class TestLLMExtractBatch:
    """Tests for LLMExtractor.extract_batch() - sequential processing + circuit breaker."""

    @pytest.fixture
    def llm_extractor(self):
        return LLMExtractor(model="qwen2.5-coder:7b")

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

    def test_batch_processes_all_pairs(self, llm_extractor, valid_response):
        import asyncio

        async def _test():
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(return_value=valid_response)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                pairs = [{"instruction": f"How to use tool {i}?", "category": "test", "topic": "tools"} for i in range(5)]
                source_docs = [f"test:{i}" for i in range(5)]
                triples = await llm_extractor.extract_batch(pairs, source_docs, delay=0.0)

                assert mock_instance.generate.call_count == 5
                assert isinstance(triples, list)

        asyncio.run(_test())

    def test_batch_circuit_breaker_opens(self, llm_extractor):
        import asyncio

        async def _test():
            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(side_effect=Exception("Connection refused"))
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                pairs = [{"instruction": f"test {i}", "category": "t", "topic": "t"} for i in range(10)]
                source_docs = [f"test:{i}" for i in range(10)]
                triples = await llm_extractor.extract_batch(pairs, source_docs, delay=0.0)

                # Should have stopped after 5 consecutive errors
                assert mock_instance.generate.call_count == 5
                assert triples == []

        asyncio.run(_test())

    def test_batch_circuit_breaker_resets_on_success(self, llm_extractor, valid_response):
        import asyncio

        async def _test():
            responses = [
                Exception("timeout"),
                Exception("timeout"),
                valid_response,
                Exception("timeout"),
                Exception("timeout"),
                valid_response,
            ]

            with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockClient:
                mock_instance = AsyncMock()
                mock_instance.generate = AsyncMock(side_effect=responses)
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = mock_instance

                pairs = [{"instruction": f"test {i}", "category": "t", "topic": "t"} for i in range(6)]
                source_docs = [f"test:{i}" for i in range(6)]
                triples = await llm_extractor.extract_batch(pairs, source_docs, delay=0.0)

                # All 6 should be attempted (breaker never hit 5 consecutive)
                assert mock_instance.generate.call_count == 6

        asyncio.run(_test())


# --- Pipeline Tests ---


class TestExtractionPipeline:
    def test_pipeline_with_test_data(self, tmp_dir):
        """Test pipeline with synthetic training pairs."""
        # Create datalake structure
        training_dir = tmp_dir / "02-processed" / "training-pairs"
        training_dir.mkdir(parents=True)

        # Write test training pairs
        pairs = [
            {
                "instruction": "How to create REST API with FastAPI?",
                "output": "Use FastAPI with Pydantic for validation and SQLAlchemy for DB.",
                "category": "api",
            },
            {
                "instruction": "Fix connection timeout in PostgreSQL",
                "output": "Use connection pooling and retry with exponential backoff.",
                "category": "debugging",
            },
            {
                "instruction": "Implement hexagonal architecture in Python",
                "output": "Use ports and adapters pattern with dependency injection.",
                "category": "architecture",
            },
        ]
        jsonl_file = training_dir / "test-pairs.jsonl"
        with open(jsonl_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        # Create graph engine in temp dir
        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        # Patch settings to use our temp datalake
        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            import asyncio
            stats = asyncio.run(pipeline.build(force=True))

            assert stats["files_processed"] > 0
            assert stats["pairs_processed"] == 3
            assert stats["triples_extracted"] > 0
            assert stats["errors"] == 0

            # Verify graph has content
            graph_stats = engine.get_stats()
            assert graph_stats["entity_count"] > 0
            assert graph_stats["edge_count"] > 0

            # Verify known entities exist
            assert engine.find_entity_by_name("fastapi") is not None
            assert engine.find_entity_by_name("postgresql") is not None

        finally:
            settings.datalake_path = original_path

    def test_pipeline_incremental(self, tmp_dir):
        """Test that incremental builds skip already processed files."""
        training_dir = tmp_dir / "02-processed" / "training-pairs"
        training_dir.mkdir(parents=True)

        pairs = [{"instruction": "test", "output": "FastAPI is great", "category": "test"}]
        jsonl_file = training_dir / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio

            # First build
            pipeline1 = ExtractionPipeline(engine=engine, use_llm=False)
            stats1 = asyncio.run(pipeline1.build(force=True))
            assert stats1["files_processed"] == 1

            # Second build (incremental) - same files, should skip
            pipeline2 = ExtractionPipeline(engine=engine, use_llm=False)
            stats2 = asyncio.run(pipeline2.build(force=False))
            assert stats2["files_processed"] == 0

        finally:
            settings.datalake_path = original_path

    def test_pipeline_empty_datalake(self, tmp_dir):
        """Test pipeline with nonexistent datalake."""
        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir / "nonexistent"

        try:
            import asyncio
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build())
            assert stats["files_processed"] == 0
            assert stats["errors"] == 0
        finally:
            settings.datalake_path = original_path


# --- Auto-capture Extractor Tests ---


class TestAutoCapture:
    def test_extract_tech_from_file_extension(self, extractor):
        record = {
            "type": "auto_capture",
            "tool": "Edit",
            "project": "myapp",
            "file_modified": "/home/user/project/src/main.py",
            "description": "Edit: old_string -> new_string",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:1")
        tech_names = {t.subject_name for t in triples if t.subject_type == EntityType.TECHNOLOGY}
        assert "python" in tech_names

    def test_extract_tech_from_description(self, extractor):
        record = {
            "type": "auto_capture",
            "tool": "Edit",
            "project": "fabrik-codek",
            "file_modified": "/home/user/project/config.yml",
            "description": "Edit: Added FastAPI endpoint with Pydantic validation",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:2")
        tech_names = {t.subject_name for t in triples if t.subject_type == EntityType.TECHNOLOGY}
        assert "fastapi" in tech_names
        assert "pydantic" in tech_names

    def test_project_uses_technology(self, extractor):
        record = {
            "type": "auto_capture",
            "tool": "Write",
            "project": "myapp",
            "file_modified": "/home/user/project/app.ts",
            "description": "Write: new file",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:3")
        uses_rels = [
            t for t in triples
            if t.relation_type == RelationType.USES
            and t.subject_name == "myapp"
        ]
        assert len(uses_rels) > 0
        assert any(t.object_name == "typescript" for t in uses_rels)

    def test_low_confidence_scores(self, extractor):
        record = {
            "type": "auto_capture",
            "tool": "Edit",
            "project": "test",
            "file_modified": "/app/src/main.py",
            "description": "Edit: simple change",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:4")
        for t in triples:
            assert t.confidence <= 0.5

    def test_empty_record(self, extractor):
        record = {"type": "auto_capture"}
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:5")
        assert isinstance(triples, list)

    def test_cooccurrence_between_ext_and_desc_techs(self, extractor):
        record = {
            "type": "auto_capture",
            "tool": "Edit",
            "project": "test",
            "file_modified": "/app/src/api.py",
            "description": "Edit: Added FastAPI router with PostgreSQL query",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:6")
        cooccur = [
            t for t in triples
            if t.relation_type == RelationType.RELATED_TO
            and t.subject_type == EntityType.TECHNOLOGY
            and t.object_type == EntityType.TECHNOLOGY
            and t.subject_name != t.object_name
        ]
        assert len(cooccur) > 0


# --- _parse_multiline_json Tests ---


class TestParseMultilineJson:
    def test_parse_jsonl(self, tmp_dir):
        f = tmp_dir / "test.jsonl"
        f.write_text(
            '{"type": "auto_capture", "project": "a"}\n'
            '{"type": "auto_capture", "project": "b"}\n'
        )
        records = _parse_multiline_json(f)
        assert len(records) == 2
        assert records[0]["project"] == "a"
        assert records[1]["project"] == "b"

    def test_parse_pretty_printed(self, tmp_dir):
        f = tmp_dir / "test.jsonl"
        f.write_text(
            '{\n'
            '  "type": "auto_capture",\n'
            '  "project": "myapp"\n'
            '}\n'
            '{\n'
            '  "type": "auto_capture",\n'
            '  "project": "fabrik"\n'
            '}\n'
        )
        records = _parse_multiline_json(f)
        assert len(records) == 2
        assert records[0]["project"] == "myapp"
        assert records[1]["project"] == "fabrik"

    def test_parse_empty_file(self, tmp_dir):
        f = tmp_dir / "empty.jsonl"
        f.write_text("")
        records = _parse_multiline_json(f)
        assert records == []

    def test_parse_mixed_valid_invalid(self, tmp_dir):
        f = tmp_dir / "mixed.jsonl"
        f.write_text(
            '{"type": "auto_capture"}\n'
            'not json\n'
            '{"type": "manual"}\n'
        )
        records = _parse_multiline_json(f)
        assert len(records) == 2


# --- Pipeline JSONL Decisions/Learnings Tests ---


class TestPipelineJsonlProcessing:
    def test_pipeline_decisions_jsonl(self, tmp_dir):
        """Test pipeline reads JSONL decisions from 03-metadata."""
        decisions_dir = tmp_dir / "03-metadata" / "decisions"
        decisions_dir.mkdir(parents=True)

        records = [
            {
                "topic": "Database selection",
                "decision": "Use PostgreSQL",
                "chosen_option": "PostgreSQL with connection pooling",
                "reasoning": "Better for relational data",
                "lesson_learned": "Always benchmark before choosing",
            },
        ]
        jsonl_file = decisions_dir / "2026-01-decisions.jsonl"
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))
            assert stats["files_processed"] >= 1
            assert stats["errors"] == 0
            assert engine.find_entity_by_name("postgresql") is not None
        finally:
            settings.datalake_path = original_path

    def test_pipeline_learnings_jsonl(self, tmp_dir):
        """Test pipeline reads JSONL learnings from 03-metadata."""
        learnings_dir = tmp_dir / "03-metadata" / "learnings"
        learnings_dir.mkdir(parents=True)

        records = [
            {
                "topic": "Fine-tuning overfitting prevention",
                "applicable_to": ["pytorch", "tensorflow"],
            },
        ]
        jsonl_file = learnings_dir / "2026-01-learnings.jsonl"
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))
            assert stats["files_processed"] >= 1
            assert stats["errors"] == 0
        finally:
            settings.datalake_path = original_path

    def test_pipeline_auto_captures(self, tmp_dir):
        """Test pipeline processes auto-capture files."""
        auto_dir = tmp_dir / "01-raw" / "code-changes"
        auto_dir.mkdir(parents=True)

        records = [
            {
                "type": "auto_capture",
                "tool": "Edit",
                "project": "myapp",
                "file_modified": "/home/user/project/src/api.py",
                "description": "Edit: Added FastAPI endpoint",
            },
            {
                "type": "auto_capture",
                "tool": "Write",
                "project": "myapp",
                "file_modified": "/home/user/project/src/models.ts",
                "description": "Write: new Angular component",
            },
            {
                "type": "manual_log",
                "tool": "Edit",
                "file_modified": "/home/user/project/test.py",
                "description": "Should be ignored - not auto_capture",
            },
        ]
        cap_file = auto_dir / "2026-01-22_auto-captures.jsonl"
        with open(cap_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))
            assert stats["files_processed"] >= 1
            assert stats["triples_extracted"] > 0
            assert stats["errors"] == 0

            # Verify technologies were extracted
            assert engine.find_entity_by_name("python") is not None or \
                   engine.find_entity_by_name("fastapi") is not None
        finally:
            settings.datalake_path = original_path

    def test_pipeline_auto_captures_pretty_printed(self, tmp_dir):
        """Test pipeline handles pretty-printed JSON auto-captures."""
        auto_dir = tmp_dir / "01-raw" / "code-changes"
        auto_dir.mkdir(parents=True)

        # Simulate jq -n output (pretty-printed)
        content = """{
  "type": "auto_capture",
  "tool": "Edit",
  "project": "fabrik-codek",
  "file_modified": "/home/user/project/src/config.py",
  "description": "Edit: Updated PostgreSQL connection settings"
}
{
  "type": "auto_capture",
  "tool": "Edit",
  "project": "fabrik-codek",
  "file_modified": "/home/user/project/src/app.ts",
  "description": "Edit: Added Angular service"
}"""
        cap_file = auto_dir / "2026-02-01_auto-captures.jsonl"
        cap_file.write_text(content)

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))
            assert stats["files_processed"] >= 1
            assert stats["triples_extracted"] > 0
            assert stats["errors"] == 0
        finally:
            settings.datalake_path = original_path


# --- Enrichment Tests ---


def _make_transcript_jsonl(messages: list[dict]) -> str:
    """Helper: create transcript JSONL from message dicts."""
    return "\n".join(json.dumps(m) for m in messages)


def _build_chain(user_text: str, thinking_text: str, assistant_text: str, tool_name: str = "Edit"):
    """Helper: build a minimal parentUuid chain for testing.

    Returns (messages, tool_use_id) where messages is a list of dicts
    simulating a transcript with user → thinking → text → tool_use chain.
    """
    user_uuid = "user-001"
    thinking_uuid = "think-001"
    text_uuid = "text-001"
    tool_uuid = "tool-001"
    tool_use_id = "toolu_test123"

    messages = [
        {
            "type": "user",
            "uuid": user_uuid,
            "parentUuid": None,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
            "timestamp": "2026-02-06T10:00:00+01:00",
        },
        {
            "type": "assistant",
            "uuid": thinking_uuid,
            "parentUuid": user_uuid,
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": thinking_text}],
            },
            "timestamp": "2026-02-06T10:00:01+01:00",
        },
        {
            "type": "assistant",
            "uuid": text_uuid,
            "parentUuid": thinking_uuid,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
            "timestamp": "2026-02-06T10:00:02+01:00",
        },
        {
            "type": "assistant",
            "uuid": tool_uuid,
            "parentUuid": text_uuid,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": {"file_path": "/home/user/project/src/main.py"},
                    },
                ],
            },
            "timestamp": "2026-02-06T10:00:03+01:00",
        },
    ]
    return messages, tool_use_id


class TestEnrichment:
    def test_load_transcript(self, tmp_dir):
        """Test loading and indexing a transcript file."""
        messages, _ = _build_chain("Fix the bug", "Thinking about it", "I will fix it")
        transcript_file = tmp_dir / "test.jsonl"
        transcript_file.write_text(_make_transcript_jsonl(messages))

        transcript = load_transcript(transcript_file)
        assert len(transcript["messages"]) == 4
        assert len(transcript["by_uuid"]) == 4
        assert "user-001" in transcript["by_uuid"]

    def test_find_tool_use_message(self, tmp_dir):
        """Test finding a tool_use message by its ID."""
        messages, tool_use_id = _build_chain("Fix", "Think", "Do it")
        transcript_file = tmp_dir / "test.jsonl"
        transcript_file.write_text(_make_transcript_jsonl(messages))

        transcript = load_transcript(transcript_file)
        msg = find_tool_use_message(transcript, tool_use_id)
        assert msg is not None
        assert msg["uuid"] == "tool-001"

    def test_find_tool_use_not_found(self, tmp_dir):
        """Test that missing tool_use_id returns None."""
        messages, _ = _build_chain("Fix", "Think", "Do it")
        transcript_file = tmp_dir / "test.jsonl"
        transcript_file.write_text(_make_transcript_jsonl(messages))

        transcript = load_transcript(transcript_file)
        msg = find_tool_use_message(transcript, "nonexistent_id")
        assert msg is None

    def test_extract_reasoning_from_transcript(self, tmp_dir):
        """Test walking up parentUuid to extract thinking and text context."""
        thinking = "The function needs to handle edge cases for null inputs and validate boundaries before processing. This is critical for preventing runtime errors in production."
        user_msg = "Fix the null pointer bug in the validation module"
        assistant_msg = "I will add null checks to the validation function"

        messages, tool_use_id = _build_chain(user_msg, thinking, assistant_msg)
        transcript_file = tmp_dir / "test.jsonl"
        transcript_file.write_text(_make_transcript_jsonl(messages))

        transcript = load_transcript(transcript_file)
        tool_msg = find_tool_use_message(transcript, tool_use_id)
        context = walk_up_for_context(transcript, tool_msg["parentUuid"])

        assert thinking in context["thinking"]
        assert assistant_msg in context["assistant_text"]
        assert user_msg in context["user_prompt"]

    def test_enrich_no_match(self, tmp_dir):
        """Test enrichment when no transcript exists - returns None."""
        record = {
            "type": "auto_capture",
            "timestamp": "2026-02-06T10:00:00+01:00",
            "tool": "Edit",
            "project": "nonexistent-project",
            "file_modified": "/path/to/file.py",
            "description": "Edit: changed something",
        }
        cache = {}
        result = enrich_record(record, cache)
        assert result is None

    def test_enrich_with_transcript_path(self, tmp_dir):
        """Test enrichment via direct transcript_path + tool_use_id."""
        thinking = "The user needs authentication middleware added. I need to check the existing middleware chain and insert the auth check before route handlers. This ensures all protected routes require valid tokens."
        messages, tool_use_id = _build_chain(
            "Add auth middleware",
            thinking,
            "Adding authentication middleware to the chain",
        )
        transcript_file = tmp_dir / "transcript.jsonl"
        transcript_file.write_text(_make_transcript_jsonl(messages))

        record = {
            "type": "auto_capture",
            "timestamp": "2026-02-06T10:00:03+01:00",
            "tool": "Edit",
            "project": "test-project",
            "file_modified": "/home/user/project/src/main.py",
            "description": "Edit: added auth check",
            "transcript_path": str(transcript_file),
            "tool_use_id": tool_use_id,
        }
        cache = {}
        result = enrich_record(record, cache)
        assert result is not None
        assert result["type"] == "enriched_capture"
        assert result["enrichment_confidence"] == "high"
        assert thinking in result["reasoning"]
        assert "Adding authentication" in result["assistant_context"]

    def test_determine_confidence_high(self):
        """Test high confidence when thinking is substantial."""
        context = {"thinking": "A" * 150, "assistant_text": "some text"}
        assert determine_confidence(context) == "high"

    def test_determine_confidence_medium(self):
        """Test medium confidence when only text (no thinking)."""
        context = {"thinking": "", "assistant_text": "A" * 60}
        assert determine_confidence(context) == "medium"

    def test_determine_confidence_low(self):
        """Test low confidence when no substantial content."""
        context = {"thinking": "", "assistant_text": ""}
        assert determine_confidence(context) == "low"

    def test_load_nonexistent_transcript(self):
        """Test loading a transcript that doesn't exist."""
        transcript = load_transcript("/nonexistent/path.jsonl")
        assert transcript["by_uuid"] == {}
        assert transcript["messages"] == []


class TestHeuristicReasoningBoost:
    def test_confidence_boost_high(self, extractor):
        """Test that enriched records with high confidence get boosted scores."""
        record = {
            "type": "enriched_capture",
            "tool": "Edit",
            "project": "myapp",
            "file_modified": "/home/user/project/src/main.py",
            "description": "Edit: old -> new",
            "reasoning": "Using retry with backoff pattern to handle connection timeouts in the API layer",
            "enrichment_confidence": "high",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="enriched:1")

        # All triples should have confidence >= 0.6
        for t in triples:
            assert t.confidence >= 0.6, f"Triple {t.subject_name} has low confidence {t.confidence}"

    def test_confidence_boost_medium(self, extractor):
        """Test medium confidence boost."""
        record = {
            "type": "enriched_capture",
            "tool": "Edit",
            "project": "myapp",
            "file_modified": "/home/user/project/src/main.py",
            "description": "Edit: old -> new",
            "enrichment_confidence": "medium",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="enriched:2")

        for t in triples:
            assert t.confidence >= 0.55, f"Triple {t.subject_name} has low confidence {t.confidence}"

    def test_no_boost_without_enrichment(self, extractor):
        """Test that non-enriched records keep low confidence."""
        record = {
            "type": "auto_capture",
            "tool": "Edit",
            "project": "test",
            "file_modified": "/app/src/main.py",
            "description": "Edit: simple change",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="auto:10")
        for t in triples:
            assert t.confidence <= 0.5

    def test_reasoning_extracts_more_entities(self, extractor):
        """Test that reasoning text is used for entity extraction."""
        # Without reasoning - only file extension tech
        record_basic = {
            "type": "auto_capture",
            "tool": "Edit",
            "project": "test",
            "file_modified": "/app/src/main.py",
            "description": "Edit: updated config",
        }
        triples_basic = extractor.extract_from_auto_capture(record_basic, source_doc="test:b")
        techs_basic = {t.subject_name for t in triples_basic if t.subject_type == EntityType.TECHNOLOGY}

        # With reasoning mentioning FastAPI and Pydantic
        record_enriched = {
            "type": "enriched_capture",
            "tool": "Edit",
            "project": "test",
            "file_modified": "/app/src/main.py",
            "description": "Edit: updated config",
            "reasoning": "Need to configure FastAPI with Pydantic validation for the new endpoint",
            "enrichment_confidence": "high",
        }
        triples_enriched = extractor.extract_from_auto_capture(record_enriched, source_doc="test:e")
        techs_enriched = {t.subject_name for t in triples_enriched if t.subject_type == EntityType.TECHNOLOGY}

        # Enriched should find more technologies from reasoning
        assert "fastapi" in techs_enriched
        assert "pydantic" in techs_enriched
        assert len(techs_enriched) > len(techs_basic)

    def test_reasoning_extracts_strategies(self, extractor):
        """Test that strategies in reasoning text are extracted."""
        record = {
            "type": "enriched_capture",
            "tool": "Edit",
            "project": "test",
            "file_modified": "/app/src/api.py",
            "description": "Edit: added error handling",
            "reasoning": "Implementing retry with exponential backoff and circuit breaker to handle transient failures",
            "enrichment_confidence": "high",
        }
        triples = extractor.extract_from_auto_capture(record, source_doc="test:s")
        strategy_names = {t.subject_name for t in triples if t.subject_type == EntityType.STRATEGY}
        assert "retry with backoff" in strategy_names
        assert "circuit breaker" in strategy_names


class TestPipelineEnrichedCaptures:
    def test_pipeline_processes_enriched(self, tmp_dir):
        """Test that pipeline processes enriched captures from enriched/ dir."""
        auto_dir = tmp_dir / "01-raw" / "code-changes"
        enriched_dir = auto_dir / "enriched"
        enriched_dir.mkdir(parents=True)

        records = [
            {
                "type": "enriched_capture",
                "tool": "Edit",
                "project": "myapp",
                "file_modified": "/home/user/project/src/api.py",
                "description": "Edit: Added FastAPI endpoint",
                "reasoning": "Adding a new REST endpoint with Pydantic validation for user registration",
                "assistant_context": "I will add the registration endpoint",
                "user_prompt": "Add user registration",
                "enrichment_confidence": "high",
                "tags": ["auto-captured", "enriched"],
            },
        ]
        enr_file = enriched_dir / "2026-02-06_enriched.jsonl"
        with open(enr_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio
            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))
            assert stats["files_processed"] >= 1
            assert stats["triples_extracted"] > 0
            assert stats["errors"] == 0

            # Verify technologies were extracted with boosted confidence
            assert engine.find_entity_by_name("fastapi") is not None
            assert engine.find_entity_by_name("pydantic") is not None
        finally:
            settings.datalake_path = original_path

    def test_pipeline_enriched_incremental(self, tmp_dir):
        """Test incremental build skips already processed enriched files."""
        auto_dir = tmp_dir / "01-raw" / "code-changes"
        enriched_dir = auto_dir / "enriched"
        enriched_dir.mkdir(parents=True)

        records = [
            {
                "type": "enriched_capture",
                "tool": "Edit",
                "project": "test",
                "file_modified": "/app/main.py",
                "description": "Edit: test",
                "reasoning": "Testing incremental builds with enriched data",
                "enrichment_confidence": "high",
            },
        ]
        enr_file = enriched_dir / "2026-02-06_enriched.jsonl"
        with open(enr_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        from src.config import settings
        original_path = settings.datalake_path
        settings.datalake_path = tmp_dir

        try:
            import asyncio

            # First build
            pipeline1 = ExtractionPipeline(engine=engine, use_llm=False)
            stats1 = asyncio.run(pipeline1.build(force=True))
            assert stats1["files_processed"] >= 1

            # Second build (incremental) - should skip
            pipeline2 = ExtractionPipeline(engine=engine, use_llm=False)
            stats2 = asyncio.run(pipeline2.build(force=False))
            assert stats2["files_processed"] == 0
        finally:
            settings.datalake_path = original_path


# --- Pipeline + LLM Integration Tests ---


class TestPipelineWithLLM:
    """Integration tests for ExtractionPipeline with LLM extractor."""

    @pytest.fixture
    def tmp_datalake(self, tmp_dir):
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

    def test_pipeline_llm_adds_triples(self, tmp_dir, tmp_datalake):
        import asyncio

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

        async def _test():
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

        asyncio.run(_test())

    def test_pipeline_graceful_degradation(self, tmp_dir, tmp_datalake):
        import asyncio

        async def _test():
            graph_dir = tmp_dir / "graphdb"
            engine = GraphEngine(data_dir=graph_dir)

            from src.config import settings
            original_path = settings.datalake_path
            settings.datalake_path = tmp_datalake

            try:
                with patch("src.knowledge.extraction.llm_extractor.LLMClient") as MockExtractor, \
                     patch("src.knowledge.extraction.pipeline.LLMClient") as MockPipeline:
                    mock_instance = AsyncMock()
                    mock_instance.health_check = AsyncMock(return_value=False)
                    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                    mock_instance.__aexit__ = AsyncMock(return_value=False)
                    MockExtractor.return_value = mock_instance
                    MockPipeline.return_value = mock_instance

                    pipeline = ExtractionPipeline(engine=engine, use_llm=True)
                    stats = await pipeline.build(force=True)

                    # Heuristic still works
                    assert stats["triples_extracted"] > 0
                    # LLM didn't contribute
                    assert stats.get("llm_triples_extracted", 0) == 0
            finally:
                settings.datalake_path = original_path

        asyncio.run(_test())

    def test_pipeline_stats_include_llm_count(self, tmp_dir, tmp_datalake):
        import asyncio

        llm_response = LLMResponse(
            content=json.dumps({
                "entities": [{"name": "X", "type": "concept"}],
                "relations": [],
            }),
            model="qwen2.5-coder:7b",
            tokens_used=50,
            latency_ms=1000.0,
        )

        async def _test():
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

        asyncio.run(_test())


# --- Inline Enrichment (extract_reasoning.py) Tests ---


class TestExtractReasoning:
    def test_load_transcript(self, tmp_dir):
        """Test standalone transcript loading."""
        messages, _ = _build_chain("Fix bug", "Need to check inputs", "Fixing now")
        tf = tmp_dir / "t.jsonl"
        tf.write_text(_make_transcript_jsonl(messages))

        by_uuid, msgs = er.load_transcript(str(tf))
        assert len(by_uuid) == 4
        assert len(msgs) == 4

    def test_load_nonexistent(self):
        """Test loading nonexistent transcript returns empty."""
        by_uuid, msgs = er.load_transcript("/nonexistent/path.jsonl")
        assert by_uuid == {}
        assert msgs == []

    def test_find_tool_use(self, tmp_dir):
        """Test finding tool_use by ID."""
        messages, tool_use_id = _build_chain("Fix", "Think", "Do")
        tf = tmp_dir / "t.jsonl"
        tf.write_text(_make_transcript_jsonl(messages))

        _, msgs = er.load_transcript(str(tf))
        msg = er.find_tool_use_message(msgs, tool_use_id)
        assert msg is not None
        assert msg["uuid"] == "tool-001"

    def test_walk_up_extracts_all(self, tmp_dir):
        """Test walking up chain extracts thinking, text, and user prompt."""
        thinking = "The function crashes on null input because the validation step is missing entirely from the pipeline"
        user_msg = "Fix the null pointer"
        assistant_msg = "Adding null validation"

        messages, tool_use_id = _build_chain(user_msg, thinking, assistant_msg)
        tf = tmp_dir / "t.jsonl"
        tf.write_text(_make_transcript_jsonl(messages))

        by_uuid, msgs = er.load_transcript(str(tf))
        tool_msg = er.find_tool_use_message(msgs, tool_use_id)
        context = er.walk_up_for_context(by_uuid, tool_msg["parentUuid"])

        assert thinking in context["thinking"]
        assert assistant_msg in context["assistant_text"]
        assert user_msg in context["user_prompt"]

    def test_confidence_levels(self):
        """Test confidence determination."""
        assert er.determine_confidence({"thinking": "A" * 150, "assistant_text": ""}) == "high"
        assert er.determine_confidence({"thinking": "", "assistant_text": "A" * 60}) == "medium"
        assert er.determine_confidence({"thinking": "", "assistant_text": ""}) == "low"

    def test_main_writes_enriched_file(self, tmp_dir):
        """Test the full main() flow writes an enriched JSONL record."""
        thinking = "Need to add proper error handling for the database connection layer because timeouts cause silent data corruption"
        messages, tool_use_id = _build_chain("Add error handling", thinking, "Adding try/except")
        tf = tmp_dir / "transcript.jsonl"
        tf.write_text(_make_transcript_jsonl(messages))

        output_dir = tmp_dir / "enriched"

        import subprocess
        result = subprocess.run(
            [
                "python3", str(Path(__file__).resolve().parent.parent / "scripts" / "extract_reasoning.py"),
                "--transcript-path", str(tf),
                "--tool-use-id", tool_use_id,
                "--timestamp", "2026-02-06T10:00:00+01:00",
                "--tool", "Edit",
                "--project", "test-project",
                "--file-modified", "/app/src/main.py",
                "--description", "Edit: added error handling",
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

        # Verify enriched file was written
        enriched_files = list(output_dir.glob("*.jsonl"))
        assert len(enriched_files) == 1

        with open(enriched_files[0]) as f:
            record = json.loads(f.readline())

        assert record["type"] == "enriched_capture"
        assert record["enrichment_confidence"] == "high"
        assert thinking in record["reasoning"]
        assert record["project"] == "test-project"
        assert "inline" in record["tags"]

    def test_main_silent_on_missing_transcript(self, tmp_dir):
        """Test that script exits 0 when transcript doesn't exist."""
        import subprocess
        result = subprocess.run(
            [
                "python3", str(Path(__file__).resolve().parent.parent / "scripts" / "extract_reasoning.py"),
                "--transcript-path", "/nonexistent/path.jsonl",
                "--tool-use-id", "toolu_fake",
                "--timestamp", "2026-02-06T10:00:00+01:00",
                "--tool", "Edit",
                "--project", "test",
                "--file-modified", "/app/main.py",
                "--description", "Edit: test",
                "--output-dir", str(tmp_dir / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        # No output file should be created
        assert not (tmp_dir / "out").exists()


# --- TranscriptExtractor Tests ---


class TestTranscriptExtractor:
    """Tests for TranscriptExtractor - thinking block extraction from session transcripts."""

    def _make_transcript_line(self, thinking_text: str) -> str:
        """Create a JSONL line with an assistant message containing a thinking block."""
        msg = {
            "type": "assistant",
            "uuid": "asst-001",
            "parentUuid": "user-001",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking_text},
                    {"type": "text", "text": "Here is the response."},
                ],
            },
        }
        return json.dumps(msg)

    def _make_user_line(self, text: str) -> str:
        """Create a JSONL line with a user message."""
        msg = {
            "type": "user",
            "uuid": "user-001",
            "parentUuid": None,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        }
        return json.dumps(msg)

    def _make_progress_line(self) -> str:
        """Create a non-message JSONL line (progress type)."""
        msg = {"type": "progress", "uuid": "prog-001", "data": {"percent": 50}}
        return json.dumps(msg)

    def test_extract_thinking_blocks_from_transcript(self, tmp_dir):
        """Parse transcript with thinking block mentioning FastAPI + PostgreSQL -> triples found."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        thinking_text = (
            "I need to configure the FastAPI application with a PostgreSQL database connection. "
            "The user wants a REST endpoint that queries the database and returns results with "
            "proper error handling and validation using Pydantic models for input and output."
        )
        lines = [
            self._make_user_line("Set up the API"),
            self._make_transcript_line(thinking_text),
        ]
        transcript_file = tmp_dir / "test.jsonl"
        transcript_file.write_text("\n".join(lines) + "\n")

        extractor = TranscriptExtractor()
        triples = extractor.extract_from_transcript(transcript_file)

        assert len(triples) > 0
        tech_names = {t.subject_name for t in triples if t.subject_type == EntityType.TECHNOLOGY}
        assert "fastapi" in tech_names
        assert "postgresql" in tech_names

    def test_filter_short_thinking_blocks(self, tmp_dir):
        """Thinking block < 100 chars -> 0 triples."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        short_thinking = "Just a short thought."
        lines = [self._make_transcript_line(short_thinking)]
        transcript_file = tmp_dir / "short.jsonl"
        transcript_file.write_text("\n".join(lines) + "\n")

        extractor = TranscriptExtractor()
        triples = extractor.extract_from_transcript(transcript_file)
        assert triples == []

    def test_confidence_level(self, tmp_dir):
        """All triples from transcript extraction have confidence 0.65."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        thinking_text = (
            "I need to set up FastAPI with PostgreSQL for the backend service. "
            "The connection pooling strategy will use SQLAlchemy async engine with proper "
            "timeout configuration and retry with exponential backoff for transient failures."
        )
        lines = [self._make_transcript_line(thinking_text)]
        transcript_file = tmp_dir / "conf.jsonl"
        transcript_file.write_text("\n".join(lines) + "\n")

        extractor = TranscriptExtractor()
        triples = extractor.extract_from_transcript(transcript_file)

        assert len(triples) > 0
        for t in triples:
            assert t.confidence == 0.65, f"Triple {t.subject_name}->{t.object_name} has confidence {t.confidence}, expected 0.65"

    def test_empty_transcript(self, tmp_dir):
        """Empty file -> []."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        transcript_file = tmp_dir / "empty.jsonl"
        transcript_file.write_text("")

        extractor = TranscriptExtractor()
        triples = extractor.extract_from_transcript(transcript_file)
        assert triples == []

    def test_malformed_jsonl(self, tmp_dir):
        """Corrupted lines skipped, valid ones parsed."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        thinking_text = (
            "We need to implement the Docker container orchestration with Kubernetes. "
            "The deployment requires proper health checks, resource limits, and horizontal "
            "pod autoscaling based on CPU and memory utilization metrics from Prometheus."
        )
        lines = [
            "not valid json at all{{{",
            self._make_transcript_line(thinking_text),
            '{"broken": true, "missing_end',
        ]
        transcript_file = tmp_dir / "malformed.jsonl"
        transcript_file.write_text("\n".join(lines) + "\n")

        extractor = TranscriptExtractor()
        triples = extractor.extract_from_transcript(transcript_file)

        assert len(triples) > 0
        tech_names = {t.subject_name for t in triples if t.subject_type == EntityType.TECHNOLOGY}
        assert "docker" in tech_names or "kubernetes" in tech_names

    def test_project_filter(self, tmp_dir):
        """scan_all_transcripts with project_filter only processes matching dirs."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        thinking_text = (
            "Implementing the Angular component with RxJS observables for real-time data streaming. "
            "The component needs proper subscription management and cleanup to avoid memory leaks "
            "when navigating between routes in the single-page application architecture."
        )

        # Create matching dir with transcript
        matching_dir = tmp_dir / "myorg-project"
        matching_dir.mkdir()
        t1 = matching_dir / "session.jsonl"
        t1.write_text(self._make_transcript_line(thinking_text) + "\n")

        # Create non-matching dir with transcript (should be ignored)
        other_dir = tmp_dir / "other-project"
        other_dir.mkdir()
        t2 = other_dir / "session.jsonl"
        t2.write_text(self._make_transcript_line(thinking_text) + "\n")

        extractor = TranscriptExtractor()
        triples, stats = extractor.scan_all_transcripts(tmp_dir, project_filter="myorg")

        assert len(triples) > 0
        assert stats["transcripts_scanned"] == 1

    def test_scan_stats(self, tmp_dir):
        """Stats fields are correct: found=2, processed=1 for one long + one short thinking block."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        long_thinking = (
            "I need to implement connection pooling for the PostgreSQL database. "
            "The current setup creates a new connection for each request which is "
            "causing performance issues under load. Using SQLAlchemy async engine."
        )
        short_thinking = "Quick fix."

        quantum_dir = tmp_dir / "test-project"
        quantum_dir.mkdir()
        transcript_file = quantum_dir / "session.jsonl"
        lines = [
            self._make_transcript_line(long_thinking),
            self._make_transcript_line(short_thinking),
        ]
        transcript_file.write_text("\n".join(lines) + "\n")

        extractor = TranscriptExtractor()
        triples, stats = extractor.scan_all_transcripts(tmp_dir)

        assert stats["transcripts_scanned"] == 1
        assert stats["thinking_blocks_found"] == 2
        assert stats["thinking_blocks_processed"] == 1
        assert stats["triples_extracted"] == len(triples)
        assert stats["errors"] == 0

    def test_scan_empty_dir(self, tmp_dir):
        """Empty dir -> ([], stats with 0s)."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        empty_dir = tmp_dir / "empty-scan"
        empty_dir.mkdir()

        extractor = TranscriptExtractor()
        triples, stats = extractor.scan_all_transcripts(empty_dir)

        assert triples == []
        assert stats["transcripts_scanned"] == 0
        assert stats["thinking_blocks_found"] == 0
        assert stats["thinking_blocks_processed"] == 0
        assert stats["triples_extracted"] == 0
        assert stats["errors"] == 0

    def test_scan_nonexistent_dir(self):
        """Nonexistent dir -> ([], stats with 0s)."""
        from src.knowledge.extraction.transcript_extractor import TranscriptExtractor

        extractor = TranscriptExtractor()
        triples, stats = extractor.scan_all_transcripts(Path("/nonexistent/path"))

        assert triples == []
        assert stats["transcripts_scanned"] == 0
        assert stats["thinking_blocks_found"] == 0
        assert stats["thinking_blocks_processed"] == 0
        assert stats["triples_extracted"] == 0
        assert stats["errors"] == 0


# --- Pipeline + TranscriptExtractor Integration Tests ---


class TestPipelineWithTranscripts:
    """Tests for ExtractionPipeline with include_transcripts flag."""

    def _make_transcript_line(self, thinking_text: str) -> str:
        """Create a JSONL line with an assistant message containing a thinking block."""
        msg = {
            "type": "assistant",
            "uuid": "test-uuid-001",
            "parentUuid": "test-parent-001",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking_text},
                    {"type": "text", "text": "Some response."},
                ],
            },
        }
        return json.dumps(msg)

    def test_pipeline_with_transcripts(self, tmp_dir):
        """Pipeline with include_transcripts=True processes transcript thinking blocks."""
        # Create datalake structure
        datalake = tmp_dir / "datalake"
        datalake.mkdir()

        # Create transcripts dir with project subdir
        transcripts_dir = tmp_dir / "transcripts"
        quantum_proj = transcripts_dir / "-home-user-projects-myapp"
        quantum_proj.mkdir(parents=True)

        thinking_text = (
            "I need to configure the FastAPI application with a PostgreSQL database connection. "
            "The user wants a REST endpoint that queries the database and returns results with "
            "proper error handling and validation using Pydantic models for input and output."
        )
        transcript_file = quantum_proj / "session.jsonl"
        transcript_file.write_text(self._make_transcript_line(thinking_text) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        with patch("src.knowledge.extraction.pipeline.settings") as mock_settings:
            mock_settings.datalake_path = datalake

            pipeline = ExtractionPipeline(engine=engine, use_llm=False, include_transcripts=True)
            stats = asyncio.run(pipeline.build(force=True, transcripts_dir=transcripts_dir))

            assert stats["transcript_triples_extracted"] > 0
            graph_stats = engine.get_stats()
            assert graph_stats["entity_count"] > 0

    def test_pipeline_without_transcripts_flag(self, tmp_dir):
        """Pipeline default (include_transcripts=False) -> transcript_extractor is None."""
        datalake = tmp_dir / "datalake"
        datalake.mkdir()

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        with patch("src.knowledge.extraction.pipeline.settings") as mock_settings:
            mock_settings.datalake_path = datalake

            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            assert pipeline.transcript_extractor is None

            stats = asyncio.run(pipeline.build(force=True))
            assert stats["transcript_triples_extracted"] == 0

    def test_pipeline_transcript_stats(self, tmp_dir):
        """Stats include transcript_triples_extracted field with correct count."""
        datalake = tmp_dir / "datalake"
        datalake.mkdir()

        transcripts_dir = tmp_dir / "transcripts"
        quantum_proj = transcripts_dir / "-home-user-projects-test"
        quantum_proj.mkdir(parents=True)

        thinking_text = (
            "We need to implement the Docker container orchestration with Kubernetes. "
            "The deployment requires proper health checks, resource limits, and horizontal "
            "pod autoscaling based on CPU and memory utilization metrics from Prometheus."
        )
        transcript_file = quantum_proj / "session.jsonl"
        transcript_file.write_text(self._make_transcript_line(thinking_text) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        with patch("src.knowledge.extraction.pipeline.settings") as mock_settings:
            mock_settings.datalake_path = datalake

            pipeline = ExtractionPipeline(engine=engine, use_llm=False, include_transcripts=True)
            stats = asyncio.run(pipeline.build(force=True, transcripts_dir=transcripts_dir))

            assert "transcript_triples_extracted" in stats
            assert stats["transcript_triples_extracted"] > 0
            # Verify the count matches what TranscriptExtractor would produce
            from src.knowledge.extraction.transcript_extractor import TranscriptExtractor
            extractor = TranscriptExtractor()
            expected_triples = extractor.extract_from_transcript(transcript_file)
            assert stats["transcript_triples_extracted"] == len(expected_triples)

    def test_pipeline_incremental_transcripts(self, tmp_dir):
        """Second build(force=False) skips already-processed transcripts."""
        datalake = tmp_dir / "datalake"
        datalake.mkdir()

        transcripts_dir = tmp_dir / "transcripts"
        quantum_proj = transcripts_dir / "-home-user-projects-incremental"
        quantum_proj.mkdir(parents=True)

        thinking_text = (
            "I need to set up FastAPI with PostgreSQL for the backend service. "
            "The connection pooling strategy will use SQLAlchemy async engine with proper "
            "timeout configuration and retry with exponential backoff for transient failures."
        )
        transcript_file = quantum_proj / "session.jsonl"
        transcript_file.write_text(self._make_transcript_line(thinking_text) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        with patch("src.knowledge.extraction.pipeline.settings") as mock_settings:
            mock_settings.datalake_path = datalake

            pipeline = ExtractionPipeline(engine=engine, use_llm=False, include_transcripts=True)

            # First build with force=True
            stats1 = asyncio.run(pipeline.build(force=True, transcripts_dir=transcripts_dir))
            assert stats1["transcript_triples_extracted"] > 0

            # Second build with force=False on same engine (incremental)
            stats2 = asyncio.run(pipeline.build(force=False, transcripts_dir=transcripts_dir))
            assert stats2["transcript_triples_extracted"] == 0


# --- Graph Completion Tests ---


class TestGraphCompletion:
    """Tests for GraphEngine.complete() - transitive inference."""

    def _add_chain(self, engine, a_name, b_name, c_name, relation_type):
        """Helper: add A->B->C chain of given relation_type."""
        from src.knowledge.graph_schema import Entity, Relation, make_entity_id
        a_id = make_entity_id("technology", a_name)
        b_id = make_entity_id("technology", b_name)
        c_id = make_entity_id("technology", c_name)

        engine.add_entity(Entity(id=a_id, name=a_name, entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=b_id, name=b_name, entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=c_id, name=c_name, entity_type=EntityType.TECHNOLOGY))

        engine.add_relation(Relation(
            source_id=a_id, target_id=b_id,
            relation_type=relation_type, weight=0.7,
            source_docs=["test:1"],
        ))
        engine.add_relation(Relation(
            source_id=b_id, target_id=c_id,
            relation_type=relation_type, weight=0.6,
            source_docs=["test:2"],
        ))
        return a_id, b_id, c_id

    def test_complete_depends_on(self, tmp_dir):
        """A->B->C with DEPENDS_ON -> infers A->C."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "fastapi", "starlette", "uvicorn", RelationType.DEPENDS_ON)

        stats = engine.complete()

        assert engine._graph.has_edge(a_id, c_id)
        edge = engine._graph.edges[a_id, c_id]
        assert edge["relation_type"] == RelationType.DEPENDS_ON.value
        assert stats["inferred_count"] > 0
        assert stats["depends_on_inferred"] > 0

    def test_complete_part_of(self, tmp_dir):
        """A->B->C with PART_OF -> infers A->C."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "router", "api-layer", "backend", RelationType.PART_OF)

        stats = engine.complete()

        assert engine._graph.has_edge(a_id, c_id)
        edge = engine._graph.edges[a_id, c_id]
        assert edge["relation_type"] == RelationType.PART_OF.value
        assert stats["part_of_inferred"] > 0

    def test_complete_no_duplicate(self, tmp_dir):
        """If A->C already exists, complete() does NOT create a duplicate or modify it."""
        from src.knowledge.graph_schema import Entity, Relation, make_entity_id
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "x", "y", "z", RelationType.DEPENDS_ON)

        # Add direct A->C edge with weight=0.8
        engine.add_relation(Relation(
            source_id=a_id, target_id=c_id,
            relation_type=RelationType.DEPENDS_ON, weight=0.8,
            source_docs=["direct:1"],
        ))
        original_weight = engine._graph.edges[a_id, c_id]["weight"]

        stats = engine.complete()

        # Edge should still exist with original weight (not overwritten)
        assert engine._graph.edges[a_id, c_id]["weight"] == original_weight
        assert stats["inferred_count"] == 0

    def test_complete_different_types_no_inference(self, tmp_dir):
        """A depends_on B, B part_of C -> NO inference (different relation types)."""
        from src.knowledge.graph_schema import Entity, Relation, make_entity_id
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")

        a_id = make_entity_id("technology", "svc-a")
        b_id = make_entity_id("technology", "svc-b")
        c_id = make_entity_id("technology", "svc-c")

        engine.add_entity(Entity(id=a_id, name="svc-a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=b_id, name="svc-b", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=c_id, name="svc-c", entity_type=EntityType.TECHNOLOGY))

        engine.add_relation(Relation(
            source_id=a_id, target_id=b_id,
            relation_type=RelationType.DEPENDS_ON, weight=0.7,
        ))
        engine.add_relation(Relation(
            source_id=b_id, target_id=c_id,
            relation_type=RelationType.PART_OF, weight=0.6,
        ))

        stats = engine.complete()

        assert not engine._graph.has_edge(a_id, c_id)
        assert stats["inferred_count"] == 0

    def test_complete_uses_not_transitive(self, tmp_dir):
        """A uses B, B uses C -> NO inference (USES is not transitive)."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "app", "lib", "core", RelationType.USES)

        stats = engine.complete()

        assert not engine._graph.has_edge(a_id, c_id)
        assert stats["inferred_count"] == 0

    def test_complete_stats(self, tmp_dir):
        """Stats return correct counts for multiple inferences."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        # Chain 1: DEPENDS_ON
        self._add_chain(engine, "a1", "b1", "c1", RelationType.DEPENDS_ON)
        # Chain 2: PART_OF
        self._add_chain(engine, "a2", "b2", "c2", RelationType.PART_OF)

        stats = engine.complete()

        assert stats["inferred_count"] == 2
        assert stats["depends_on_inferred"] == 1
        assert stats["part_of_inferred"] == 1

    def test_complete_inferred_metadata(self, tmp_dir):
        """Inferred edges have weight=0.3, source_docs=["inferred:transitive"], metadata={"inferred": true}."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "p", "q", "r", RelationType.DEPENDS_ON)

        engine.complete()

        edge = engine._graph.edges[a_id, c_id]
        assert edge["weight"] == 0.3
        assert edge["source_docs"] == ["inferred:transitive"]
        assert edge["metadata"] == {"inferred": True}

    def test_complete_empty_graph(self, tmp_dir):
        """Empty graph -> stats with all 0s, no errors."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")

        stats = engine.complete()

        assert stats["inferred_count"] == 0
        assert stats["depends_on_inferred"] == 0
        assert stats["part_of_inferred"] == 0

    def test_pipeline_runs_completion(self, tmp_dir):
        """Pipeline build() runs completion, stats include inferred_triples."""
        import asyncio
        from unittest.mock import patch
        from src.knowledge.extraction.pipeline import ExtractionPipeline

        # Create datalake with training pair that creates a DEPENDS_ON chain
        training_dir = tmp_dir / "datalake" / "02-processed" / "training-pairs"
        training_dir.mkdir(parents=True)
        pairs = [
            {
                "instruction": "FastAPI depends on Starlette which depends on uvicorn",
                "output": "FastAPI is built on top of Starlette, and Starlette depends on uvicorn as ASGI server.",
                "category": "api",
            },
        ]
        jsonl_file = training_dir / "test-deps.jsonl"
        with open(jsonl_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        with patch("src.knowledge.extraction.pipeline.settings") as mock_settings:
            mock_settings.datalake_path = tmp_dir / "datalake"

            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))

            # Pipeline should have run completion
            assert "inferred_triples" in stats
            assert isinstance(stats["inferred_triples"], int)

    def test_complete_idempotent(self, tmp_dir):
        """Running complete() twice produces no additional edges."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        self._add_chain(engine, "a", "b", "c", RelationType.DEPENDS_ON)

        stats1 = engine.complete()
        stats2 = engine.complete()

        assert stats1["inferred_count"] == 1
        assert stats2["inferred_count"] == 0
