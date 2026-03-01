"""Tests for HybridRAGEngine."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.graph_engine import GraphEngine
from src.knowledge.graph_schema import EntityType, RelationType, Triple
from src.knowledge.hybrid_rag import HybridRAGEngine


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def graph_engine(tmp_dir):
    """Graph engine with test data."""
    engine = GraphEngine(data_dir=tmp_dir / "graphdb")

    # Add test entities and relations
    engine.ingest_triple(
        Triple(
            subject_name="FastAPI",
            subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic",
            object_type=EntityType.TECHNOLOGY,
            source_doc="doc1",
        )
    )
    engine.ingest_triple(
        Triple(
            subject_name="FastAPI",
            subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Starlette",
            object_type=EntityType.TECHNOLOGY,
            source_doc="doc2",
        )
    )
    engine.ingest_triple(
        Triple(
            subject_name="retry with backoff",
            subject_type=EntityType.STRATEGY,
            relation_type=RelationType.FIXES,
            object_name="connection error",
            object_type=EntityType.ERROR_TYPE,
            source_doc="doc3",
        )
    )
    engine.ingest_triple(
        Triple(
            subject_name="hexagonal architecture",
            subject_type=EntityType.PATTERN,
            relation_type=RelationType.RELATED_TO,
            object_name="domain-driven design",
            object_type=EntityType.CONCEPT,
            source_doc="doc4",
        )
    )

    return engine


@pytest.fixture
def mock_rag_engine():
    """Mock RAGEngine for testing without LanceDB."""
    rag = MagicMock()
    rag._init = AsyncMock()
    rag.close = AsyncMock()
    rag.retrieve = AsyncMock(
        return_value=[
            {
                "text": "FastAPI is a modern web framework",
                "source": "file1",
                "category": "api",
                "score": 0.9,
            },
            {
                "text": "Pydantic provides data validation",
                "source": "file2",
                "category": "api",
                "score": 0.8,
            },
            {
                "text": "Use SQLAlchemy for ORM",
                "source": "file3",
                "category": "database",
                "score": 0.7,
            },
        ]
    )
    return rag


# --- RRF Fusion Tests ---


class TestRRFFusion:
    def test_rrf_vector_only(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        vector_results = [
            {"text": "result A", "source": "f1", "category": "c1", "score": 0.9},
            {"text": "result B", "source": "f2", "category": "c2", "score": 0.8},
        ]
        fused = hybrid._rrf_fusion(vector_results, [], limit=5)
        assert len(fused) == 2
        assert fused[0]["text"] == "result A"

    def test_rrf_graph_only(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        graph_results = [
            {"text": "graph result A", "source": "f1", "category": "c1", "score": 0.9},
        ]
        fused = hybrid._rrf_fusion([], graph_results, limit=5)
        assert len(fused) == 1
        assert fused[0]["origin"] == "graph"

    def test_rrf_combined(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        vector = [
            {"text": "shared result from vector", "source": "f1", "category": "c1", "score": 0.9},
            {"text": "vector only", "source": "f2", "category": "c2", "score": 0.8},
        ]
        graph = [
            {
                "text": "shared result from vector",
                "source": "f1",
                "category": "c1",
                "score": 0.7,
                "origin": "graph",
            },
            {
                "text": "graph only",
                "source": "f3",
                "category": "c3",
                "score": 0.6,
                "origin": "graph",
            },
        ]
        fused = hybrid._rrf_fusion(vector, graph, limit=5)

        # Shared result should rank highest (gets both vector + graph scores)
        assert fused[0]["text"] == "shared result from vector"
        assert fused[0]["origin"] == "hybrid"

    def test_rrf_dedup(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        vector = [
            {"text": "same doc", "source": "f1", "category": "c1", "score": 0.9},
        ]
        graph = [
            {"text": "same doc", "source": "f1", "category": "c1", "score": 0.7, "origin": "graph"},
        ]
        fused = hybrid._rrf_fusion(vector, graph, limit=5)
        assert len(fused) == 1

    def test_rrf_respects_limit(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        vector = [
            {"text": f"result {i}", "source": f"f{i}", "category": "c", "score": 0.5}
            for i in range(10)
        ]
        fused = hybrid._rrf_fusion(vector, [], limit=3)
        assert len(fused) == 3

    def test_rrf_weights(self, graph_engine, mock_rag_engine):
        """Test that changing weights affects ranking."""
        vector = [
            {"text": "vector preferred", "source": "f1", "category": "c1", "score": 0.9},
        ]
        graph = [
            {
                "text": "graph preferred",
                "source": "f2",
                "category": "c2",
                "score": 0.9,
                "origin": "graph",
            },
        ]

        # Vector-heavy weights
        hybrid_v = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
            vector_weight=0.9,
            graph_weight=0.1,
        )
        fused_v = hybrid_v._rrf_fusion(vector, graph)
        assert fused_v[0]["text"] == "vector preferred"

        # Graph-heavy weights
        hybrid_g = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
            vector_weight=0.1,
            graph_weight=0.9,
        )
        fused_g = hybrid_g._rrf_fusion(vector, graph)
        assert fused_g[0]["text"] == "graph preferred"


# --- Entity Recognition Tests ---


class TestEntityRecognition:
    def test_recognize_known_entity(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        ids = hybrid._recognize_entities("How to use FastAPI with authentication?")
        assert len(ids) > 0
        entity = graph_engine.get_entity(ids[0])
        assert entity.name == "fastapi"

    def test_recognize_multiple_entities(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        ids = hybrid._recognize_entities("FastAPI with Pydantic validation")
        names = {graph_engine.get_entity(eid).name for eid in ids}
        assert "fastapi" in names
        assert "pydantic" in names

    def test_recognize_no_entities(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        ids = hybrid._recognize_entities("hello world")
        assert len(ids) == 0

    def test_rejects_partial_substring_match(self, graph_engine, mock_rag_engine):
        """Short tokens that partially match an entity name are rejected."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        # "retry" is a substring of "retry with backoff" but word ratio 1/3 = 0.33 < 0.8
        ids = hybrid._recognize_entities("retry the operation")
        entity_names = {graph_engine.get_entity(eid).name for eid in ids}
        assert "retry with backoff" not in entity_names

    def test_accepts_exact_entity_match(self, graph_engine, mock_rag_engine):
        """Exact entity name in query is accepted."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        ids = hybrid._recognize_entities("hexagonal architecture patterns")
        entity_names = {graph_engine.get_entity(eid).name for eid in ids}
        assert "hexagonal architecture" in entity_names

    def test_accepts_single_word_entity(self, graph_engine, mock_rag_engine):
        """Single-word entity matched exactly is accepted (ratio 1/1 = 1.0)."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        ids = hybrid._recognize_entities("Use pydantic for validation")
        entity_names = {graph_engine.get_entity(eid).name for eid in ids}
        assert "pydantic" in entity_names


# --- Full Retrieval Tests ---


class TestHybridRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("How to use FastAPI?", limit=5)
        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_graph_context(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("FastAPI with Pydantic", limit=5)
        # Some results should have graph_context
        has_context = any("graph_context" in r for r in results)
        assert has_context

    @pytest.mark.asyncio
    async def test_retrieve_no_graph_match(self, graph_engine, mock_rag_engine):
        """When no entities match, should still return vector results."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve(
            "something totally unrelated xyz", limit=5, min_relevance=0.0
        )
        # Should fall back to vector-only results
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_with_context(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        prompt = await hybrid.query_with_context("How to use FastAPI?")
        assert "Pregunta:" in prompt
        assert "FastAPI" in prompt or "fastapi" in prompt.lower()

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_dir):
        """Test async context manager (with mocked RAG)."""
        mock_rag = MagicMock()
        mock_rag._init = AsyncMock()
        mock_rag.close = AsyncMock()
        mock_rag.retrieve = AsyncMock(return_value=[])

        graph = GraphEngine(data_dir=tmp_dir / "graphdb")

        hybrid = HybridRAGEngine(rag_engine=mock_rag, graph_engine=graph)
        async with hybrid as h:
            results = await h.retrieve("test")
            assert isinstance(results, list)


# --- Edge Cases ---


class TestEdgeCases:
    def test_empty_query(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        ids = hybrid._recognize_entities("")
        assert ids == []

    @pytest.mark.asyncio
    async def test_retrieve_empty_graph(self, mock_rag_engine, tmp_dir):
        """Works fine with empty graph - falls back to vector only."""
        empty_graph = GraphEngine(data_dir=tmp_dir / "graphdb")
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=empty_graph,
        )
        results = await hybrid.retrieve("test query", limit=3, min_relevance=0.0)
        assert len(results) > 0

    def test_rrf_empty_inputs(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        fused = hybrid._rrf_fusion([], [])
        assert fused == []


# --- Three-Tier RRF (Vector + Graph + Fulltext) ---


class TestThreeTierRRF:
    """Verify RRF fusion works with three sources: vector + graph + fulltext."""

    def test_rrf_with_fulltext_results(self, graph_engine, mock_rag_engine):
        """Full-text results contribute to fusion score."""
        engine = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
            vector_weight=0.4,
            graph_weight=0.3,
            fulltext_weight=0.3,
        )

        vector_results = [
            {
                "text": "FastAPI uses Pydantic for validation",
                "source": "a",
                "category": "training",
                "score": 0.9,
            },
        ]
        graph_results = [
            {
                "text": "FastAPI uses Pydantic for validation",
                "source": "a",
                "category": "training",
                "score": 0.8,
                "origin": "graph",
            },
        ]
        fulltext_results = [
            {
                "text": "FastAPI uses Pydantic for validation",
                "source": "a",
                "category": "training",
                "score": 1.0,
                "origin": "fulltext",
            },
        ]

        fused = engine._rrf_fusion(vector_results, graph_results, fulltext_results, limit=5)
        assert len(fused) == 1
        assert fused[0]["origin"] == "hybrid"
        assert fused[0]["score"] > 0

    def test_rrf_fulltext_only_results(self, graph_engine, mock_rag_engine):
        """Results only in fulltext still appear."""
        engine = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
            fulltext_weight=0.3,
        )

        fulltext_results = [
            {
                "text": "exact error match",
                "source": "err",
                "category": "learning",
                "score": 1.0,
                "origin": "fulltext",
            },
        ]

        fused = engine._rrf_fusion([], [], fulltext_results, limit=5)
        assert len(fused) == 1
        assert fused[0]["origin"] == "fulltext"

    def test_rrf_no_fulltext_weight_disables(self, graph_engine, mock_rag_engine):
        """When fulltext_weight=0, fulltext results don't contribute."""
        engine = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
            fulltext_weight=0.0,
        )

        fulltext_results = [
            {
                "text": "should not appear",
                "source": "x",
                "category": "x",
                "score": 1.0,
                "origin": "fulltext",
            },
        ]

        fused = engine._rrf_fusion([], [], fulltext_results, limit=5)
        assert len(fused) == 0

    def test_rrf_three_sources_ranking(self, graph_engine, mock_rag_engine):
        """Document in 3 sources ranks higher than in 2 or 1."""
        engine = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
            vector_weight=0.34,
            graph_weight=0.33,
            fulltext_weight=0.33,
        )

        vector = [
            {"text": "in all three", "source": "a", "category": "c", "score": 0.9},
            {"text": "vector only", "source": "b", "category": "c", "score": 0.8},
        ]
        graph = [
            {
                "text": "in all three",
                "source": "a",
                "category": "c",
                "score": 0.8,
                "origin": "graph",
            },
        ]
        fulltext = [
            {
                "text": "in all three",
                "source": "a",
                "category": "c",
                "score": 1.0,
                "origin": "fulltext",
            },
            {
                "text": "fulltext only",
                "source": "d",
                "category": "c",
                "score": 1.0,
                "origin": "fulltext",
            },
        ]

        fused = engine._rrf_fusion(vector, graph, fulltext, limit=5)
        assert fused[0]["text"] == "in all three"
        assert fused[0]["origin"] == "hybrid"

    def test_rrf_backward_compatible_two_args(self, graph_engine, mock_rag_engine):
        """Calling _rrf_fusion with 2 args still works (backward compatible)."""
        engine = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )

        vector = [{"text": "result", "source": "f", "category": "c", "score": 0.9}]
        fused = engine._rrf_fusion(vector, [])
        assert len(fused) == 1


class TestHybridRetrieveWithFullText:
    """Verify retrieve() includes fulltext results when engine is available."""

    @pytest.mark.asyncio
    async def test_retrieve_with_fulltext_engine(self, graph_engine):
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {"text": "vector result", "source": "v", "category": "training", "score": 0.9},
            ]
        )

        mock_fulltext = AsyncMock()
        mock_fulltext.search = AsyncMock(
            return_value=[
                {
                    "text": "fulltext result",
                    "source": "f",
                    "category": "training",
                    "score": 1.0,
                    "origin": "fulltext",
                },
            ]
        )

        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
            fulltext_engine=mock_fulltext,
            fulltext_weight=0.3,
        )

        results = await engine.retrieve("test query", limit=5, min_relevance=0.0)
        assert len(results) >= 1
        mock_fulltext.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_without_fulltext_engine(self, graph_engine):
        """When no fulltext engine, works as before (vector + graph only)."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {"text": "vector result", "source": "v", "category": "training", "score": 0.9},
            ]
        )

        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )

        results = await engine.retrieve("test query", limit=5, min_relevance=0.0)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_fulltext_failure_doesnt_break_retrieve(self, graph_engine):
        """If fulltext engine throws, retrieve still returns vector+graph results."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {"text": "vector result", "source": "v", "category": "training", "score": 0.9},
            ]
        )

        mock_fulltext = AsyncMock()
        mock_fulltext.search = AsyncMock(side_effect=Exception("Meilisearch down"))

        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
            fulltext_engine=mock_fulltext,
            fulltext_weight=0.3,
        )

        results = await engine.retrieve("test query", limit=5, min_relevance=0.0)
        assert len(results) >= 1


# --- Graph-Guided Expansion Tests ---


class TestGraphGuidedExpansion:
    """Tests for _build_expansion_queries() and the rewritten _graph_retrieve()."""

    def test_build_expansion_queries_returns_pairs(self, graph_engine, mock_rag_engine):
        """Seed with neighbors produces expansion query pairs."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        # FastAPI has neighbors: pydantic, starlette
        entity_ids = hybrid._recognize_entities("How to use FastAPI?")
        assert len(entity_ids) > 0

        queries = hybrid._build_expansion_queries(entity_ids)
        assert len(queries) > 0

        # Each query is a (string, float) tuple
        for q, score in queries:
            assert isinstance(q, str)
            assert isinstance(score, float)
            assert " " in q  # composite query with space between names

    def test_build_expansion_queries_deduplicates(self, graph_engine, mock_rag_engine):
        """Same pair (A,B) and (B,A) only appears once via frozenset dedup."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        # FastAPI -> pydantic and pydantic -> fastapi should deduplicate
        # Recognize both FastAPI and Pydantic as seeds
        entity_ids = hybrid._recognize_entities("FastAPI with Pydantic")
        assert len(entity_ids) >= 2

        queries = hybrid._build_expansion_queries(entity_ids)
        # Extract frozenset keys to verify no duplicates
        pair_keys = set()
        for q, _score in queries:
            parts = q.split(" ", 1)
            pair_key = frozenset(parts)
            assert pair_key not in pair_keys, f"Duplicate pair found: {q}"
            pair_keys.add(pair_key)

    def test_build_expansion_queries_sorted_by_score(self, graph_engine, mock_rag_engine):
        """Pairs are sorted by proximity score (highest first)."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        entity_ids = hybrid._recognize_entities("How to use FastAPI?")
        queries = hybrid._build_expansion_queries(entity_ids, depth=2)

        if len(queries) > 1:
            scores = [score for _q, score in queries]
            assert scores == sorted(scores, reverse=True)

    def test_build_expansion_queries_no_entities(self, graph_engine, mock_rag_engine):
        """Empty seed list returns empty list."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        queries = hybrid._build_expansion_queries([])
        assert queries == []

    @pytest.mark.asyncio
    async def test_graph_retrieve_uses_expansion_queries(self, graph_engine, mock_rag_engine):
        """Verify expansion queries are used for vector search."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        results = await hybrid._graph_retrieve("How to use FastAPI?", limit=10)
        assert len(results) > 0

        # The mock_rag_engine.retrieve should have been called with composite queries
        # (e.g., "fastapi pydantic") not just "fastapi"
        call_args = [call.args[0] for call in mock_rag_engine.retrieve.call_args_list]
        composite_calls = [arg for arg in call_args if " " in arg and len(arg.split()) >= 2]
        assert len(composite_calls) > 0, f"Expected composite queries, got: {call_args}"

    @pytest.mark.asyncio
    async def test_graph_retrieve_tags_results(self, graph_engine, mock_rag_engine):
        """Results have origin='graph' and graph_expansion key."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        results = await hybrid._graph_retrieve("How to use FastAPI?", limit=10)
        assert len(results) > 0

        for r in results:
            assert r["origin"] == "graph"

        # At least some results should have graph_expansion (from expansion queries)
        expansion_results = [r for r in results if "graph_expansion" in r]
        assert len(expansion_results) > 0

    @pytest.mark.asyncio
    async def test_graph_retrieve_deduplicates(self, graph_engine):
        """Same text from multiple expansion queries appears only once."""
        # Create a mock that always returns the same result
        rag = MagicMock()
        rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "FastAPI is a modern web framework for building APIs",
                    "source": "file1",
                    "category": "api",
                    "score": 0.9,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=rag,
            graph_engine=graph_engine,
        )
        results = await hybrid._graph_retrieve("How to use FastAPI?", limit=10)

        # Despite multiple expansion queries + seed search all returning the same text,
        # it should only appear once
        texts = [r["text"] for r in results]
        assert len(texts) == len(set(t[:100] for t in texts))

    @pytest.mark.asyncio
    async def test_graph_retrieve_includes_seed_fallback(self, graph_engine):
        """Direct seed entity name search also runs (not only expansion queries)."""
        call_log = []

        async def tracking_retrieve(query, limit=5):
            call_log.append(query)
            return [
                {
                    "text": f"result for {query}",
                    "source": "file1",
                    "category": "api",
                    "score": 0.9,
                },
            ]

        rag = MagicMock()
        rag.retrieve = AsyncMock(side_effect=tracking_retrieve)

        hybrid = HybridRAGEngine(
            rag_engine=rag,
            graph_engine=graph_engine,
        )
        results = await hybrid._graph_retrieve("How to use FastAPI?", limit=20)
        assert len(results) > 0

        # Should have both composite queries AND direct seed name queries
        # Direct seed query would be just "fastapi" (single word, entity name)
        seed_calls = [q for q in call_log if q == "fastapi"]
        assert len(seed_calls) > 0, f"Expected seed fallback call 'fastapi', got: {call_log}"

    @pytest.mark.asyncio
    async def test_graph_retrieve_empty_graph_returns_empty(self, mock_rag_engine, tmp_dir):
        """No entities recognized -> empty list."""
        empty_graph = GraphEngine(data_dir=tmp_dir / "graphdb")
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=empty_graph,
        )
        results = await hybrid._graph_retrieve("something unknown", limit=10)
        assert results == []

    def test_expansion_queries_capped_at_max(self, graph_engine, mock_rag_engine):
        """Expansion queries are limited to MAX_EXPANSION_QUERIES."""
        from src.knowledge.hybrid_rag import MAX_EXPANSION_QUERIES

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        entity_ids = hybrid._recognize_entities("FastAPI with Pydantic")
        queries = hybrid._build_expansion_queries(entity_ids, depth=2)
        assert len(queries) <= MAX_EXPANSION_QUERIES

    def test_neighbors_per_seed_capped(self, graph_engine, mock_rag_engine):
        """Each seed expands at most MAX_NEIGHBORS_PER_SEED neighbors."""
        from src.knowledge.hybrid_rag import MAX_NEIGHBORS_PER_SEED

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        entity_ids = hybrid._recognize_entities("How to use FastAPI?")
        assert len(entity_ids) > 0

        queries = hybrid._build_expansion_queries(entity_ids, depth=2)
        seed_entity = graph_engine.get_entity(entity_ids[0])
        seed_queries = [q for q, _ in queries if seed_entity.name in q]
        assert len(seed_queries) <= MAX_NEIGHBORS_PER_SEED

    @pytest.mark.asyncio
    async def test_graph_retrieve_integration(self, graph_engine):
        """Full retrieve() pipeline with expansion queries works end-to-end."""
        call_log = []

        async def tracking_retrieve(query, limit=5):
            call_log.append(query)
            return [
                {
                    "text": f"result for {query}",
                    "source": "file1",
                    "category": "api",
                    "score": 0.9,
                },
            ]

        rag = MagicMock()
        rag.retrieve = AsyncMock(side_effect=tracking_retrieve)

        hybrid = HybridRAGEngine(
            rag_engine=rag,
            graph_engine=graph_engine,
        )
        # Full retrieve() pipeline (vector + graph + fusion)
        results = await hybrid.retrieve("How to use FastAPI?", limit=5)
        assert len(results) > 0

        # Verify graph expansion queries were part of the calls
        composite_calls = [q for q in call_log if " " in q and len(q.split()) >= 2]
        assert (
            len(composite_calls) > 0
        ), f"Expected composite expansion queries in full pipeline, got: {call_log}"


# --- Adaptive Retrieval Integration Tests (FC-46) ---


class TestHybridAdaptiveRetrieval:
    """Verify HybridRAGEngine passes adaptive params through to RAGEngine."""

    @pytest.mark.asyncio
    async def test_retrieve_with_adaptive_params(self, graph_engine):
        """HybridRAGEngine passes confidence_threshold through to retrieve_adaptive."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(return_value=[])
        mock_rag.retrieve_adaptive = AsyncMock(
            return_value={
                "results": [
                    {
                        "text": "adaptive result",
                        "source": "f1",
                        "category": "training",
                        "score": 0.85,
                    },
                ],
                "chunks_retrieved": 1,
                "confidence": 0.85,
                "stop_reason": "threshold",
            }
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )

        results = await hybrid.retrieve(
            "test query",
            limit=5,
            confidence_threshold=0.7,
            min_k=2,
            max_k=10,
            query_entities=["fastapi"],
        )

        mock_rag.retrieve_adaptive.assert_called_once_with(
            "test query",
            min_k=2,
            max_k=10,
            confidence_threshold=0.7,
            query_entities=["fastapi"],
        )
        # Regular retrieve should NOT have been called for vector search
        # (it may still be called by _graph_retrieve)
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_retrieve_without_adaptive_params(self, graph_engine):
        """Default behavior unchanged when confidence_threshold is None."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "vector result",
                    "source": "v",
                    "category": "training",
                    "score": 0.9,
                },
            ]
        )
        mock_rag.retrieve_adaptive = AsyncMock()

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )

        results = await hybrid.retrieve("test query", limit=5)

        mock_rag.retrieve_adaptive.assert_not_called()
        # Regular retrieve should have been called
        assert mock_rag.retrieve.call_count >= 1
        assert len(results) >= 0


# --- Relevance Gate Integration Tests ---


class TestRelevanceGate:
    """Verify relevance gate reduces noise in graph expansion."""

    def test_default_weights_are_updated(self):
        """Constants reflect relevance gate values."""
        from src.knowledge.hybrid_rag import (
            DEFAULT_GRAPH_WEIGHT,
            DEFAULT_MIN_WEIGHT,
            DEFAULT_VECTOR_WEIGHT,
            ENTITY_MATCH_RATIO,
            MAX_EXPANSION_QUERIES,
            MAX_NEIGHBORS_PER_SEED,
        )

        assert DEFAULT_VECTOR_WEIGHT == 0.75
        assert DEFAULT_GRAPH_WEIGHT == 0.25
        assert DEFAULT_MIN_WEIGHT == 0.5
        assert MAX_NEIGHBORS_PER_SEED == 3
        assert MAX_EXPANSION_QUERIES == 5
        assert ENTITY_MATCH_RATIO == 0.8

    @pytest.mark.asyncio
    async def test_generic_query_skips_graph(self, graph_engine, mock_rag_engine):
        """Generic queries with no exact entity matches produce no graph results."""
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        graph_results = await hybrid._graph_retrieve("how to write clean code", limit=10)
        assert graph_results == []

    @pytest.mark.asyncio
    async def test_domain_query_gets_capped_graph(self, graph_engine):
        """Domain queries produce graph results capped at budget."""
        from src.knowledge.hybrid_rag import MAX_EXPANSION_QUERIES

        call_count = 0

        async def counting_retrieve(query, limit=5):
            nonlocal call_count
            call_count += 1
            return [
                {
                    "text": f"result for {query} #{call_count}",
                    "source": "file1",
                    "category": "api",
                    "score": 0.9,
                },
            ]

        rag = MagicMock()
        rag.retrieve = AsyncMock(side_effect=counting_retrieve)

        hybrid = HybridRAGEngine(
            rag_engine=rag,
            graph_engine=graph_engine,
        )
        results = await hybrid._graph_retrieve("How to use FastAPI?", limit=20)

        # Expansion queries + seed fallback, but expansion capped at MAX_EXPANSION_QUERIES
        entity_ids = hybrid._recognize_entities("How to use FastAPI?")
        max_calls = MAX_EXPANSION_QUERIES + len(entity_ids)  # expansion + seed fallback
        assert call_count <= max_calls

    def test_high_min_weight_filters_weak_edges(self, tmp_dir, mock_rag_engine):
        """Edges with weight < 0.5 are filtered by the new default."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")

        # Add entity with a weak edge
        engine.ingest_triple(
            Triple(
                subject_name="React",
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.USES,
                object_name="Webpack",
                object_type=EntityType.TECHNOLOGY,
                source_doc="doc1",
            )
        )

        # Manually set edge weight below threshold
        for u, v, data in engine._graph.edges(data=True):
            data["weight"] = 0.4  # Below DEFAULT_MIN_WEIGHT=0.5

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=engine,
        )

        entity_ids = hybrid._recognize_entities("React components")
        if entity_ids:
            queries = hybrid._build_expansion_queries(entity_ids)
            # No expansion queries because all edges are below min_weight
            assert queries == []


# --- Post-Retrieval Relevance Filter Tests ---


class TestComputeRelevance:
    """Unit tests for _compute_relevance static method."""

    def test_exact_overlap(self):
        """Query tokens all present in text -> 1.0."""
        score = HybridRAGEngine._compute_relevance(
            "FastAPI Pydantic validation",
            "FastAPI uses Pydantic for data validation and serialization",
        )
        assert score == 1.0

    def test_partial_overlap(self):
        """Some query tokens in text -> fraction."""
        score = HybridRAGEngine._compute_relevance(
            "FastAPI Pydantic validation",
            "FastAPI is a web framework built on Starlette",
        )
        # "fastapi" matches, "pydantic" and "validation" don't -> 1/3
        assert 0.3 <= score <= 0.4

    def test_no_overlap(self):
        """No query tokens in text -> 0.0."""
        score = HybridRAGEngine._compute_relevance(
            "docker compose networking",
            "Unrelated project configuration for appointment system",
        )
        assert score == 0.0

    def test_empty_query(self):
        """Empty query -> 1.0 (let it pass)."""
        score = HybridRAGEngine._compute_relevance("", "some text here")
        assert score == 1.0

    def test_stopword_only_query(self):
        """Query with only stopwords -> 1.0 (no meaningful tokens)."""
        score = HybridRAGEngine._compute_relevance("the is a", "some text")
        assert score == 1.0

    def test_empty_text(self):
        """Empty text -> 0.0."""
        score = HybridRAGEngine._compute_relevance("docker compose", "")
        assert score == 0.0

    def test_stopwords_filtered(self):
        """Stopwords don't count as overlap."""
        score = HybridRAGEngine._compute_relevance(
            "how to use FastAPI",
            "FastAPI is great for building APIs",
        )
        assert score > 0.0

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        score = HybridRAGEngine._compute_relevance(
            "FASTAPI pydantic",
            "fastapi and Pydantic work together",
        )
        assert score == 1.0

    def test_text_capped_at_500_chars(self):
        """Only first 500 chars of text are tokenized."""
        long_text = "x " * 1000 + "docker compose networking"
        score = HybridRAGEngine._compute_relevance(
            "docker compose networking",
            long_text,
        )
        assert score == 0.0

    def test_bilingual_stopwords(self):
        """Both Spanish and English stopwords are filtered."""
        score = HybridRAGEngine._compute_relevance(
            "qué es terraform",
            "terraform modules provide infrastructure as code",
        )
        assert score == 1.0

    def test_short_tokens_ignored(self):
        """Tokens with < 2 chars are ignored by regex."""
        score = HybridRAGEngine._compute_relevance(
            "a b c docker",
            "docker container management",
        )
        assert score == 1.0


class TestRelevanceFilter:
    """Integration tests for the post-retrieval relevance filter in retrieve()."""

    @pytest.mark.asyncio
    async def test_filter_drops_irrelevant_results(self, graph_engine):
        """Results with no token overlap are dropped."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "docker compose networking and container orchestration",
                    "source": "relevant",
                    "category": "devops",
                    "score": 0.9,
                },
                {
                    "text": "Unrelated appointment scheduling system configuration",
                    "source": "irrelevant",
                    "category": "medical",
                    "score": 0.8,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("docker compose networking", limit=5)

        texts = [r["text"] for r in results]
        assert any("docker" in t.lower() for t in texts)
        assert not any("Unrelated" in t for t in texts)

    @pytest.mark.asyncio
    async def test_filter_keeps_relevant_results(self, graph_engine):
        """Results with good token overlap pass the filter."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "FastAPI framework uses Pydantic for data validation",
                    "source": "f1",
                    "category": "api",
                    "score": 0.9,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("FastAPI Pydantic validation", limit=5)

        assert len(results) >= 1
        assert any("FastAPI" in r["text"] for r in results)

    @pytest.mark.asyncio
    async def test_filter_adds_relevance_score(self, graph_engine):
        """Each result gets a 'relevance' field."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "FastAPI web framework for Python APIs",
                    "source": "f1",
                    "category": "api",
                    "score": 0.9,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("FastAPI Python", limit=5)

        for r in results:
            assert "relevance" in r
            assert 0.0 <= r["relevance"] <= 1.0

    @pytest.mark.asyncio
    async def test_filter_disabled_with_zero_threshold(self, graph_engine):
        """Setting min_relevance=0.0 disables the filter."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "completely unrelated appointment system",
                    "source": "f1",
                    "category": "medical",
                    "score": 0.9,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("docker compose networking", limit=5, min_relevance=0.0)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_filter_custom_threshold(self, graph_engine):
        """Custom min_relevance overrides the default."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "FastAPI web framework overview with Starlette",
                    "source": "f1",
                    "category": "api",
                    "score": 0.9,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )

        results_strict = await hybrid.retrieve(
            "FastAPI Pydantic validation serialization",
            limit=5,
            min_relevance=0.9,
        )
        results_permissive = await hybrid.retrieve(
            "FastAPI Pydantic validation serialization",
            limit=5,
            min_relevance=0.1,
        )

        assert len(results_strict) <= len(results_permissive)

    @pytest.mark.asyncio
    async def test_filter_preserves_result_order(self, graph_engine):
        """Filtered results maintain their RRF score ordering."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                {
                    "text": "terraform modules for AWS infrastructure provisioning",
                    "source": "f1",
                    "category": "devops",
                    "score": 0.95,
                },
                {
                    "text": "terraform state management and remote backends",
                    "source": "f2",
                    "category": "devops",
                    "score": 0.85,
                },
                {
                    "text": "completely unrelated inventory software system",
                    "source": "f3",
                    "category": "medical",
                    "score": 0.80,
                },
            ]
        )

        hybrid = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )
        results = await hybrid.retrieve("terraform infrastructure AWS", limit=5)

        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.asyncio
    async def test_default_threshold_from_constant(self):
        """Default threshold matches DEFAULT_RELEVANCE_THRESHOLD constant."""
        from src.knowledge.hybrid_rag import DEFAULT_RELEVANCE_THRESHOLD

        assert DEFAULT_RELEVANCE_THRESHOLD == 0.12
