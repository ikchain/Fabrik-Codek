"""Tests for HybridRAGEngine."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge.graph_engine import GraphEngine
from src.knowledge.graph_schema import Entity, EntityType, Relation, RelationType, Triple
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
    engine.ingest_triple(Triple(
        subject_name="FastAPI",
        subject_type=EntityType.TECHNOLOGY,
        relation_type=RelationType.USES,
        object_name="Pydantic",
        object_type=EntityType.TECHNOLOGY,
        source_doc="doc1",
    ))
    engine.ingest_triple(Triple(
        subject_name="FastAPI",
        subject_type=EntityType.TECHNOLOGY,
        relation_type=RelationType.USES,
        object_name="Starlette",
        object_type=EntityType.TECHNOLOGY,
        source_doc="doc2",
    ))
    engine.ingest_triple(Triple(
        subject_name="retry with backoff",
        subject_type=EntityType.STRATEGY,
        relation_type=RelationType.FIXES,
        object_name="connection error",
        object_type=EntityType.ERROR_TYPE,
        source_doc="doc3",
    ))
    engine.ingest_triple(Triple(
        subject_name="hexagonal architecture",
        subject_type=EntityType.PATTERN,
        relation_type=RelationType.RELATED_TO,
        object_name="domain-driven design",
        object_type=EntityType.CONCEPT,
        source_doc="doc4",
    ))

    return engine


@pytest.fixture
def mock_rag_engine():
    """Mock RAGEngine for testing without LanceDB."""
    rag = MagicMock()
    rag._init = AsyncMock()
    rag.close = AsyncMock()
    rag.retrieve = AsyncMock(return_value=[
        {"text": "FastAPI is a modern web framework", "source": "file1", "category": "api", "score": 0.9},
        {"text": "Pydantic provides data validation", "source": "file2", "category": "api", "score": 0.8},
        {"text": "Use SQLAlchemy for ORM", "source": "file3", "category": "database", "score": 0.7},
    ])
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
            {"text": "shared result from vector", "source": "f1", "category": "c1", "score": 0.7, "origin": "graph"},
            {"text": "graph only", "source": "f3", "category": "c3", "score": 0.6, "origin": "graph"},
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
        vector = [{"text": f"result {i}", "source": f"f{i}", "category": "c", "score": 0.5}
                  for i in range(10)]
        fused = hybrid._rrf_fusion(vector, [], limit=3)
        assert len(fused) == 3

    def test_rrf_weights(self, graph_engine, mock_rag_engine):
        """Test that changing weights affects ranking."""
        vector = [
            {"text": "vector preferred", "source": "f1", "category": "c1", "score": 0.9},
        ]
        graph = [
            {"text": "graph preferred", "source": "f2", "category": "c2", "score": 0.9, "origin": "graph"},
        ]

        # Vector-heavy weights
        hybrid_v = HybridRAGEngine(
            rag_engine=mock_rag_engine, graph_engine=graph_engine,
            vector_weight=0.9, graph_weight=0.1,
        )
        fused_v = hybrid_v._rrf_fusion(vector, graph)
        assert fused_v[0]["text"] == "vector preferred"

        # Graph-heavy weights
        hybrid_g = HybridRAGEngine(
            rag_engine=mock_rag_engine, graph_engine=graph_engine,
            vector_weight=0.1, graph_weight=0.9,
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
        results = await hybrid.retrieve("something totally unrelated xyz", limit=5)
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
        results = await hybrid.retrieve("test query", limit=3)
        assert len(results) > 0

    def test_rrf_empty_inputs(self, graph_engine, mock_rag_engine):
        hybrid = HybridRAGEngine(
            rag_engine=mock_rag_engine,
            graph_engine=graph_engine,
        )
        fused = hybrid._rrf_fusion([], [])
        assert fused == []
