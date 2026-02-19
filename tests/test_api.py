"""Tests for the Fabrik-Codek Web API."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from src import __version__
from src.core.llm_client import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content="test answer", model="qwen2.5-coder:7b", tokens=42, latency=100.0):
    return LLMResponse(content=content, model=model, tokens_used=tokens, latency_ms=latency)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.generate = AsyncMock(return_value=_make_llm_response())
    client.chat = AsyncMock(return_value=_make_llm_response(content="chat reply"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_rag():
    rag = AsyncMock()
    rag.retrieve = AsyncMock(return_value=[
        {"text": "some doc", "source": "file.jsonl", "category": "testing", "score": 0.9},
    ])
    rag.close = AsyncMock()
    return rag


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.load.return_value = True
    graph.search_entities.return_value = []
    graph.get_neighbors.return_value = []
    graph.get_stats.return_value = {
        "entity_count": 10,
        "edge_count": 25,
        "connected_components": 3,
        "entity_types": {"technology": 5, "concept": 3, "tool": 2},
        "relation_types": {"uses": 10, "depends_on": 8, "part_of": 7},
        "graph_path": "/tmp/graph.json",
        "graph_exists": True,
    }
    return graph


@pytest.fixture
def mock_hybrid():
    hybrid = AsyncMock()
    hybrid.retrieve = AsyncMock(return_value=[
        {"text": "hybrid doc", "source": "h.jsonl", "category": "agents", "score": 0.95, "origin": "vector"},
    ])
    hybrid._owns_rag = False
    return hybrid


def _build_app(llm, rag=None, graph=None, hybrid=None, ollama_ok=True, api_key=None):
    """Build a FastAPI app with pre-injected state (bypasses lifespan)."""
    import time as _time

    from fastapi import FastAPI

    from src.interfaces import api as api_module
    from src.interfaces.api import (
        AskResponse,
        ChatResponse,
        GraphSearchResponse,
        GraphStatsResponse,
        HealthResponse,
        SearchResponse,
        StatusResponse,
    )

    test_app = FastAPI()

    # Inject state
    test_app.state.llm = llm
    test_app.state.ollama_ok = ollama_ok
    test_app.state.ollama_checked_at = _time.monotonic()
    test_app.state.rag = rag
    test_app.state.graph = graph
    test_app.state.hybrid = hybrid

    # API key middleware (if configured)
    if api_key:
        with patch("src.interfaces.api.settings") as mock_settings:
            mock_settings.api_key = api_key
            # Re-import to get fresh middleware closure? No - middleware reads settings at request time.
            pass

    # Register endpoints
    test_app.get("/health", response_model=HealthResponse)(api_module.health)
    test_app.get("/status", response_model=StatusResponse)(api_module.status)
    test_app.post("/ask", response_model=AskResponse)(api_module.ask)
    test_app.post("/chat", response_model=ChatResponse)(api_module.chat)
    test_app.post("/search", response_model=SearchResponse)(api_module.search)
    test_app.post("/graph/search", response_model=GraphSearchResponse)(api_module.graph_search)
    test_app.get("/graph/stats", response_model=GraphStatsResponse)(api_module.graph_stats)

    return test_app


def _build_auth_app(llm, api_key="test-secret-key"):
    """Build app with auth middleware from the real api module."""
    from src.interfaces.api import app as real_app

    real_app.state.llm = llm
    real_app.state.ollama_ok = True
    real_app.state.ollama_checked_at = time.monotonic()
    real_app.state.rag = None
    real_app.state.graph = None
    real_app.state.hybrid = None
    return real_app


@pytest.fixture
def full_app(mock_llm, mock_rag, mock_graph, mock_hybrid):
    return _build_app(mock_llm, mock_rag, mock_graph, mock_hybrid)


@pytest.fixture
def no_ollama_app(mock_llm, mock_rag, mock_graph):
    mock_llm.health_check = AsyncMock(return_value=False)
    return _build_app(mock_llm, mock_rag, mock_graph, ollama_ok=False)


@pytest.fixture
def no_graph_app(mock_llm, mock_rag):
    return _build_app(mock_llm, mock_rag, graph=None)


@pytest.fixture
def no_rag_app(mock_llm, mock_graph):
    return _build_app(mock_llm, rag=None, graph=mock_graph)


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self, full_app):
        async with _client(full_app) as c:
            r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_has_version(self, full_app):
        async with _client(full_app) as c:
            r = await c.get("/health")
        assert r.json()["version"] == __version__

    @pytest.mark.asyncio
    async def test_health_always_200_even_if_ollama_down(self, no_ollama_app):
        async with _client(no_ollama_app) as c:
            r = await c.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# TestStatus
# ---------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_all_healthy(self, full_app):
        async with _client(full_app) as c:
            r = await c.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["ollama"] == "ok"
        assert data["rag"] == "ok"
        assert data["graph"] == "ok"

    @pytest.mark.asyncio
    async def test_ollama_down(self, no_ollama_app):
        async with _client(no_ollama_app) as c:
            r = await c.get("/status")
        assert r.json()["ollama"] == "unavailable"

    @pytest.mark.asyncio
    async def test_no_graph(self, no_graph_app):
        async with _client(no_graph_app) as c:
            r = await c.get("/status")
        assert r.json()["graph"] == "unavailable"

    @pytest.mark.asyncio
    async def test_no_rag(self, no_rag_app):
        async with _client(no_rag_app) as c:
            r = await c.get("/status")
        assert r.json()["rag"] == "unavailable"


# ---------------------------------------------------------------------------
# TestAsk
# ---------------------------------------------------------------------------


class TestAsk:
    @pytest.mark.asyncio
    async def test_basic(self, full_app, mock_llm):
        async with _client(full_app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "test answer"
        assert data["model"] == "qwen2.5-coder:7b"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_rag(self, full_app, mock_rag, mock_llm):
        async with _client(full_app) as c:
            r = await c.post("/ask", json={"prompt": "how to test", "use_rag": True})
        assert r.status_code == 200
        mock_rag.retrieve.assert_called_once()
        prompt_sent = mock_llm.generate.call_args.args[0]
        assert "context" in prompt_sent.lower()

    @pytest.mark.asyncio
    async def test_with_graph(self, full_app, mock_hybrid, mock_llm):
        async with _client(full_app) as c:
            r = await c.post("/ask", json={"prompt": "explain DDD", "use_graph": True})
        assert r.status_code == 200
        mock_hybrid.retrieve.assert_called_once()
        assert len(r.json()["sources"]) > 0

    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self, full_app):
        async with _client(full_app) as c:
            r = await c.post("/ask", json={"prompt": ""})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_503_when_ollama_down(self, no_ollama_app):
        async with _client(no_ollama_app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_model_override(self, full_app, mock_llm):
        async with _client(full_app) as c:
            r = await c.post("/ask", json={"prompt": "hi", "model": "qwen2.5-coder:32b"})
        assert r.status_code == 200
        assert mock_llm.generate.call_args.kwargs.get("model") == "qwen2.5-coder:32b"

    @pytest.mark.asyncio
    async def test_fallback_without_rag(self, no_rag_app, mock_llm):
        async with _client(no_rag_app) as c:
            r = await c.post("/ask", json={"prompt": "hello", "use_rag": True})
        assert r.status_code == 200
        assert r.json()["answer"] == "test answer"

    @pytest.mark.asyncio
    async def test_sources_empty_when_no_context(self, full_app):
        async with _client(full_app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.json()["sources"] == []


# ---------------------------------------------------------------------------
# TestChat
# ---------------------------------------------------------------------------


class TestChat:
    def _msgs(self, *contents):
        return [{"role": "user", "content": c} for c in contents]

    @pytest.mark.asyncio
    async def test_basic(self, full_app, mock_llm):
        async with _client(full_app) as c:
            r = await c.post("/chat", json={"messages": self._msgs("hi")})
        assert r.status_code == 200
        assert r.json()["reply"] == "chat reply"

    @pytest.mark.asyncio
    async def test_history(self, full_app, mock_llm):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
        ]
        async with _client(full_app) as c:
            r = await c.post("/chat", json={"messages": msgs})
        assert r.status_code == 200
        assert len(mock_llm.chat.call_args.args[0]) == 3

    @pytest.mark.asyncio
    async def test_empty_messages_rejected(self, full_app):
        async with _client(full_app) as c:
            r = await c.post("/chat", json={"messages": []})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_503_when_ollama_down(self, no_ollama_app):
        async with _client(no_ollama_app) as c:
            r = await c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_system_message(self, full_app, mock_llm):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        async with _client(full_app) as c:
            r = await c.post("/chat", json={"messages": msgs})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_model_override(self, full_app, mock_llm):
        async with _client(full_app) as c:
            r = await c.post("/chat", json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "custom-model",
            })
        assert r.status_code == 200
        assert mock_llm.chat.call_args.kwargs.get("model") == "custom-model"


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_results(self, full_app, mock_rag):
        async with _client(full_app) as c:
            r = await c.post("/search", json={"query": "testing"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["results"][0]["category"] == "testing"

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self, full_app):
        async with _client(full_app) as c:
            r = await c.post("/search", json={"query": ""})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_category_filter(self, full_app, mock_rag):
        async with _client(full_app) as c:
            r = await c.post("/search", json={"query": "test", "category": "nonexistent"})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_rag_unavailable(self, no_rag_app):
        async with _client(no_rag_app) as c:
            r = await c.post("/search", json={"query": "anything"})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_limit(self, full_app, mock_rag):
        async with _client(full_app) as c:
            r = await c.post("/search", json={"query": "test", "limit": 2})
        assert r.status_code == 200
        mock_rag.retrieve.assert_called_once_with("test", limit=2)


# ---------------------------------------------------------------------------
# TestGraphSearch
# ---------------------------------------------------------------------------


class TestGraphSearch:
    @pytest.mark.asyncio
    async def test_finds_entities(self, full_app, mock_graph):
        from src.knowledge.graph_schema import Entity, EntityType

        entity = Entity(
            id="python_tech",
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            mention_count=5,
            aliases=["py"],
        )
        mock_graph.search_entities.return_value = [entity]
        mock_graph.get_neighbors.return_value = []

        async with _client(full_app) as c:
            r = await c.post("/graph/search", json={"query": "python"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["entities"][0]["name"] == "Python"

    @pytest.mark.asyncio
    async def test_no_results(self, full_app, mock_graph):
        mock_graph.search_entities.return_value = []

        async with _client(full_app) as c:
            r = await c.post("/graph/search", json={"query": "nonexistent"})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_no_graph(self, no_graph_app):
        async with _client(no_graph_app) as c:
            r = await c.post("/graph/search", json={"query": "anything"})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_depth_param(self, full_app, mock_graph):
        mock_graph.search_entities.return_value = []

        async with _client(full_app) as c:
            r = await c.post("/graph/search", json={"query": "x", "depth": 3})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_limit_param(self, full_app, mock_graph):
        mock_graph.search_entities.return_value = []

        async with _client(full_app) as c:
            r = await c.post("/graph/search", json={"query": "x", "limit": 20})
        assert r.status_code == 200
        mock_graph.search_entities.assert_called_once_with("x", limit=20)


# ---------------------------------------------------------------------------
# TestGraphStats
# ---------------------------------------------------------------------------


class TestGraphStats:
    @pytest.mark.asyncio
    async def test_with_data(self, full_app, mock_graph):
        async with _client(full_app) as c:
            r = await c.get("/graph/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["entity_count"] == 10
        assert data["edge_count"] == 25
        assert data["connected_components"] == 3

    @pytest.mark.asyncio
    async def test_no_graph(self, no_graph_app):
        async with _client(no_graph_app) as c:
            r = await c.get("/graph/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["entity_count"] == 0
        assert data["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_type_breakdown(self, full_app, mock_graph):
        async with _client(full_app) as c:
            r = await c.get("/graph/stats")
        data = r.json()
        assert "technology" in data["entity_types"]
        assert "uses" in data["relation_types"]

    @pytest.mark.asyncio
    async def test_format(self, full_app, mock_graph):
        async with _client(full_app) as c:
            r = await c.get("/graph/stats")
        data = r.json()
        required_keys = {"entity_count", "edge_count", "connected_components", "entity_types", "relation_types"}
        assert required_keys <= set(data.keys())


# ---------------------------------------------------------------------------
# TestAPIKeyAuth
# ---------------------------------------------------------------------------


class TestAPIKeyAuth:
    """Tests for API key authentication middleware."""

    @pytest.mark.asyncio
    async def test_no_auth_when_key_not_configured(self, full_app):
        """Without api_key configured, all requests pass through."""
        async with _client(full_app) as c:
            r = await c.get("/status")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_rejects_without_key(self, mock_llm):
        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = "secret-123"
            ms.api_cors_origins = ["*"]
            ms.default_model = "qwen2.5-coder:7b"
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = False
            app = _build_auth_app(mock_llm, api_key="secret-123")
            async with _client(app) as c:
                r = await c.get("/status")
            assert r.status_code == 401
            assert "API key" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_accepts_x_api_key_header(self, mock_llm):
        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = "secret-123"
            ms.api_cors_origins = ["*"]
            ms.default_model = "qwen2.5-coder:7b"
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = False
            app = _build_auth_app(mock_llm, api_key="secret-123")
            async with _client(app) as c:
                r = await c.get("/status", headers={"X-API-Key": "secret-123"})
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_accepts_bearer_token(self, mock_llm):
        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = "secret-123"
            ms.api_cors_origins = ["*"]
            ms.default_model = "qwen2.5-coder:7b"
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = False
            app = _build_auth_app(mock_llm, api_key="secret-123")
            async with _client(app) as c:
                r = await c.get("/status", headers={"Authorization": "Bearer secret-123"})
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_health_is_public(self, mock_llm):
        """Health endpoint should not require auth."""
        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = "secret-123"
            ms.api_cors_origins = ["*"]
            app = _build_auth_app(mock_llm, api_key="secret-123")
            async with _client(app) as c:
                r = await c.get("/health")
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, mock_llm):
        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = "secret-123"
            ms.api_cors_origins = ["*"]
            ms.default_model = "qwen2.5-coder:7b"
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = False
            app = _build_auth_app(mock_llm, api_key="secret-123")
            async with _client(app) as c:
                r = await c.get("/status", headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 401


# ---------------------------------------------------------------------------
# TestOllamaHealthTTL
# ---------------------------------------------------------------------------


class TestOllamaHealthTTL:
    """Tests for the TTL-cached Ollama health check."""

    @pytest.mark.asyncio
    async def test_skips_recheck_within_ttl(self, mock_llm):
        """When Ollama is down but TTL hasn't expired, don't re-check."""
        mock_llm.health_check = AsyncMock(return_value=False)
        app = _build_app(mock_llm, ollama_ok=False)
        # Set checked_at to now (within TTL)
        app.state.ollama_checked_at = time.monotonic()

        async with _client(app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.status_code == 503
        # health_check should NOT have been called (within TTL)
        mock_llm.health_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_rechecks_after_ttl_expired(self, mock_llm):
        """When TTL has expired, re-check Ollama health."""
        mock_llm.health_check = AsyncMock(return_value=False)
        app = _build_app(mock_llm, ollama_ok=False)
        # Set checked_at to 10s ago (past TTL)
        app.state.ollama_checked_at = time.monotonic() - 10.0

        async with _client(app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.status_code == 503
        mock_llm.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovers_after_ttl(self, mock_llm):
        """When Ollama comes back up after TTL, requests succeed."""
        mock_llm.health_check = AsyncMock(return_value=True)
        app = _build_app(mock_llm, ollama_ok=False)
        # Force TTL to be expired
        app.state.ollama_checked_at = time.monotonic() - 10.0

        async with _client(app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_no_recheck_when_healthy(self, mock_llm):
        """When Ollama is healthy, don't waste time re-checking."""
        app = _build_app(mock_llm, ollama_ok=True)
        mock_llm.health_check.reset_mock()

        async with _client(app) as c:
            r = await c.post("/ask", json={"prompt": "hello"})
        assert r.status_code == 200
        # _ensure_ollama should short-circuit, no health_check call
        mock_llm.health_check.assert_not_called()


# ---------------------------------------------------------------------------
# TestCORS
# ---------------------------------------------------------------------------


class TestCORS:
    """Tests for CORS middleware."""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, mock_llm):
        """Real app (with middleware) should return CORS headers."""
        from src.interfaces.api import app as real_app

        real_app.state.llm = mock_llm
        real_app.state.ollama_ok = True
        real_app.state.ollama_checked_at = time.monotonic()
        real_app.state.rag = None
        real_app.state.graph = None
        real_app.state.hybrid = None

        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = None
            ms.api_cors_origins = ["*"]
            ms.default_model = "qwen2.5-coder:7b"
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = False

            async with _client(real_app) as c:
                r = await c.get("/health", headers={"Origin": "http://localhost:3000"})
            assert r.headers.get("access-control-allow-origin") == "*"

    @pytest.mark.asyncio
    async def test_cors_preflight(self, mock_llm):
        """OPTIONS preflight should return CORS headers."""
        from src.interfaces.api import app as real_app

        real_app.state.llm = mock_llm
        real_app.state.ollama_ok = True
        real_app.state.ollama_checked_at = time.monotonic()
        real_app.state.rag = None
        real_app.state.graph = None
        real_app.state.hybrid = None

        with patch("src.interfaces.api.settings") as ms:
            ms.api_key = None
            ms.api_cors_origins = ["*"]

            async with _client(real_app) as c:
                r = await c.options(
                    "/ask",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                    },
                )
            assert r.status_code == 200
            assert "access-control-allow-methods" in r.headers
