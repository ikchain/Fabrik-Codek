# Fabrik-Codek MCP Server - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose Fabrik-Codek as a standard MCP server so any MCP-compatible agent can use it as a local knowledge base.

**Architecture:** New `src/interfaces/mcp_server.py` module using the official `mcp` Python SDK (FastMCP). Wraps existing LLMClient, RAGEngine, GraphEngine, and HybridRAGEngine. New `fabrik mcp` CLI command for stdio transport. Existing REST API unchanged.

**Tech Stack:** `mcp[cli]` SDK, existing Fabrik services (LLMClient, RAGEngine, GraphEngine, HybridRAGEngine)

---

### Task 1: Add MCP SDK dependency

**Files:**
- Modify: `pyproject.toml:13-49` (dependencies section)

**Step 1: Add mcp dependency to pyproject.toml**

In the `dependencies` list in `pyproject.toml`, add after the Web API section:

```toml
    # MCP Server
    "mcp[cli]>=1.0.0",
```

**Step 2: Install the dependency**

Run: `pip install "mcp[cli]" --break-system-packages`
Expected: Successfully installed mcp-*

**Step 3: Verify installation**

Run: `python3 -c "from mcp.server.fastmcp import FastMCP; print('MCP SDK OK')"`
Expected: `MCP SDK OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "BUILD: Agregar dependencia mcp[cli] para MCP server"
```

---

### Task 2: Create MCP server module with fabrik_status tool

**Files:**
- Create: `src/interfaces/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Step 1: Write the failing test for fabrik_status**

Create `tests/test_mcp_server.py`:

```python
"""Tests for the Fabrik-Codek MCP Server."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


class TestFabrikStatus:
    @pytest.mark.asyncio
    async def test_status_all_healthy(self, mock_llm):
        with patch("src.interfaces.mcp_server.LLMClient", return_value=mock_llm), \
             patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "qwen2.5-coder:7b"
            ms.ollama_host = "http://localhost:11434"

            from src.interfaces.mcp_server import fabrik_status
            result = await fabrik_status()

        assert "ok" in result
        assert "qwen2.5-coder:7b" in result

    @pytest.mark.asyncio
    async def test_status_ollama_down(self, mock_llm):
        mock_llm.health_check = AsyncMock(return_value=False)
        with patch("src.interfaces.mcp_server.LLMClient", return_value=mock_llm), \
             patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "qwen2.5-coder:7b"
            ms.ollama_host = "http://localhost:11434"

            from src.interfaces.mcp_server import fabrik_status
            result = await fabrik_status()

        assert "unavailable" in result
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikStatus -v`
Expected: FAIL (ModuleNotFoundError or ImportError)

**Step 3: Write minimal MCP server with fabrik_status**

Create `src/interfaces/mcp_server.py`:

```python
"""Fabrik-Codek MCP Server - Expose knowledge base to any MCP-compatible agent."""

import json

from mcp.server.fastmcp import FastMCP

from src import __version__
from src.config import settings
from src.core import LLMClient

mcp = FastMCP(
    "fabrik-codek",
    version=__version__,
    description="Local AI dev assistant with hybrid RAG knowledge base",
)


@mcp.tool()
async def fabrik_status() -> str:
    """Check Fabrik-Codek system status: Ollama, RAG, Knowledge Graph, and Datalake."""
    async with LLMClient() as llm:
        ollama_ok = await llm.health_check()

    # Check graph
    try:
        from src.knowledge.graph_engine import GraphEngine
        graph = GraphEngine()
        graph_ok = graph.load()
        if graph_ok:
            stats = graph.get_stats()
            graph_info = f"ok ({stats['entity_count']} entities, {stats['edge_count']} edges)"
        else:
            graph_info = "not built (run: fabrik graph build)"
    except Exception:
        graph_info = "unavailable"

    status = {
        "ollama": "ok" if ollama_ok else "unavailable",
        "model": settings.default_model,
        "datalake": "ok" if settings.datalake_path.exists() else "unavailable",
        "knowledge_graph": graph_info,
        "version": __version__,
    }
    return json.dumps(status, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikStatus -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/interfaces/mcp_server.py tests/test_mcp_server.py
git commit -m "FEAT: Crear MCP server con tool fabrik_status"
```

---

### Task 3: Add fabrik_search tool (vector semantic search)

**Files:**
- Modify: `src/interfaces/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

Append to `tests/test_mcp_server.py`:

```python
@pytest.fixture
def mock_rag():
    rag = AsyncMock()
    rag.retrieve = AsyncMock(return_value=[
        {"text": "Use repository pattern for data access", "source": "ddd.jsonl", "category": "ddd", "score": 0.92},
    ])
    rag.__aenter__ = AsyncMock(return_value=rag)
    rag.__aexit__ = AsyncMock(return_value=None)
    rag.close = AsyncMock()
    return rag


class TestFabrikSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_rag):
        with patch("src.interfaces.mcp_server.RAGEngine", return_value=mock_rag):
            from src.interfaces.mcp_server import fabrik_search
            result = await fabrik_search(query="repository pattern")

        parsed = json.loads(result)
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["category"] == "ddd"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, mock_rag):
        with patch("src.interfaces.mcp_server.RAGEngine", return_value=mock_rag):
            from src.interfaces.mcp_server import fabrik_search
            await fabrik_search(query="test", limit=3)

        mock_rag.retrieve.assert_called_once_with("test", limit=3)

    @pytest.mark.asyncio
    async def test_search_rag_unavailable(self):
        mock_rag = AsyncMock()
        mock_rag.__aenter__ = AsyncMock(side_effect=Exception("no datalake"))
        with patch("src.interfaces.mcp_server.RAGEngine", return_value=mock_rag):
            from src.interfaces.mcp_server import fabrik_search
            result = await fabrik_search(query="anything")

        parsed = json.loads(result)
        assert parsed["results"] == []
        assert "error" in parsed
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikSearch -v`
Expected: FAIL (fabrik_search not defined)

**Step 3: Implement fabrik_search**

Add to `src/interfaces/mcp_server.py`:

```python
from src.knowledge.rag import RAGEngine


@mcp.tool()
async def fabrik_search(query: str, limit: int = 5) -> str:
    """Semantic search in Fabrik-Codek's knowledge base. Returns relevant documents from training data, decisions, and learnings.

    Args:
        query: The search query (natural language)
        limit: Maximum number of results (1-20, default 5)
    """
    limit = max(1, min(20, limit))
    try:
        async with RAGEngine() as rag:
            results = await rag.retrieve(query, limit=limit)
        return json.dumps({
            "results": [
                {
                    "text": r["text"][:500],
                    "source": r.get("source", ""),
                    "category": r.get("category", ""),
                    "score": round(r.get("score", 0.0), 3),
                }
                for r in results
            ],
            "count": len(results),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"results": [], "count": 0, "error": str(exc)})
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikSearch -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/interfaces/mcp_server.py tests/test_mcp_server.py
git commit -m "FEAT: Agregar tool fabrik_search al MCP server"
```

---

### Task 4: Add fabrik_graph_search tool (knowledge graph traversal)

**Files:**
- Modify: `src/interfaces/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

Append to `tests/test_mcp_server.py`:

```python
from src.knowledge.graph_schema import Entity, EntityType


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.load.return_value = True
    entity = Entity(
        id="fastapi_tech",
        name="fastapi",
        entity_type=EntityType.TECHNOLOGY,
        mention_count=8,
        aliases=["fast-api"],
    )
    graph.search_entities.return_value = [entity]
    graph.get_neighbors.return_value = [(
        Entity(id="pydantic_tech", name="pydantic", entity_type=EntityType.TECHNOLOGY, mention_count=5),
        0.8,
    )]
    return graph


class TestFabrikGraphSearch:
    @pytest.mark.asyncio
    async def test_graph_search_returns_entities(self, mock_graph):
        with patch("src.interfaces.mcp_server.GraphEngine", return_value=mock_graph):
            from src.interfaces.mcp_server import fabrik_graph_search
            result = await fabrik_graph_search(query="FastAPI")

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["entities"][0]["name"] == "fastapi"
        assert "pydantic" in parsed["entities"][0]["neighbors"]

    @pytest.mark.asyncio
    async def test_graph_not_built(self):
        mock_g = MagicMock()
        mock_g.load.return_value = False
        with patch("src.interfaces.mcp_server.GraphEngine", return_value=mock_g):
            from src.interfaces.mcp_server import fabrik_graph_search
            result = await fabrik_graph_search(query="anything")

        parsed = json.loads(result)
        assert parsed["count"] == 0
        assert "not built" in parsed.get("note", "")
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikGraphSearch -v`
Expected: FAIL (fabrik_graph_search not defined)

**Step 3: Implement fabrik_graph_search**

Add to `src/interfaces/mcp_server.py`:

```python
from src.knowledge.graph_engine import GraphEngine


@mcp.tool()
async def fabrik_graph_search(query: str, depth: int = 2, limit: int = 10) -> str:
    """Search the knowledge graph for entities (technologies, patterns, strategies) and their relationships.

    Args:
        query: Entity name or keyword to search
        depth: How deep to traverse relationships (1-5, default 2)
        limit: Maximum entities to return (1-50, default 10)
    """
    depth = max(1, min(5, depth))
    limit = max(1, min(50, limit))

    graph = GraphEngine()
    if not graph.load():
        return json.dumps({
            "entities": [],
            "count": 0,
            "note": "Knowledge graph not built. Run: fabrik graph build",
        })

    entities = graph.search_entities(query, limit=limit)
    results = []
    for e in entities:
        neighbors = graph.get_neighbors(e.id, depth=depth, min_weight=0.3)
        results.append({
            "name": e.name,
            "type": e.entity_type.value,
            "mention_count": e.mention_count,
            "aliases": e.aliases,
            "neighbors": [n.name for n, _ in neighbors[:5]],
        })

    return json.dumps({"entities": results, "count": len(results)}, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikGraphSearch -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/interfaces/mcp_server.py tests/test_mcp_server.py
git commit -m "FEAT: Agregar tool fabrik_graph_search al MCP server"
```

---

### Task 5: Add fabrik_ask tool (LLM query with optional RAG)

**Files:**
- Modify: `src/interfaces/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

Append to `tests/test_mcp_server.py`:

```python
from src.core.llm_client import LLMResponse


def _make_llm_response(content="test answer", model="qwen2.5-coder:7b"):
    return LLMResponse(content=content, model=model, tokens_used=42, latency_ms=100.0)


class TestFabrikAsk:
    @pytest.mark.asyncio
    async def test_basic_ask(self, mock_llm):
        mock_llm.generate = AsyncMock(return_value=_make_llm_response())
        with patch("src.interfaces.mcp_server.LLMClient", return_value=mock_llm):
            from src.interfaces.mcp_server import fabrik_ask
            result = await fabrik_ask(prompt="What is DDD?")

        parsed = json.loads(result)
        assert parsed["answer"] == "test answer"
        assert parsed["model"] == "qwen2.5-coder:7b"

    @pytest.mark.asyncio
    async def test_ask_with_rag(self, mock_llm, mock_rag):
        mock_llm.generate = AsyncMock(return_value=_make_llm_response())
        with patch("src.interfaces.mcp_server.LLMClient", return_value=mock_llm), \
             patch("src.interfaces.mcp_server.RAGEngine", return_value=mock_rag):
            from src.interfaces.mcp_server import fabrik_ask
            result = await fabrik_ask(prompt="What is DDD?", use_rag=True)

        parsed = json.loads(result)
        assert parsed["answer"] == "test answer"
        assert len(parsed["sources"]) > 0
        # Verify RAG context was injected into prompt
        prompt_sent = mock_llm.generate.call_args.args[0]
        assert "context" in prompt_sent.lower()

    @pytest.mark.asyncio
    async def test_ask_ollama_down(self):
        mock = AsyncMock()
        mock.health_check = AsyncMock(return_value=False)
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=None)
        with patch("src.interfaces.mcp_server.LLMClient", return_value=mock):
            from src.interfaces.mcp_server import fabrik_ask
            result = await fabrik_ask(prompt="hello")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "ollama" in parsed["error"].lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikAsk -v`
Expected: FAIL (fabrik_ask not defined)

**Step 3: Implement fabrik_ask**

Add to `src/interfaces/mcp_server.py`:

```python
@mcp.tool()
async def fabrik_ask(prompt: str, use_rag: bool = False, use_graph: bool = False) -> str:
    """Ask Fabrik-Codek a coding question. Optionally includes context from the knowledge base.

    Args:
        prompt: Your question or task description
        use_rag: Include vector search context from the knowledge base
        use_graph: Include hybrid RAG context (vector + knowledge graph)
    """
    async with LLMClient() as llm:
        if not await llm.health_check():
            return json.dumps({"error": "Ollama is not available. Run: ollama serve"})

        final_prompt = prompt
        sources = []

        # Hybrid RAG (vector + graph)
        if use_graph:
            try:
                from src.knowledge.hybrid_rag import HybridRAGEngine
                async with HybridRAGEngine() as hybrid:
                    results = await hybrid.retrieve(prompt, limit=5, graph_depth=2)
                    if results:
                        context = "\n---\n".join(
                            f"[{r.get('category', '?')}] {r['text'][:500]}" for r in results
                        )
                        final_prompt = (
                            f"Context from knowledge base:\n{context}\n\n---\n"
                            f"Question: {prompt}\n\nAnswer using the context when relevant."
                        )
                        sources = [
                            {"source": r.get("source", ""), "category": r.get("category", "")}
                            for r in results
                        ]
            except Exception:
                pass  # Fall through to basic ask

        # Vector RAG only
        elif use_rag:
            try:
                async with RAGEngine() as rag:
                    results = await rag.retrieve(prompt, limit=5)
                    if results:
                        context = "\n---\n".join(
                            f"[{r['category']}] {r['text'][:500]}" for r in results
                        )
                        final_prompt = (
                            f"Context from knowledge base:\n{context}\n\n---\n"
                            f"Question: {prompt}\n\nAnswer using the context when relevant."
                        )
                        sources = [
                            {"source": r.get("source", ""), "category": r.get("category", "")}
                            for r in results
                        ]
            except Exception:
                pass

        response = await llm.generate(final_prompt)
        return json.dumps({
            "answer": response.content,
            "model": response.model,
            "tokens_used": response.tokens_used,
            "latency_ms": round(response.latency_ms, 1),
            "sources": sources,
        }, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikAsk -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/interfaces/mcp_server.py tests/test_mcp_server.py
git commit -m "FEAT: Agregar tool fabrik_ask al MCP server"
```

---

### Task 6: Add fabrik_graph_stats tool and MCP resources

**Files:**
- Modify: `src/interfaces/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

Append to `tests/test_mcp_server.py`:

```python
class TestFabrikGraphStats:
    @pytest.mark.asyncio
    async def test_graph_stats(self, mock_graph):
        mock_graph.get_stats.return_value = {
            "entity_count": 10,
            "edge_count": 25,
            "connected_components": 3,
            "entity_types": {"technology": 5},
            "relation_types": {"uses": 10},
        }
        with patch("src.interfaces.mcp_server.GraphEngine", return_value=mock_graph):
            from src.interfaces.mcp_server import fabrik_graph_stats
            result = await fabrik_graph_stats()

        parsed = json.loads(result)
        assert parsed["entity_count"] == 10
        assert parsed["edge_count"] == 25

    @pytest.mark.asyncio
    async def test_graph_stats_not_built(self):
        mock_g = MagicMock()
        mock_g.load.return_value = False
        with patch("src.interfaces.mcp_server.GraphEngine", return_value=mock_g):
            from src.interfaces.mcp_server import fabrik_graph_stats
            result = await fabrik_graph_stats()

        parsed = json.loads(result)
        assert parsed["entity_count"] == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikGraphStats -v`
Expected: FAIL (fabrik_graph_stats not defined)

**Step 3: Implement fabrik_graph_stats and resources**

Add to `src/interfaces/mcp_server.py`:

```python
@mcp.tool()
async def fabrik_graph_stats() -> str:
    """Get statistics about the knowledge graph: entity counts, relationship types, and graph density."""
    graph = GraphEngine()
    if not graph.load():
        return json.dumps({
            "entity_count": 0,
            "edge_count": 0,
            "connected_components": 0,
            "entity_types": {},
            "relation_types": {},
            "note": "Knowledge graph not built. Run: fabrik graph build",
        })

    stats = graph.get_stats()
    return json.dumps({
        "entity_count": stats["entity_count"],
        "edge_count": stats["edge_count"],
        "connected_components": stats["connected_components"],
        "entity_types": stats["entity_types"],
        "relation_types": stats["relation_types"],
    }, indent=2)


# ---------------------------------------------------------------------------
# Resources (read-only data for agents)
# ---------------------------------------------------------------------------


@mcp.resource("fabrik://status")
async def resource_status() -> str:
    """Current Fabrik-Codek system status."""
    return await fabrik_status()


@mcp.resource("fabrik://graph/stats")
async def resource_graph_stats() -> str:
    """Current knowledge graph statistics."""
    return await fabrik_graph_stats()


@mcp.resource("fabrik://config")
def resource_config() -> str:
    """Current Fabrik-Codek configuration (sanitized, no secrets)."""
    return json.dumps({
        "version": __version__,
        "model": settings.default_model,
        "fallback_model": settings.fallback_model,
        "embedding_model": settings.embedding_model,
        "ollama_host": settings.ollama_host,
        "vector_db": settings.vector_db,
        "flywheel_enabled": settings.flywheel_enabled,
        "api_port": settings.api_port,
        "mcp_port": settings.mcp_port,
    }, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mcp_server.py::TestFabrikGraphStats -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/interfaces/mcp_server.py tests/test_mcp_server.py
git commit -m "FEAT: Agregar tool fabrik_graph_stats y resources al MCP server"
```

---

### Task 7: Add `fabrik mcp` CLI command

**Files:**
- Modify: `src/interfaces/cli.py` (add new command)
- Modify: `tests/test_cli.py` (add test)

**Step 1: Write the failing test**

Check what test pattern `test_cli.py` uses, then add:

```python
class TestMCPCommand:
    def test_mcp_command_exists(self):
        """Verify the mcp command is registered."""
        from src.interfaces.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output or "mcp" in result.output
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py::TestMCPCommand -v`
Expected: FAIL (no such command "mcp")

**Step 3: Add the mcp command to CLI**

Add to `src/interfaces/cli.py` (before the `main` callback at the end):

```python
@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port for SSE transport"),
):
    """Start Fabrik-Codek as an MCP server for agent integration."""
    from src.interfaces.mcp_server import mcp as mcp_server

    if transport == "sse":
        from src.config import settings
        bind_port = port or settings.mcp_port
        console.print(
            Panel.fit(
                f"[bold blue]Fabrik-Codek MCP Server[/bold blue] (SSE)\n"
                f"http://127.0.0.1:{bind_port}/sse",
                subtitle="Ctrl+C to stop",
            )
        )
        mcp_server.run(transport="sse", port=bind_port)
    else:
        # stdio mode - no fancy output (would corrupt the protocol)
        mcp_server.run(transport="stdio")
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py::TestMCPCommand -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/interfaces/cli.py tests/test_cli.py
git commit -m "FEAT: Agregar comando fabrik mcp al CLI"
```

---

### Task 8: Run full test suite and verify nothing broke

**Files:** None (verification only)

**Step 1: Run all existing tests**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All 413+ tests pass (413 original + ~12 new MCP tests)

**Step 2: Verify MCP server starts (stdio, quick exit)**

Run: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | timeout 5 python3 -m src.interfaces.mcp_server 2>/dev/null || true`
Expected: JSON response with server capabilities (or timeout after 5s, which is fine)

**Step 3: Commit all with version bump**

Bump version in `src/__init__.py` from `0.1.0` to `0.2.0`:

```python
__version__ = "0.2.0"
```

```bash
git add src/__init__.py
git commit -m "FEAT: Fabrik-Codek MCP Server v0.2.0"
```

---

### Task 9: Update README and sync to GitHub

**Files:**
- Modify: `README.md` (add MCP section)

**Step 1: Add MCP Server section to README**

After the "API Reference" section, add:

```markdown
## MCP Server

Fabrik-Codek can run as an MCP (Model Context Protocol) server, allowing any compatible agent to use it as a local knowledge base.

### Starting the MCP Server

```bash
# stdio mode (for Claude Code, Cursor, etc.)
fabrik mcp

# SSE mode (for network access)
fabrik mcp --transport sse --port 8421
```

### Configuring in Claude Code

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "fabrik-codek": {
      "command": "fabrik",
      "args": ["mcp"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `fabrik_status` | Check system health (Ollama, RAG, Graph) |
| `fabrik_search` | Semantic search in the knowledge base |
| `fabrik_graph_search` | Search knowledge graph entities and relationships |
| `fabrik_graph_stats` | Get knowledge graph statistics |
| `fabrik_ask` | Ask a question with optional RAG/graph context |

### Available Resources

| URI | Description |
|-----|-------------|
| `fabrik://status` | System component status |
| `fabrik://graph/stats` | Knowledge graph statistics |
| `fabrik://config` | Current configuration (sanitized) |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "DOCS: Agregar seccion MCP Server al README"
```

**Step 3: Sync to GitHub repo**

Copy changed files from original to clean repo, sanitize, and push:

```bash
# Copy modified files
cp src/interfaces/mcp_server.py /path/to/public-repo/src/interfaces/
cp src/interfaces/cli.py /path/to/public-repo/src/interfaces/
cp src/__init__.py /path/to/public-repo/src/
cp tests/test_mcp_server.py /path/to/public-repo/tests/
cp tests/test_cli.py /path/to/public-repo/tests/
cp pyproject.toml /path/to/public-repo/
cp README.md /path/to/public-repo/

# Verify no private data leaked
cd /path/to/project-api
grep -r "qwen2.5-coder:7b\|myproject\|myproject\|example\|[0-9]" src/ tests/ || echo "CLEAN"

# Commit and push
git add -A
git commit -m "FEAT: Add MCP Server for agent integration"
git push origin main
```

---

## Summary

| Task | What | New tests |
|------|------|-----------|
| 1 | Add mcp[cli] dependency | 0 |
| 2 | MCP server + fabrik_status | 2 |
| 3 | fabrik_search tool | 3 |
| 4 | fabrik_graph_search tool | 2 |
| 5 | fabrik_ask tool | 3 |
| 6 | fabrik_graph_stats + resources | 2 |
| 7 | `fabrik mcp` CLI command | 1 |
| 8 | Full test suite verification | 0 |
| 9 | README + GitHub sync | 0 |
| **Total** | | **~13 new tests** |
