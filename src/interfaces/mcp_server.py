"""Fabrik-Codek MCP Server - Model Context Protocol interface.

Exposes Fabrik-Codek capabilities as MCP tools and resources for
integration with Claude Code and other MCP-compatible clients.
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from mcp.server.fastmcp import FastMCP

from src import __version__
from src.config import settings

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Lifespan: init/cleanup shared resources
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialise shared resources on startup; tear down on shutdown."""
    from src.core import LLMClient
    from src.knowledge.graph_engine import GraphEngine

    # 1. LLM client
    llm = LLMClient()
    await llm.__aenter__()
    _state["llm"] = llm

    # 2. Ollama health
    try:
        _state["ollama_ok"] = await llm.health_check()
    except Exception:
        _state["ollama_ok"] = False

    # 3. RAG engine
    try:
        from src.knowledge.rag import RAGEngine

        rag = RAGEngine()
        await rag.__aenter__()
        _state["rag"] = rag
    except Exception as exc:
        logger.warning("mcp_rag_init_failed", error=str(exc))
        _state["rag"] = None

    # 4. Graph engine
    try:
        graph = GraphEngine()
        loaded = graph.load()
        _state["graph"] = graph if loaded else None
    except Exception as exc:
        logger.warning("mcp_graph_init_failed", error=str(exc))
        _state["graph"] = None

    # 4b. Full-text engine (Meilisearch) - optional
    try:
        from src.knowledge.fulltext_engine import FullTextEngine

        ft = FullTextEngine()
        if await ft.health_check():
            await ft.ensure_index()
            _state["fulltext"] = ft
        else:
            await ft.close()
            _state["fulltext"] = None
            logger.info("mcp_fulltext_unavailable", msg="Meilisearch not running, skipping")
    except Exception as exc:
        logger.warning("mcp_fulltext_init_failed", error=str(exc))
        _state["fulltext"] = None

    # 5. Hybrid RAG (vector + graph + optional fulltext)
    if _state.get("rag") is not None:
        try:
            from src.knowledge.hybrid_rag import HybridRAGEngine

            hybrid = HybridRAGEngine(
                rag_engine=_state["rag"],
                graph_engine=_state["graph"],
                fulltext_engine=_state.get("fulltext"),
                fulltext_weight=settings.fulltext_weight,
            )
            hybrid._owns_rag = False
            _state["hybrid"] = hybrid
        except Exception as exc:
            logger.warning("mcp_hybrid_init_failed", error=str(exc))
            _state["hybrid"] = None
    else:
        _state["hybrid"] = None

    # 6. Task Router
    try:
        from src.core.competence_model import get_active_competence_map
        from src.core.personal_profile import get_active_profile
        from src.core.task_router import TaskRouter, load_learned_classifier

        profile = get_active_profile()
        competence_map = get_active_competence_map()
        learned = load_learned_classifier(settings)
        _state["router"] = TaskRouter(competence_map, profile, settings, learned_classifier=learned)
    except Exception as exc:
        logger.warning("mcp_router_init_failed", error=str(exc))
        _state["router"] = None

    logger.info(
        "mcp_startup_complete",
        ollama=_state.get("ollama_ok"),
        rag=_state.get("rag") is not None,
        graph=_state.get("graph") is not None,
        fulltext=_state.get("fulltext") is not None,
        router=_state.get("router") is not None,
    )

    yield

    # Shutdown
    if _state.get("fulltext"):
        await _state["fulltext"].close()
    if _state.get("rag"):
        await _state["rag"].close()
    await llm.__aexit__(None, None, None)
    _state.clear()
    logger.info("mcp_shutdown_complete")


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="fabrik-codek",
    host=settings.api_host,
    port=settings.mcp_port,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="fabrik_status",
    description="Check system health: Ollama, RAG, Graph, Datalake status.",
)
async def fabrik_status() -> str:
    """Check system health of all Fabrik-Codek components."""
    llm = _state.get("llm")

    # Re-check Ollama health
    ollama_ok = False
    if llm:
        try:
            ollama_ok = await llm.health_check()
        except Exception:
            ollama_ok = False

    result = {
        "ollama": "ok" if ollama_ok else "unavailable",
        "rag": "ok" if _state.get("rag") is not None else "unavailable",
        "graph": "ok" if _state.get("graph") is not None else "unavailable",
        "fulltext": "ok" if _state.get("fulltext") is not None else "unavailable",
        "model": settings.default_model,
        "datalake": "ok" if settings.datalake_path.exists() else "unavailable",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(result)


@mcp.tool(
    name="fabrik_search",
    description=(
        "Semantic vector search in the knowledge base. "
        "Returns relevant documents with source, category, and relevance score."
    ),
)
async def fabrik_search(query: str, limit: int = 5, category: str | None = None) -> str:
    """Search the knowledge base using vector similarity."""
    limit = max(1, min(limit, 50))

    rag = _state.get("rag")
    if rag is None:
        return json.dumps({"results": [], "count": 0, "error": "RAG engine not available"})

    try:
        results = await rag.retrieve(query, limit=limit)
    except Exception as exc:
        return json.dumps({"results": [], "count": 0, "error": str(exc)})

    if category:
        results = [r for r in results if r.get("category") == category]

    output = {
        "results": [
            {
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "category": r.get("category", ""),
                "score": r.get("score", 0.0),
            }
            for r in results
        ],
        "count": len(results),
    }
    return json.dumps(output)


@mcp.tool(
    name="fabrik_graph_search",
    description=(
        "Search entities in the knowledge graph. "
        "Returns matching entities with their type, aliases, and neighbors."
    ),
)
async def fabrik_graph_search(query: str, limit: int = 10, depth: int = 2) -> str:
    """Search entities in the knowledge graph."""
    limit = max(1, min(limit, 50))
    depth = max(1, min(depth, 5))

    graph = _state.get("graph")
    if graph is None:
        return json.dumps({"entities": [], "count": 0, "error": "Graph engine not available"})

    try:
        entities = graph.search_entities(query, limit=limit)
    except Exception as exc:
        return json.dumps({"entities": [], "count": 0, "error": str(exc)})

    entity_results = []
    for e in entities:
        try:
            neighbors = graph.get_neighbors(e.id, depth=depth, min_weight=0.3)
            neighbor_names = [n.name for n, _ in neighbors[:5]]
        except Exception:
            neighbor_names = []

        entity_results.append(
            {
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type.value,
                "mention_count": e.mention_count,
                "aliases": e.aliases,
                "neighbors": neighbor_names,
            }
        )

    return json.dumps({"entities": entity_results, "count": len(entity_results)})


@mcp.tool(
    name="fabrik_ask",
    description=(
        "Ask a question to the local LLM with optional RAG/graph context. "
        "Set use_rag=True for vector search context, use_graph=True for "
        "hybrid (vector + graph) context."
    ),
)
async def fabrik_ask(
    prompt: str,
    model: str | None = None,
    use_rag: bool = False,
    use_graph: bool = False,
    graph_depth: int = 2,
) -> str:
    """Ask a question with optional RAG/graph context."""
    graph_depth = max(1, min(graph_depth, 5))

    llm = _state.get("llm")
    if llm is None:
        return json.dumps({"error": "LLM client not available"})

    # Check Ollama health
    try:
        ollama_ok = await llm.health_check()
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        return json.dumps({"error": "Ollama is not available"})

    # Route the query
    router = _state.get("router")
    decision = await router.route(prompt) if router else None
    effective_depth = decision.strategy.graph_depth if decision else graph_depth

    effective_prompt = prompt
    sources: list[dict] = []

    # Hybrid RAG (vector + graph)
    if use_graph and _state.get("hybrid"):
        try:
            results = await _state["hybrid"].retrieve(
                prompt,
                limit=5,
                graph_depth=effective_depth,
            )
            if results:
                context = "\n---\n".join(
                    f"[{r.get('category', '?')}] {r['text'][:500]}" for r in results
                )
                effective_prompt = (
                    f"Context from knowledge base:\n{context}\n\n---\n"
                    f"Question: {prompt}\n\nAnswer using the context when relevant."
                )
                sources = [
                    {
                        "source": r.get("source", ""),
                        "category": r.get("category", ""),
                        "origin": r.get("origin", ""),
                    }
                    for r in results
                ]
        except Exception as exc:
            logger.warning("mcp_hybrid_retrieve_failed", error=str(exc))

    # Vector RAG only
    elif use_rag and _state.get("rag"):
        try:
            rag_results = await _state["rag"].retrieve(prompt, limit=5)
            if rag_results:
                context = "\n---\n".join(
                    f"[{r['category']}] {r['text'][:500]}" for r in rag_results
                )
                effective_prompt = (
                    f"Context from knowledge base:\n{context}\n\n---\n"
                    f"Question: {prompt}\n\nAnswer using the context when relevant."
                )
                sources = [
                    {
                        "source": r.get("source", ""),
                        "category": r.get("category", ""),
                    }
                    for r in rag_results
                ]
        except Exception as exc:
            logger.warning("mcp_rag_retrieve_failed", error=str(exc))

    try:
        response = await llm.generate(
            effective_prompt,
            model=decision.model if decision else model,
            system=decision.system_prompt if decision else None,
        )
    except Exception as exc:
        return json.dumps({"error": f"LLM generation failed: {exc}"})

    result = {
        "answer": response.content,
        "model": response.model,
        "tokens_used": response.tokens_used,
        "latency_ms": response.latency_ms,
        "sources": sources,
        "routing": (
            {
                "task_type": decision.task_type,
                "topic": decision.topic,
                "competence": decision.competence_level,
                "method": decision.classification_method,
            }
            if decision
            else None
        ),
    }
    return json.dumps(result)


@mcp.tool(
    name="fabrik_graph_stats",
    description="Get knowledge graph statistics: entity count, edges, components, type breakdown.",
)
async def fabrik_graph_stats() -> str:
    """Get statistics of the knowledge graph."""
    graph = _state.get("graph")
    if graph is None:
        return json.dumps(
            {
                "entity_count": 0,
                "edge_count": 0,
                "connected_components": 0,
                "entity_types": {},
                "relation_types": {},
                "error": "Graph engine not available",
            }
        )

    try:
        stats = graph.get_stats()
    except Exception as exc:
        return json.dumps(
            {
                "entity_count": 0,
                "edge_count": 0,
                "connected_components": 0,
                "entity_types": {},
                "relation_types": {},
                "error": str(exc),
            }
        )

    return json.dumps(
        {
            "entity_count": stats["entity_count"],
            "edge_count": stats["edge_count"],
            "connected_components": stats["connected_components"],
            "entity_types": stats["entity_types"],
            "relation_types": stats["relation_types"],
        }
    )


@mcp.tool(
    name="fabrik_fulltext_search",
    description=(
        "Full-text keyword search in the knowledge base via Meilisearch. "
        "Best for exact matches: error messages, function names, config keys. "
        "Complements semantic vector search (fabrik_search) for precision queries."
    ),
)
async def fabrik_fulltext_search(
    query: str,
    limit: int = 5,
    category: str | None = None,
) -> str:
    """Full-text search using Meilisearch."""
    limit = max(1, min(limit, 50))

    fulltext = _state.get("fulltext")
    if fulltext is None:
        return json.dumps({"results": [], "count": 0, "error": "Full-text engine not available"})

    try:
        results = await fulltext.search(query, limit=limit, category=category)
    except Exception as exc:
        return json.dumps({"results": [], "count": 0, "error": str(exc)})

    output = {
        "results": [
            {
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "category": r.get("category", ""),
                "score": r.get("score", 0.0),
                "origin": "fulltext",
            }
            for r in results
        ],
        "count": len(results),
    }
    return json.dumps(output)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource(
    "fabrik://status",
    name="fabrik_status_resource",
    description="Current system health status of all Fabrik-Codek components.",
    mime_type="application/json",
)
async def status_resource() -> str:
    """Delegates to fabrik_status tool."""
    return await fabrik_status()


@mcp.resource(
    "fabrik://graph/stats",
    name="fabrik_graph_stats_resource",
    description="Knowledge graph statistics: entities, edges, components.",
    mime_type="application/json",
)
async def graph_stats_resource() -> str:
    """Delegates to fabrik_graph_stats tool."""
    return await fabrik_graph_stats()


@mcp.resource(
    "fabrik://config",
    name="fabrik_config_resource",
    description="Sanitized configuration (no secrets).",
    mime_type="application/json",
)
async def config_resource() -> str:
    """Return sanitized configuration without secrets."""
    config = {
        "default_model": settings.default_model,
        "fallback_model": settings.fallback_model,
        "embedding_model": settings.embedding_model,
        "ollama_host": settings.ollama_host,
        "vector_db": settings.vector_db,
        "flywheel_enabled": settings.flywheel_enabled,
        "datalake_path": str(settings.datalake_path),
        "api_port": settings.api_port,
        "mcp_port": settings.mcp_port,
        "version": __version__,
    }
    return json.dumps(config)
