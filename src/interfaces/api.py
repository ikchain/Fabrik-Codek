"""Fabrik-Codek Web API - FastAPI interface for IDE/web integrations."""

import time
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import __version__
from src.config import settings

logger = structlog.get_logger()

# TTL for cached Ollama health check (seconds)
_OLLAMA_HEALTH_TTL = 5.0


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None
    use_rag: bool = False
    use_graph: bool = False
    graph_depth: int = Field(default=2, ge=1, le=5)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = None
    limit: int = Field(default=5, ge=1, le=50)


class GraphSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class StatusResponse(BaseModel):
    ollama: str
    rag: str
    graph: str
    fulltext: str
    model: str
    datalake: str


class AskResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    sources: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0


class SearchResult(BaseModel):
    text: str
    source: str
    category: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    count: int


class EntityResult(BaseModel):
    id: str
    name: str
    entity_type: str
    mention_count: int
    aliases: list[str] = Field(default_factory=list)
    neighbors: list[str] = Field(default_factory=list)


class GraphSearchResponse(BaseModel):
    entities: list[EntityResult]
    count: int


class GraphStatsResponse(BaseModel):
    entity_count: int
    edge_count: int
    connected_components: int
    entity_types: dict[str, int]
    relation_types: dict[str, int]


class FulltextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = None
    limit: int = Field(default=5, ge=1, le=50)


class FulltextSearchResult(BaseModel):
    text: str
    source: str
    category: str
    score: float
    origin: str = "fulltext"


class FulltextSearchResponse(BaseModel):
    results: list[FulltextSearchResult]
    count: int


# ---------------------------------------------------------------------------
# Lifespan: init/cleanup shared resources in app.state
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; tear down on shutdown."""
    from src.core import LLMClient
    from src.knowledge.graph_engine import GraphEngine

    # 1. LLM client (shared httpx session)
    llm = LLMClient()
    await llm.__aenter__()
    app.state.llm = llm

    # 2. Ollama health (non-blocking flag)
    app.state.ollama_ok = await llm.health_check()
    app.state.ollama_checked_at = time.monotonic()
    if not app.state.ollama_ok:
        logger.warning("ollama_unavailable_at_startup")

    # 3. RAG engine (may fail if no datalake)
    try:
        from src.knowledge.rag import RAGEngine

        rag = RAGEngine()
        await rag.__aenter__()
        app.state.rag = rag
    except Exception as exc:
        logger.warning("rag_init_failed", error=str(exc))
        app.state.rag = None

    # 4. Graph engine (sync, load from disk)
    try:
        graph = GraphEngine()
        loaded = graph.load()
        app.state.graph = graph if loaded else None
    except Exception as exc:
        logger.warning("graph_init_failed", error=str(exc))
        app.state.graph = None

    # 4b. Full-text engine (Meilisearch) - optional
    try:
        from src.knowledge.fulltext_engine import FullTextEngine

        ft = FullTextEngine()
        if await ft.health_check():
            await ft.ensure_index()
            app.state.fulltext = ft
        else:
            await ft.close()
            app.state.fulltext = None
            logger.info("fulltext_unavailable", msg="Meilisearch not running, skipping")
    except Exception as exc:
        logger.warning("fulltext_init_failed", error=str(exc))
        app.state.fulltext = None

    # 5. Hybrid RAG (vector + graph + optional fulltext)
    if app.state.rag is not None:
        try:
            from src.knowledge.hybrid_rag import HybridRAGEngine

            hybrid = HybridRAGEngine(
                rag_engine=app.state.rag,
                graph_engine=app.state.graph,
                fulltext_engine=getattr(app.state, "fulltext", None),
                fulltext_weight=settings.fulltext_weight,
            )
            hybrid._owns_rag = False
            app.state.hybrid = hybrid
        except Exception as exc:
            logger.warning("hybrid_rag_init_failed", error=str(exc))
            app.state.hybrid = None
    else:
        app.state.hybrid = None

    # 6. Task Router
    try:
        from src.core.competence_model import get_active_competence_map
        from src.core.personal_profile import get_active_profile
        from src.core.task_router import TaskRouter, load_learned_classifier

        profile = get_active_profile()
        competence_map = get_active_competence_map()
        learned = load_learned_classifier(settings)
        app.state.router = TaskRouter(competence_map, profile, settings, learned_classifier=learned)
    except Exception as exc:
        logger.warning("task_router_init_failed", error=str(exc))
        app.state.router = None

    logger.info(
        "api_startup_complete",
        ollama=app.state.ollama_ok,
        rag=app.state.rag is not None,
        graph=app.state.graph is not None,
        fulltext=getattr(app.state, "fulltext", None) is not None,
        router=app.state.router is not None,
    )

    yield  # ---- application runs ----

    # Shutdown
    if getattr(app.state, "fulltext", None):
        await app.state.fulltext.close()
    if app.state.rag:
        await app.state.rag.close()
    await llm.__aexit__(None, None, None)
    logger.info("api_shutdown_complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fabrik-Codek API",
    version=__version__,
    description="Local AI dev assistant API",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API key authentication middleware
# ---------------------------------------------------------------------------

_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Reject requests without valid API key (when configured)."""
    if not settings.api_key:
        return await call_next(request)

    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    key = request.headers.get("X-API-Key") or _bearer_token(request)
    if key != settings.api_key:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


# ---------------------------------------------------------------------------
# Helper: check Ollama with TTL cache
# ---------------------------------------------------------------------------


async def _ensure_ollama(state) -> None:
    """Raise 503 if Ollama is unreachable. Caches result for _OLLAMA_HEALTH_TTL seconds."""
    now = time.monotonic()
    checked_at = getattr(state, "ollama_checked_at", 0.0)

    if state.ollama_ok:
        return

    # Only re-check if TTL has expired
    if now - checked_at >= _OLLAMA_HEALTH_TTL:
        state.ollama_ok = await state.llm.health_check()
        state.ollama_checked_at = now

    if not state.ollama_ok:
        raise HTTPException(status_code=503, detail="Ollama is not available")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe - always returns 200."""
    return HealthResponse()


@app.get("/status", response_model=StatusResponse)
async def status(request: Request):
    """Status of all components."""
    state = request.app.state
    state.ollama_ok = await state.llm.health_check()
    state.ollama_checked_at = time.monotonic()

    return StatusResponse(
        ollama="ok" if state.ollama_ok else "unavailable",
        rag="ok" if state.rag is not None else "unavailable",
        graph="ok" if state.graph is not None else "unavailable",
        fulltext="ok" if getattr(state, "fulltext", None) is not None else "unavailable",
        model=settings.default_model,
        datalake="ok" if settings.datalake_path.exists() else "unavailable",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    """Ask a question with optional RAG/graph context."""
    state = request.app.state
    await _ensure_ollama(state)

    # Route the query
    router = getattr(state, "router", None)
    decision = await router.route(req.prompt) if router else None
    graph_depth = decision.strategy.graph_depth if decision else req.graph_depth

    prompt = req.prompt
    sources: list[dict] = []

    # Hybrid RAG (vector + graph)
    if req.use_graph and getattr(state, "hybrid", None):
        results = await state.hybrid.retrieve(
            req.prompt,
            limit=5,
            graph_depth=graph_depth,
        )
        if results:
            context = "\n---\n".join(
                f"[{r.get('category', '?')}] {r['text'][:500]}" for r in results
            )
            prompt = (
                f"Context from knowledge base:\n{context}\n\n---\n"
                f"Question: {req.prompt}\n\nAnswer using the context when relevant."
            )
            sources = [
                {
                    "source": r.get("source", ""),
                    "category": r.get("category", ""),
                    "origin": r.get("origin", ""),
                }
                for r in results
            ]

    # Vector RAG only
    elif req.use_rag and getattr(state, "rag", None):
        rag_results = await state.rag.retrieve(req.prompt, limit=5)
        if rag_results:
            context = "\n---\n".join(f"[{r['category']}] {r['text'][:500]}" for r in rag_results)
            prompt = (
                f"Context from knowledge base:\n{context}\n\n---\n"
                f"Question: {req.prompt}\n\nAnswer using the context when relevant."
            )
            sources = [
                {"source": r.get("source", ""), "category": r.get("category", "")}
                for r in rag_results
            ]

    response = await state.llm.generate(
        prompt,
        model=decision.model if decision else req.model,
        system=decision.system_prompt if decision else None,
    )

    return AskResponse(
        answer=response.content,
        model=response.model,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
        sources=sources,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Chat with message history."""
    state = request.app.state
    await _ensure_ollama(state)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    response = await state.llm.chat(messages, model=req.model)

    return ChatResponse(
        reply=response.content,
        model=response.model,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    """Semantic search in the knowledge base."""
    state = request.app.state
    if getattr(state, "rag", None) is None:
        return SearchResponse(results=[], count=0)

    rag_results = await state.rag.retrieve(req.query, limit=req.limit)

    if req.category:
        rag_results = [r for r in rag_results if r.get("category") == req.category]

    results = [
        SearchResult(
            text=r["text"],
            source=r.get("source", ""),
            category=r.get("category", ""),
            score=r.get("score", 0.0),
        )
        for r in rag_results
    ]
    return SearchResponse(results=results, count=len(results))


@app.post("/graph/search", response_model=GraphSearchResponse)
async def graph_search(req: GraphSearchRequest, request: Request):
    """Search entities in the knowledge graph."""
    state = request.app.state
    if getattr(state, "graph", None) is None:
        return GraphSearchResponse(entities=[], count=0)

    entities = state.graph.search_entities(req.query, limit=req.limit)

    entity_results = []
    for e in entities:
        neighbors = state.graph.get_neighbors(e.id, depth=req.depth, min_weight=0.3)
        neighbor_names = [n.name for n, _ in neighbors[:5]]
        entity_results.append(
            EntityResult(
                id=e.id,
                name=e.name,
                entity_type=e.entity_type.value,
                mention_count=e.mention_count,
                aliases=e.aliases,
                neighbors=neighbor_names,
            )
        )

    return GraphSearchResponse(entities=entity_results, count=len(entity_results))


@app.post("/fulltext/search", response_model=FulltextSearchResponse)
async def fulltext_search(req: FulltextSearchRequest, request: Request):
    """Full-text keyword search via Meilisearch."""
    state = request.app.state
    if getattr(state, "fulltext", None) is None:
        return FulltextSearchResponse(results=[], count=0)

    results = await state.fulltext.search(req.query, limit=req.limit, category=req.category)

    return FulltextSearchResponse(
        results=[
            FulltextSearchResult(
                text=r["text"],
                source=r.get("source", ""),
                category=r.get("category", ""),
                score=r.get("score", 0.0),
            )
            for r in results
        ],
        count=len(results),
    )


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats(request: Request):
    """Statistics of the knowledge graph."""
    state = request.app.state
    if getattr(state, "graph", None) is None:
        return GraphStatsResponse(
            entity_count=0,
            edge_count=0,
            connected_components=0,
            entity_types={},
            relation_types={},
        )

    stats = state.graph.get_stats()
    return GraphStatsResponse(
        entity_count=stats["entity_count"],
        edge_count=stats["edge_count"],
        connected_components=stats["connected_components"],
        entity_types=stats["entity_types"],
        relation_types=stats["relation_types"],
    )
