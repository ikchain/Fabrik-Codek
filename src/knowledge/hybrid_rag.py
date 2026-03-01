"""Hybrid RAG Engine - Combines vector search (LanceDB) with graph traversal (NetworkX).

Uses Reciprocal Rank Fusion (RRF) to merge results from multiple sources.
Supports three retrieval tiers: vector (semantic), graph (relational), fulltext (keyword).
"""

import re

import structlog

from src.knowledge.graph_engine import GraphEngine
from src.knowledge.rag import RAGEngine

logger = structlog.get_logger()

# Default RRF parameters
DEFAULT_VECTOR_WEIGHT = 0.75
DEFAULT_GRAPH_WEIGHT = 0.25
DEFAULT_FULLTEXT_WEIGHT = 0.0
DEFAULT_RRF_K = 60
DEFAULT_GRAPH_DEPTH = 2
DEFAULT_MIN_WEIGHT = 0.5

# Relevance gate parameters
MAX_NEIGHBORS_PER_SEED = 3
MAX_EXPANSION_QUERIES = 5
ENTITY_MATCH_RATIO = 0.8

# Post-retrieval relevance filter
DEFAULT_RELEVANCE_THRESHOLD = 0.12
_STOPWORDS = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "de",
        "del",
        "en",
        "con",
        "por",
        "para",
        "que",
        "es",
        "son",
        "este",
        "esta",
        "estos",
        "estas",
        "como",
        "qué",
        "cómo",
        "por",
        "más",
        "este",
        "código",
        "code",
        "the",
        "a",
        "an",
        "is",
        "are",
        "this",
        "that",
        "how",
        "what",
        "why",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "with",
        "it",
        "do",
        "does",
        "has",
        "have",
        "not",
        "no",
        "si",
        "se",
        "su",
        "al",
        "lo",
        "le",
        "y",
        "o",
        "hay",
        "tiene",
    }
)


class HybridRAGEngine:
    """Hybrid retrieval engine composing RAGEngine (vector), GraphEngine (graph),
    and optionally FullTextEngine (keyword).

    Does NOT modify any sub-engine. Uses composition to combine retrieval methods.
    """

    def __init__(
        self,
        rag_engine: RAGEngine | None = None,
        graph_engine: GraphEngine | None = None,
        fulltext_engine=None,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT,
        fulltext_weight: float = DEFAULT_FULLTEXT_WEIGHT,
    ):
        self._rag = rag_engine
        self._graph = graph_engine
        self._fulltext = fulltext_engine
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.fulltext_weight = fulltext_weight
        self._owns_rag = rag_engine is None

    async def __aenter__(self):
        if self._rag is None:
            self._rag = RAGEngine()
            await self._rag._init()
        if self._graph is None:
            self._graph = GraphEngine()
            self._graph.load()
        return self

    async def __aexit__(self, *args):
        if self._owns_rag and self._rag:
            await self._rag.close()

    @staticmethod
    def _compute_relevance(query: str, text: str) -> float:
        """Compute token-overlap relevance between query and retrieved text.

        Uses Jaccard-like overlap on meaningful tokens (stopwords removed).
        Returns a score in [0.0, 1.0] where 0 = no overlap, 1 = perfect.
        """

        def _tokenize(s: str) -> set[str]:
            tokens = set(re.findall(r"\b\w{2,}\b", s.lower()))
            return tokens - _STOPWORDS

        query_tokens = _tokenize(query)
        if not query_tokens:
            return 1.0  # Cannot evaluate — let it pass

        text_tokens = _tokenize(text[:500])  # Cap text length for speed
        if not text_tokens:
            return 0.0

        overlap = len(query_tokens & text_tokens)
        # Asymmetric: what fraction of query tokens appear in the text
        return overlap / len(query_tokens)

    async def retrieve(
        self,
        query: str,
        limit: int = 3,
        graph_depth: int = DEFAULT_GRAPH_DEPTH,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        confidence_threshold: float | None = None,
        min_k: int | None = None,
        max_k: int | None = None,
        query_entities: list[str] | None = None,
        min_relevance: float | None = None,
    ) -> list[dict]:
        """Hybrid retrieval using RRF fusion of vector, graph, and fulltext results.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            graph_depth: BFS depth for graph traversal.
            min_weight: Minimum edge weight for graph traversal.
            confidence_threshold: If set, use adaptive retrieval for vector search.
            min_k: Minimum results for adaptive retrieval.
            max_k: Maximum results for adaptive retrieval.
            query_entities: Entity names for confidence entity coverage.
            min_relevance: Minimum query-text token overlap to keep a result.
                If None, uses DEFAULT_RELEVANCE_THRESHOLD.
                Set to 0.0 to disable filtering.

        Returns:
            List of dicts with keys: text, source, category, score, origin.
        """
        logger.debug("hybrid_retrieve_start", query=query[:100], limit=limit)

        # 1. Vector search
        if confidence_threshold is not None:
            adaptive_result = await self._rag.retrieve_adaptive(
                query,
                min_k=min_k or 1,
                max_k=max_k or limit * 2,
                confidence_threshold=confidence_threshold,
                query_entities=query_entities,
            )
            vector_results = adaptive_result["results"]
        else:
            vector_results = await self._rag.retrieve(query, limit=limit * 2)

        # 2. Graph-enhanced retrieval
        graph_results = await self._graph_retrieve(
            query,
            limit=limit * 2,
            depth=graph_depth,
            min_weight=min_weight,
        )

        # 3. Full-text search (when available and weighted)
        fulltext_results = []
        if self._fulltext and self.fulltext_weight > 0:
            try:
                fulltext_results = await self._fulltext.search(query, limit=limit * 2)
            except Exception as exc:
                logger.debug("fulltext_retrieve_failed", error=str(exc))

        # 4. RRF fusion (three sources)
        fused = self._rrf_fusion(vector_results, graph_results, fulltext_results, limit=limit)

        # 5. Post-retrieval relevance filter
        threshold = min_relevance if min_relevance is not None else DEFAULT_RELEVANCE_THRESHOLD
        if threshold > 0 and fused:
            pre_count = len(fused)
            filtered = []
            for result in fused:
                relevance = self._compute_relevance(query, result.get("text", ""))
                result["relevance"] = round(relevance, 3)
                if relevance >= threshold:
                    filtered.append(result)
                else:
                    logger.debug(
                        "relevance_filter_dropped",
                        relevance=round(relevance, 3),
                        threshold=threshold,
                        text_preview=result.get("text", "")[:80],
                    )
            fused = filtered
            if pre_count > len(fused):
                logger.info(
                    "relevance_filter_applied",
                    before=pre_count,
                    after=len(fused),
                    dropped=pre_count - len(fused),
                )

        # 6. Enrich with graph context paths
        entity_ids = self._recognize_entities(query)
        if entity_ids:
            context_paths = self._graph.get_context_paths(entity_ids, max_paths=3)
            if context_paths:
                for result in fused:
                    result["graph_context"] = context_paths

        logger.info(
            "hybrid_retrieve_done",
            vector_count=len(vector_results),
            graph_count=len(graph_results),
            fulltext_count=len(fulltext_results),
            fused_count=len(fused),
            entities_recognized=len(entity_ids),
        )
        return fused

    def _build_expansion_queries(
        self,
        seed_ids: list[str],
        depth: int = 2,
        min_weight: float = DEFAULT_MIN_WEIGHT,
    ) -> list[tuple[str, float]]:
        """Build composite expansion queries from seed + neighbor entity pairs.

        For each seed entity, gets its graph neighbors (capped at
        MAX_NEIGHBORS_PER_SEED) and builds "{seed_name} {neighbor_name}"
        pairs. Deduplicates symmetric pairs (A,B) == (B,A), sorts by
        graph proximity score (highest first), and caps total queries
        at MAX_EXPANSION_QUERIES.

        Returns:
            List of (expansion_query, score) tuples.
        """
        seen_pairs: set[frozenset] = set()
        expansion_queries: list[tuple[str, float]] = []

        for seed_id in seed_ids:
            seed_entity = self._graph.get_entity(seed_id)
            if not seed_entity:
                continue

            neighbors = self._graph.get_neighbors(seed_id, depth=depth, min_weight=min_weight)
            neighbors = neighbors[:MAX_NEIGHBORS_PER_SEED]
            for neighbor_entity, score in neighbors:
                pair_key = frozenset({seed_entity.name, neighbor_entity.name})
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    query = f"{seed_entity.name} {neighbor_entity.name}"
                    expansion_queries.append((query, score))

        # Sort by proximity score (highest first)
        expansion_queries.sort(key=lambda x: x[1], reverse=True)
        return expansion_queries[:MAX_EXPANSION_QUERIES]

    async def _graph_retrieve(
        self,
        query: str,
        limit: int = 10,
        depth: int = 2,
        min_weight: float = 0.3,
    ) -> list[dict]:
        """Retrieve documents via graph-guided expansion queries.

        Instead of searching by entity name alone, builds composite queries
        from graph neighborhoods (e.g., "FastAPI Pydantic") to capture
        relationship context in vector search.
        """
        entity_ids = self._recognize_entities(query)
        if not entity_ids:
            return []

        results = []
        seen_texts: set[str] = set()

        # 1. Graph-guided expansion queries
        expansion_queries = self._build_expansion_queries(
            entity_ids, depth=depth, min_weight=min_weight
        )
        for exp_query, _score in expansion_queries:
            exp_results = await self._rag.retrieve(exp_query, limit=3)
            for r in exp_results:
                text_key = r["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    r["origin"] = "graph"
                    r["graph_expansion"] = exp_query
                    results.append(r)

        # 2. Direct seed entity name search (fallback)
        for eid in entity_ids:
            entity = self._graph.get_entity(eid)
            if not entity:
                continue
            seed_results = await self._rag.retrieve(entity.name, limit=3)
            for r in seed_results:
                text_key = r["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    r["origin"] = "graph"
                    results.append(r)

        return results[:limit]

    def _recognize_entities(self, query: str) -> list[str]:
        """Recognize known entities in the query text.

        Uses word-overlap ratio >= ENTITY_MATCH_RATIO to filter
        partial substring matches that inject noise.
        """
        if not self._graph:
            return []

        entity_ids = []
        query_lower = query.lower()

        # Extract individual words and bigrams
        single_words = re.findall(r"\b\w+\b", query_lower)
        bigrams = [f"{single_words[i]} {single_words[i+1]}" for i in range(len(single_words) - 1)]
        candidates = bigrams + single_words  # Bigrams first for longer matches

        for candidate in candidates:
            candidate = candidate.strip()
            if len(candidate) < 2:
                continue

            entities = self._graph.search_entities(candidate, limit=1)
            if entities:
                entity = entities[0]
                # Word-overlap ratio instead of loose substring match
                entity_words = set(re.findall(r"\w+", entity.name.lower()))
                candidate_words = set(re.findall(r"\w+", candidate.lower()))
                if not entity_words:
                    continue
                overlap = len(entity_words & candidate_words)
                ratio = overlap / len(entity_words)
                if ratio >= ENTITY_MATCH_RATIO:
                    if entity.id not in entity_ids:
                        entity_ids.append(entity.id)

        return entity_ids

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        fulltext_results: list[dict] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two or three result sets.

        score(d) = Σ weight_i / (k + rank_i)  for each source where d appears.
        """
        k = DEFAULT_RRF_K
        scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}
        sources_seen: dict[str, set[str]] = {}

        # Score vector results
        for rank, r in enumerate(vector_results):
            key = r["text"][:100]
            scores[key] = scores.get(key, 0) + self.vector_weight / (k + rank)
            if key not in result_map:
                result_map[key] = {**r, "origin": r.get("origin", "vector")}
                sources_seen[key] = {"vector"}
            else:
                sources_seen[key].add("vector")

        # Score graph results
        for rank, r in enumerate(graph_results):
            key = r["text"][:100]
            scores[key] = scores.get(key, 0) + self.graph_weight / (k + rank)
            if key not in result_map:
                result_map[key] = {**r, "origin": "graph"}
                sources_seen[key] = {"graph"}
            else:
                sources_seen[key].add("graph")

        # Score fulltext results (when present and weighted)
        if fulltext_results and self.fulltext_weight > 0:
            for rank, r in enumerate(fulltext_results):
                key = r["text"][:100]
                scores[key] = scores.get(key, 0) + self.fulltext_weight / (k + rank)
                if key not in result_map:
                    result_map[key] = {**r, "origin": "fulltext"}
                    sources_seen[key] = {"fulltext"}
                else:
                    sources_seen[key].add("fulltext")

        # Mark multi-source results as "hybrid"
        for key, srcs in sources_seen.items():
            if len(srcs) > 1 and key in result_map:
                result_map[key]["origin"] = "hybrid"

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, score in ranked[:limit]:
            if key in result_map:
                result = result_map[key]
                result["score"] = score
                results.append(result)

        return results

    async def query_with_context(
        self,
        query: str,
        limit: int = 3,
        graph_depth: int = DEFAULT_GRAPH_DEPTH,
    ) -> str:
        """Get query with injected hybrid RAG context."""
        results = await self.retrieve(query, limit=limit, graph_depth=graph_depth)

        if not results:
            logger.debug("hybrid_query_no_results", query=query[:100])
            return query

        context_parts = []
        for r in results:
            origin = r.get("origin", "unknown")
            context_parts.append(f"[{r['category']}|{origin}] {r['text'][:300]}")

        # Add graph relationship paths if available
        graph_paths = results[0].get("graph_context", []) if results else []
        if graph_paths:
            context_parts.append(
                "\nRelaciones conocidas:\n" + "\n".join(f"- {p}" for p in graph_paths)
            )

        context = "\n---\n".join(context_parts)

        return f"""Contexto de tu base de conocimiento (vector + grafo):
{context}

---
Pregunta: {query}

Responde usando el contexto cuando sea relevante."""
