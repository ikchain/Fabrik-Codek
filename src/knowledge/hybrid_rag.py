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
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_GRAPH_WEIGHT = 0.4
DEFAULT_FULLTEXT_WEIGHT = 0.0
DEFAULT_RRF_K = 60
DEFAULT_GRAPH_DEPTH = 2
DEFAULT_MIN_WEIGHT = 0.3


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

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        graph_depth: int = DEFAULT_GRAPH_DEPTH,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        confidence_threshold: float | None = None,
        min_k: int | None = None,
        max_k: int | None = None,
        query_entities: list[str] | None = None,
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

        # 5. Enrich with graph context paths
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
        min_weight: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Build composite expansion queries from seed + neighbor entity pairs.

        For each seed entity, gets its graph neighbors and builds
        "{seed_name} {neighbor_name}" pairs. Deduplicates symmetric pairs
        (A,B) == (B,A) and sorts by graph proximity score (highest first).

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
            for neighbor_entity, score in neighbors:
                pair_key = frozenset({seed_entity.name, neighbor_entity.name})
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    query = f"{seed_entity.name} {neighbor_entity.name}"
                    expansion_queries.append((query, score))

        # Sort by proximity score (highest first)
        expansion_queries.sort(key=lambda x: x[1], reverse=True)
        return expansion_queries

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
        """Recognize known entities in the query text."""
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
                if entity.name in candidate or candidate in entity.name:
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
        limit: int = 5,
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
            context_parts.append(f"[{r['category']}|{origin}] {r['text'][:500]}")

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
