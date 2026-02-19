"""Hybrid RAG Engine - Combines vector search (LanceDB) with graph traversal (NetworkX).

Uses Reciprocal Rank Fusion (RRF) to merge results from both sources.
"""

import re

import structlog

from src.config import settings
from src.knowledge.graph_engine import GraphEngine
from src.knowledge.graph_schema import EntityType
from src.knowledge.rag import RAGEngine

logger = structlog.get_logger()

# Default RRF parameters
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_GRAPH_WEIGHT = 0.4
DEFAULT_RRF_K = 60
DEFAULT_GRAPH_DEPTH = 2
DEFAULT_MIN_WEIGHT = 0.3


class HybridRAGEngine:
    """Hybrid retrieval engine composing RAGEngine (vector) and GraphEngine (graph).

    Does NOT modify RAGEngine. Uses composition to combine both retrieval methods.
    """

    def __init__(
        self,
        rag_engine: RAGEngine | None = None,
        graph_engine: GraphEngine | None = None,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT,
    ):
        self._rag = rag_engine
        self._graph = graph_engine
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
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
    ) -> list[dict]:
        """Hybrid retrieval using RRF fusion of vector and graph results.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            graph_depth: BFS depth for graph traversal.
            min_weight: Minimum edge weight for graph traversal.

        Returns:
            List of dicts with keys: text, source, category, score, origin.
        """
        logger.debug("hybrid_retrieve_start", query=query[:100], limit=limit)

        # 1. Vector search
        vector_results = await self._rag.retrieve(query, limit=limit * 2)

        # 2. Graph-enhanced retrieval
        graph_results = await self._graph_retrieve(
            query, limit=limit * 2, depth=graph_depth, min_weight=min_weight,
        )

        # 3. RRF fusion
        fused = self._rrf_fusion(vector_results, graph_results, limit=limit)

        # 4. Enrich with graph context paths
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
            fused_count=len(fused),
            entities_recognized=len(entity_ids),
        )
        return fused

    async def _graph_retrieve(
        self,
        query: str,
        limit: int = 10,
        depth: int = 2,
        min_weight: float = 0.3,
    ) -> list[dict]:
        """Retrieve documents via graph traversal from recognized entities."""
        entity_ids = self._recognize_entities(query)
        if not entity_ids:
            return []

        # Collect source doc IDs from graph neighborhood
        all_doc_ids: set[str] = set()
        for eid in entity_ids:
            doc_ids = self._graph.get_source_docs_from_neighbors(
                eid, depth=depth, min_weight=min_weight,
            )
            all_doc_ids.update(doc_ids)

        if not all_doc_ids:
            return []

        # Fetch actual documents from vector DB using source doc IDs
        # We use vector search as fallback - the doc IDs from graph are training pair IDs
        # which may not directly map to vector DB. Use the entity names as search queries.
        results = []
        seen_texts: set[str] = set()

        for eid in entity_ids:
            entity = self._graph.get_entity(eid)
            if not entity:
                continue

            # Search vector DB using entity name
            entity_results = await self._rag.retrieve(entity.name, limit=limit // len(entity_ids) + 1)
            for r in entity_results:
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
        single_words = re.findall(r'\b\w+\b', query_lower)
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
        limit: int = 5,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two result sets.

        score(d) = vector_weight/(k + rank_vector) + graph_weight/(k + rank_graph)
        """
        k = DEFAULT_RRF_K
        scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        # Score vector results
        for rank, r in enumerate(vector_results):
            key = r["text"][:100]  # Dedup key
            scores[key] = scores.get(key, 0) + self.vector_weight / (k + rank)
            if key not in result_map:
                result_map[key] = {**r, "origin": r.get("origin", "vector")}

        # Score graph results
        for rank, r in enumerate(graph_results):
            key = r["text"][:100]
            scores[key] = scores.get(key, 0) + self.graph_weight / (k + rank)
            if key not in result_map:
                result_map[key] = {**r, "origin": "graph"}
            elif result_map[key].get("origin") == "vector":
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
            context_parts.append("\nRelaciones conocidas:\n" + "\n".join(f"- {p}" for p in graph_paths))

        context = "\n---\n".join(context_parts)

        return f"""Contexto de tu base de conocimiento (vector + grafo):
{context}

---
Pregunta: {query}

Responde usando el contexto cuando sea relevante."""
