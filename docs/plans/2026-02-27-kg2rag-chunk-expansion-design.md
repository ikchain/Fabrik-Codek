# KG2RAG Graph-Guided Chunk Expansion Design

## Problem

`_graph_retrieve()` in `hybrid_rag.py` recognizes entities and traverses their graph neighborhood, but then **discards** the structural information and falls back to entity-name vector searches. The graph's knowledge of *which concepts relate to what* is lost.

## Solution

Replace the weak entity-name search with **graph-guided expansion queries**. The KG seeds the vector search by building composite queries from seed+neighbor entity pairs, capturing relationship context.

### References

- KG2RAG: Knowledge Graph-Guided RAG (Feb 2025): https://arxiv.org/abs/2502.06864
- Practical GraphRAG at Scale (Jul 2025): https://arxiv.org/abs/2507.03226

## New Retrieval Flow

```
query "How do I handle FastAPI database connections?"
  │
  ├── 1. Recognize seeds: [FastAPI]
  │
  ├── 2. Expand via graph:
  │       FastAPI ──uses──> Pydantic (score: 0.5)
  │       FastAPI ──uses──> SQLAlchemy (score: 0.5)
  │       FastAPI ──related_to──> uvicorn (score: 0.33)
  │
  ├── 3. Build expansion queries (ordered by graph proximity):
  │       "FastAPI Pydantic"      (score 0.5)
  │       "FastAPI SQLAlchemy"    (score 0.5)
  │       "FastAPI uvicorn"       (score 0.33)
  │
  ├── 4. Vector search each expansion → deduplicated results
  │       Also: direct entity name search for seeds (fallback)
  │
  └── 5. RRF fusion with vanilla vector results (unchanged)
```

## Changes

### `hybrid_rag.py` — Rewrite `_graph_retrieve()` (~80 lines)

Replace current entity-name-only search with:

1. **`_build_expansion_queries(seed_ids, depth, min_weight)`** — new method
   - For each seed: get neighbors via `get_neighbors()`
   - Build `"{seed_name} {neighbor_name}"` pairs
   - Deduplicate by frozenset pair key
   - Sort by graph proximity score (closest first)
   - Return `list[tuple[str, float]]`

2. **`_graph_retrieve()` rewrite**
   - Build expansion queries from graph neighborhood
   - Vector search each expansion query (limit=3 per query)
   - Also direct seed entity name search (existing behavior, fallback)
   - Deduplicate by text[:100] key
   - Tag results with `origin="graph"` and `graph_expansion=query`

### No changes to:
- `_rrf_fusion()` — works as-is (rank-based, source-agnostic)
- `_recognize_entities()` — works as-is
- `query_with_context()` — works as-is
- `retrieve()` orchestration — works as-is
- `RAGEngine` — no changes needed
- `GraphEngine` — no changes needed

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/knowledge/hybrid_rag.py` | Rewrite `_graph_retrieve`, add `_build_expansion_queries` | ~80 |
| `tests/test_hybrid_rag.py` | New tests for graph-guided expansion | ~170 |
| **Total** | | **~250** |

## Backward Compatibility

- Method signatures unchanged
- Return format unchanged (`list[dict]` with same keys + optional `graph_expansion`)
- Behavior with empty graph = same (no expansion queries → falls back to empty list)
- RRF fusion unchanged
