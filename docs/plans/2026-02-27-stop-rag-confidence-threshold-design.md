# Stop-RAG Dynamic Confidence Threshold Design

## Problem

`RAGEngine.retrieve()` always returns a fixed `limit` of results regardless of query complexity or result quality. Simple queries get noisy low-relevance chunks, complex queries may need more context than the default top-5.

## Solution

Add `retrieve_adaptive()` to RAGEngine that fetches `max_k` results but returns only as many as needed based on a confidence threshold. Simplified version of Stop-RAG — no RL, just threshold-based stopping.

### References

- Stop-RAG: Value-Based Retrieval Control (Oct 2025): https://arxiv.org/abs/2510.14337
- R3-RAG: Step-by-Step Reasoning and Retrieval via RL (May 2025): https://arxiv.org/abs/2505.23794

## Adaptive Retrieval Flow

```
retrieve_adaptive(query, min_k=1, max_k=8, confidence_threshold=0.7)
  │
  ├── 1. Get embedding, fetch max_k results from LanceDB (single query)
  │
  ├── 2. Convert _distance to similarity: sim = 1.0 - distance
  │       Filter out results with similarity < MIN_SIMILARITY_FLOOR (0.2)
  │
  ├── 3. Iterate sorted results, accumulating confidence:
  │       similarity_confidence = avg(similarities so far)
  │       entity_coverage = |entities_in_results ∩ query_entities| / |query_entities|
  │       combined = 0.6 * similarity_confidence + 0.4 * entity_coverage
  │
  ├── 4. Stop when: (combined >= threshold AND count >= min_k) OR count >= max_k
  │
  └── 5. Return results + metadata (chunks_retrieved, confidence, stop_reason)
```

## Changes

### `src/knowledge/rag.py` — ~80 lines added

**1. Constants**

```python
MIN_SIMILARITY_FLOOR: float = 0.2
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
DEFAULT_MIN_K: int = 1
DEFAULT_MAX_K: int = 8
SIMILARITY_WEIGHT: float = 0.6
COVERAGE_WEIGHT: float = 0.4
```

**2. `retrieve_adaptive()` — new method**

```python
async def retrieve_adaptive(
    self,
    query: str,
    min_k: int = DEFAULT_MIN_K,
    max_k: int = DEFAULT_MAX_K,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    query_entities: list[str] | None = None,
    category: str | None = None,
) -> dict:
    """Adaptive retrieval with confidence-based stopping.

    Returns dict with keys:
        results: list[dict] — same format as retrieve()
        chunks_retrieved: int
        confidence: float
        stop_reason: str — "threshold" or "max_k" or "no_results"
    """
```

- Fetch `max_k` results from LanceDB (single embedding + single query)
- Convert `_distance` to similarity: `similarity = 1.0 - distance` (LanceDB cosine distance)
- Filter results with `similarity < MIN_SIMILARITY_FLOOR`
- Iterate results sorted by similarity (best first):
  - Compute `_compute_confidence()` on accumulated results
  - If `confidence >= threshold` AND `count >= min_k`: stop with reason "threshold"
  - If `count >= max_k`: stop with reason "max_k"
- Log metrics: chunks_retrieved, confidence, stop_reason
- Return dict with results + metadata

**3. `_compute_confidence()` — new method**

```python
def _compute_confidence(
    self,
    similarities: list[float],
    result_texts: list[str],
    query_entities: list[str] | None = None,
) -> float:
```

- `similarity_confidence = mean(similarities)`
- If `query_entities` provided:
  - `entity_coverage = |entities found in result texts| / |query_entities|`
  - `combined = SIMILARITY_WEIGHT * similarity_confidence + COVERAGE_WEIGHT * entity_coverage`
- If no entities: `combined = similarity_confidence`
- Return `combined`

**4. Existing `retrieve()` — unchanged**

Backward compatible. `retrieve()` keeps working as-is (fixed top-k).

### `src/core/task_router.py` — ~15 lines added

**1. Add fields to RetrievalStrategy**

```python
@dataclass
class RetrievalStrategy:
    # ... existing fields ...
    confidence_threshold: float = 0.7
    min_k: int = 1
    max_k: int = 8
```

**2. Add per-task defaults in TASK_STRATEGIES**

| Task Type | confidence_threshold | min_k | max_k | Rationale |
|-----------|---------------------|-------|-------|-----------|
| debugging | 0.6 | 2 | 8 | Need context, lower threshold |
| code_review | 0.7 | 1 | 5 | Standard |
| architecture | 0.5 | 2 | 8 | Complex, need lots of context |
| explanation | 0.8 | 1 | 4 | Simple, high threshold |
| testing | 0.7 | 1 | 5 | Standard |
| devops | 0.6 | 1 | 6 | Moderate context |
| ml_engineering | 0.6 | 2 | 8 | Complex domain |
| general | 0.7 | 1 | 5 | Default |

### `src/knowledge/hybrid_rag.py` — ~10 lines changed

In `retrieve()`, pass adaptive parameters from the caller:
- Add optional `confidence_threshold`, `min_k`, `max_k` params to `retrieve()`
- Use `retrieve_adaptive()` for the main vector search step instead of `retrieve()`
- Graph retrieval (`_graph_retrieve`) keeps using regular `retrieve()` (expansion queries are narrow)

### No changes to:
- `query_with_context()` — works as-is
- `_rrf_fusion()` — works as-is
- `_graph_retrieve()` / `_build_expansion_queries()` — work as-is
- CLI chat/ask core logic — confidence logging is via structlog (already captured)

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/knowledge/rag.py` | retrieve_adaptive, _compute_confidence, constants | ~80 |
| `src/core/task_router.py` | RetrievalStrategy fields + TASK_STRATEGIES defaults | ~15 |
| `src/knowledge/hybrid_rag.py` | Use retrieve_adaptive in vector step | ~10 |
| `tests/test_rag.py` | New tests for adaptive retrieval | ~120 |
| `tests/test_hybrid_rag.py` | Test adaptive params passthrough | ~20 |
| **Total** | | **~245** |

## Backward Compatibility

- `retrieve()` unchanged — same signature, same behavior
- `RetrievalStrategy` new fields have defaults matching current behavior
- `HybridRAGEngine.retrieve()` new params are optional with current defaults
- All callers that don't pass adaptive params get identical behavior to before
