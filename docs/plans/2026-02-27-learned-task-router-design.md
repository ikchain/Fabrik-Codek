# Learned Task Router Design

## Problem

The Task Router classifies queries via static keyword matching (8-10 keywords per task type). Queries without known keywords fall back to LLM classification (500ms latency). The system doesn't learn from user interactions.

## Solution

Add a TF-IDF learned classifier as a new classification layer. Trained on outcomes where `outcome=accepted` confirms the task_type was correct. Falls back gracefully when insufficient data (< 50 labeled queries).

### References

- RouterArena: Comparison of LLM Routers (Oct 2025): https://arxiv.org/abs/2510.00202
- Cross-Attention Routing for Cost-Aware LLM Selection (Sep 2025): https://arxiv.org/abs/2509.09782

## Classification Chain (3 levels)

```
query
  │
  ├── 1. Learned classifier (if corpus >= 50, confidence >= 0.3)
  │       → classification_method = "learned"
  │
  ├── 2. Keyword matching (if learned unavailable or low confidence)
  │       → classification_method = "keyword" (existing)
  │
  └── 3. LLM fallback (if keywords also low confidence)
          → classification_method = "llm" (existing)
```

## Changes

### `src/core/task_router.py` — ~120 lines added

**1. `LearnedClassifier` class (~80 lines)**

```python
class LearnedClassifier:
    """TF-IDF based query classifier learned from outcomes."""

    MIN_CORPUS_SIZE: int = 50
    CONFIDENCE_THRESHOLD: float = 0.3

    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path
        self._vectorizer: TfidfVectorizer | None = None
        self._centroids: dict[str, ndarray] | None = None
        self._corpus_size: int = 0
```

Methods:
- `build_corpus(datalake_path: Path) -> int`: Read outcomes, extract (query, task_type) where outcome=accepted, save to corpus_path. Returns corpus size.
- `fit() -> bool`: Load corpus, fit TF-IDF vectorizer, compute centroids per task_type. Returns True if enough data.
- `classify(query: str) -> tuple[str, float]`: Transform query, compute cosine similarity to each centroid, return (best_task_type, confidence).
- `save_corpus(entries: list[dict])` / `load_corpus() -> list[dict]`: JSON persistence.

**2. Corpus format** (`data/profile/router_corpus.json`)

```json
[
  {"query": "fix this connection error", "task_type": "debugging"},
  {"query": "review my API endpoint", "task_type": "code_review"},
  ...
]
```

**3. Integration in `TaskRouter`**

- `TaskRouter.__init__()`: Accept optional `learned_classifier: LearnedClassifier`
- In `_classify()` / `route()`: Try learned first, then keywords, then LLM
- Log `classification_method` in routing decision

**4. Centroid computation**

For each task_type, the centroid is the mean of all TF-IDF vectors for queries of that type:
```python
centroid[task_type] = mean(tfidf_vectors[queries_of_type])
```

Classification: `argmax(cosine_similarity(tfidf(query), centroids))`

### `src/interfaces/cli.py` — ~10 lines changed

- `fabrik competence build`: Also calls `learned_classifier.build_corpus()` + `fit()`
- `fabrik router test`: Shows `classification_method` (learned/keyword/llm) in output

### Dependencies

- `scikit-learn` (TfidfVectorizer, cosine_similarity) — already installed system-wide (1.8.0)
- Import guarded with try/except: if sklearn not available, LearnedClassifier is disabled

### No changes to:
- Keyword classification logic — stays as-is, becomes fallback
- LLM classification — stays as-is, becomes third-level fallback
- RetrievalStrategy — no changes
- MAB integration — no changes

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/core/task_router.py` | LearnedClassifier + integration | ~120 |
| `src/interfaces/cli.py` | competence build corpus + router test display | ~10 |
| `tests/test_task_router.py` | New tests | ~150 |
| **Total** | | **~280** |

## Backward Compatibility

- Without corpus (< 50 entries): behavior identical to current (keyword → LLM)
- Without scikit-learn: LearnedClassifier disabled, logs warning, falls back
- TaskRouter constructor: `learned_classifier` is optional (default None)
- RoutingDecision.classification_method: already exists, values extended to include "learned"
- `fabrik router test` output adds "learned" method, but format unchanged
