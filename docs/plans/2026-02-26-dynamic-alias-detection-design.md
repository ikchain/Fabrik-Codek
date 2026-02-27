# Dynamic Alias Detection

**Date:** 2026-02-26
**Status:** Approved

## Problem

Entity aliases are static dictionaries in `heuristic.py` (`KNOWN_TECHNOLOGIES`,
`KNOWN_PATTERNS`). If a technology appears with an unregistered variant (e.g.
"fast-api" not in aliases, or user writes "Postgres" when only "postgresql" is
canonical), it creates a **duplicate node** in the knowledge graph. The Entity
model has an `aliases` field but it is never auto-populated from extraction.

Current graph: ~180 entities. Risk of duplicates grows with each pipeline build
as new documents introduce new name variants.

## Design Principles

1. **Post-build deduplication** — Don't couple entity creation to embedding calls
2. **Dry-run first** — Show candidates before applying merges
3. **Configurable threshold** — Different domains need different sensitivity
4. **Same-type only** — Never merge entities of different types (tech + concept)
5. **Preserve data** — Merge accumulates aliases, source_docs, mention_counts

## Solution: Post-Build Alias Deduplication

### Approach

After the graph build completes (after temporal decay, before save), run an
optional deduplication step that:

1. Computes embeddings for all entity names
2. Compares entities of the same type by cosine similarity
3. When similarity > threshold, merges them (lower mention_count → alias of higher)

### Algorithm

```python
def detect_aliases(
    entities: list[Entity],
    embeddings: dict[str, list[float]],  # entity_id → embedding
    threshold: float = 0.85,
) -> list[AliasPair]:
    """Find entity pairs that are likely aliases of each other."""
    pairs = []
    # Group by type — only compare within same type
    by_type = defaultdict(list)
    for entity in entities:
        by_type[entity.entity_type].append(entity)

    for entity_type, group in by_type.items():
        for i, a in enumerate(group):
            for j, b in enumerate(group):
                if j <= i:
                    continue
                sim = cosine_similarity(embeddings[a.id], embeddings[b.id])
                if sim >= threshold:
                    # Higher mention_count is canonical
                    canonical = a if a.mention_count >= b.mention_count else b
                    alias_of = b if canonical is a else a
                    pairs.append(AliasPair(
                        canonical=canonical,
                        alias=alias_of,
                        similarity=sim,
                    ))
    return pairs
```

### Merge Logic

When merging entity B into entity A (A is canonical):
- A.aliases += [B.name] + B.aliases (deduplicated)
- A.mention_count += B.mention_count
- A.source_docs += B.source_docs (deduplicated)
- A.description = A.description or B.description
- All edges from/to B are redirected to A
- B is removed from the graph

This reuses the existing `add_entity()` merge behavior for fields, plus adds
edge redirection.

### Data Model

```python
@dataclass
class AliasPair:
    canonical: Entity       # Keep this one
    alias: Entity           # Merge into canonical
    similarity: float       # Cosine similarity score
```

### Integration Points

**Pipeline (`pipeline.py`):**
After `engine.complete()` → `engine.apply_decay()` → NEW: `engine.deduplicate_aliases()` → `engine.save()`

**CLI (`cli.py`):**
`fabrik graph aliases [--threshold 0.85] [--dry-run]`

- `--dry-run` (default): Shows merge candidates table without applying
- Without `--dry-run`: Applies merges and saves graph
- `--threshold`: Cosine similarity threshold (default 0.85)

### Embedding Strategy

Use the same Ollama embedding model already configured (`nomic-embed-text`).
Entity names are short strings (1-3 words) — embeddings for these are fast.

For ~180 entities:
- 180 embedding calls (parallelized in batches of 10, same as `_get_embeddings_batch`)
- N*(N-1)/2 comparisons per type group (cosine is trivial on small vectors)
- Total: <5 seconds for current graph size

### Threshold Selection

Default 0.85 cosine similarity. Rationale:
- "kubernetes" ↔ "k8s": ~0.88 cosine (abbreviation, same concept)
- "postgresql" ↔ "postgres": ~0.92 cosine (partial name)
- "react" ↔ "reactive": ~0.72 cosine (different concepts — NOT merged)
- "angular" ↔ "angularjs": ~0.90 cosine (version variants)

These values are approximate and need validation with the actual embedding model.

## Changes by File

### `src/knowledge/graph_engine.py`

| Change | Description |
|--------|-------------|
| `AliasPair` dataclass | New, after Entity import |
| `detect_aliases()` | New method, computes pairwise similarity within type groups |
| `merge_alias_pair()` | New method, merges alias entity into canonical |
| `deduplicate_aliases()` | New orchestrator, calls detect + merge, returns stats |

### `src/knowledge/extraction/pipeline.py`

| Change | Description |
|--------|-------------|
| `complete()` or build step | Call `engine.deduplicate_aliases()` after decay |

### `src/interfaces/cli.py`

| Change | Description |
|--------|-------------|
| `graph()` command | Add `aliases` subcommand with `--threshold` and `--dry-run` |

### `tests/test_graph_engine.py`

| Test | Validates |
|------|-----------|
| `test_detect_aliases_same_type` | Same-type entities with high similarity detected |
| `test_detect_aliases_different_type` | Cross-type entities NOT detected |
| `test_detect_aliases_below_threshold` | Similarity below threshold not detected |
| `test_merge_alias_pair_fields` | Name, aliases, mention_count, source_docs merged |
| `test_merge_alias_pair_edges` | Edges redirected from alias to canonical |
| `test_merge_alias_pair_removes_alias` | Alias entity removed from graph |
| `test_deduplicate_aliases_full` | End-to-end: detect + merge + stats |
| `test_deduplicate_aliases_dry_run` | Dry run: detect but no merge |

### `tests/test_cli.py`

| Test | Validates |
|------|-----------|
| `test_graph_aliases_dry_run` | CLI output shows candidates table |
| `test_graph_aliases_apply` | CLI applies merges |

## Error Handling

- **Ollama unavailable for embeddings**: Log warning, skip deduplication, continue
  pipeline. Graph is still valid without deduplication.
- **Empty graph**: Return early with 0 candidates.
- **Self-match**: Filtered by `j <= i` condition — entity never compared to itself.

## Out of Scope

- Real-time alias detection during `add_entity()` (too coupled to HTTP)
- Manual alias management CLI (can be added later)
- Cross-type alias detection (a technology named same as a concept)
-  (Semantic Drift Detection) — depends on this but is separate ticket

## Alternatives Considered

### B: Inline during `add_entity()`
Rejected. Would couple the synchronous `GraphEngine.add_entity()` to async HTTP
embedding calls. Architecturally wrong — graph_engine should stay pure data structure.

### C: Sidecar candidate file
Rejected. Adds UX friction. The `--dry-run` flag achieves the same goal (review
before apply) without a separate file format.
