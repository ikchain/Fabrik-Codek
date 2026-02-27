# Dynamic Alias Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect entity name duplicates using embedding similarity and merge them automatically, eliminating ghost nodes from the knowledge graph.

**Architecture:** Post-build deduplication step using cosine similarity on entity name embeddings. Same-type-only comparison with configurable threshold. CLI with dry-run support.

**Tech Stack:** Python 3.11+, httpx (Ollama embeddings), numpy-free cosine (dot product), tenacity (retry), structlog

---

### Task 1: AliasPair dataclass + cosine similarity helper

**Files:**
- Modify: `src/knowledge/graph_engine.py`
- Modify: `src/knowledge/graph_schema.py`
- Test: `tests/test_graph_engine.py`

**Step 1: Write the failing test**

In `tests/test_graph_engine.py`, add a new class:

```python
class TestAliasDetection:
    """Tests for dynamic alias detection."""

    def test_cosine_similarity_identical(self):
        from src.knowledge.graph_engine import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        from src.knowledge.graph_engine import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        from src.knowledge.graph_engine import _cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`
Expected: FAIL with ImportError

**Step 3: Implement cosine similarity + AliasPair**

In `src/knowledge/graph_engine.py`, add:

```python
import math

@dataclass
class AliasPair:
    """A pair of entities detected as likely aliases."""
    canonical: Entity
    alias: Entity
    similarity: float
```

```python
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/knowledge/graph_engine.py tests/test_graph_engine.py
git commit -m "FEAT: Add AliasPair dataclass and cosine similarity helper"
```

---

### Task 2: detect_aliases() method

**Files:**
- Modify: `src/knowledge/graph_engine.py`
- Test: `tests/test_graph_engine.py`

**Step 1: Write failing tests**

```python
def test_detect_aliases_same_type(self, engine):
    """Detect aliases between entities of the same type."""
    e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
    e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
    engine.add_entity(e1)
    engine.add_entity(e2)

    # Mock embeddings: very similar vectors
    embeddings = {
        "t1": [0.9, 0.1, 0.0],
        "t2": [0.88, 0.12, 0.01],
    }

    pairs = engine.detect_aliases(embeddings, threshold=0.85)
    assert len(pairs) == 1
    assert pairs[0].canonical.name == "kubernetes"  # Higher mention_count
    assert pairs[0].alias.name == "k8s"

def test_detect_aliases_different_type(self, engine):
    """Do NOT detect aliases across different types."""
    e1 = Entity(id="t1", name="react", entity_type=EntityType.TECHNOLOGY, mention_count=5)
    e2 = Entity(id="c1", name="reactive", entity_type=EntityType.CONCEPT, mention_count=3)
    engine.add_entity(e1)
    engine.add_entity(e2)

    embeddings = {
        "t1": [0.9, 0.1, 0.0],
        "c1": [0.88, 0.12, 0.01],
    }

    pairs = engine.detect_aliases(embeddings, threshold=0.85)
    assert len(pairs) == 0

def test_detect_aliases_below_threshold(self, engine):
    """Do NOT detect when similarity is below threshold."""
    e1 = Entity(id="t1", name="react", entity_type=EntityType.TECHNOLOGY, mention_count=5)
    e2 = Entity(id="t2", name="angular", entity_type=EntityType.TECHNOLOGY, mention_count=3)
    engine.add_entity(e1)
    engine.add_entity(e2)

    # Orthogonal vectors = 0 similarity
    embeddings = {
        "t1": [1.0, 0.0, 0.0],
        "t2": [0.0, 1.0, 0.0],
    }

    pairs = engine.detect_aliases(embeddings, threshold=0.85)
    assert len(pairs) == 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`
Expected: FAIL with AttributeError

**Step 3: Implement detect_aliases()**

In `GraphEngine`:

```python
def detect_aliases(
    self,
    embeddings: dict[str, list[float]],
    threshold: float = 0.85,
) -> list[AliasPair]:
    """Detect entity pairs that are likely aliases based on embedding similarity.

    Only compares entities of the same type. The entity with higher
    mention_count is treated as canonical.
    """
    from collections import defaultdict

    by_type: dict[EntityType, list[Entity]] = defaultdict(list)
    for entity in self._entities.values():
        if entity.id in embeddings:
            by_type[entity.entity_type].append(entity)

    pairs: list[AliasPair] = []
    for group in by_type.values():
        for i, a in enumerate(group):
            for j in range(i + 1, len(group)):
                b = group[j]
                sim = _cosine_similarity(embeddings[a.id], embeddings[b.id])
                if sim >= threshold:
                    if a.mention_count >= b.mention_count:
                        canonical, alias = a, b
                    else:
                        canonical, alias = b, a
                    pairs.append(AliasPair(
                        canonical=canonical,
                        alias=alias,
                        similarity=sim,
                    ))

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/knowledge/graph_engine.py tests/test_graph_engine.py
git commit -m "FEAT: Add detect_aliases() with same-type similarity comparison"
```

---

### Task 3: merge_alias_pair() method

**Files:**
- Modify: `src/knowledge/graph_engine.py`
- Test: `tests/test_graph_engine.py`

**Step 1: Write failing tests**

```python
def test_merge_alias_pair_fields(self, engine):
    """Merge accumulates aliases, mention_count, source_docs."""
    e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY,
                mention_count=10, source_docs=["doc1"], aliases=["kube"])
    e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY,
                mention_count=2, source_docs=["doc2"], aliases=["k8"])
    engine.add_entity(e1)
    engine.add_entity(e2)

    pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
    engine.merge_alias_pair(pair)

    merged = engine.find_entity_by_name("kubernetes")
    assert merged is not None
    assert merged.mention_count == 12
    assert "k8s" in merged.aliases
    assert "k8" in merged.aliases
    assert "kube" in merged.aliases
    assert "doc1" in merged.source_docs
    assert "doc2" in merged.source_docs
    assert engine.find_entity_by_name("k8s") is not None  # Now found via alias

def test_merge_alias_pair_edges(self, engine):
    """Edges from alias entity are redirected to canonical."""
    e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
    e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
    e3 = Entity(id="c1", name="devops", entity_type=EntityType.CONCEPT, mention_count=5)
    engine.add_entity(e1)
    engine.add_entity(e2)
    engine.add_entity(e3)

    # Edge from k8s → devops
    from src.knowledge.graph_schema import Relation, RelationType
    rel = Relation(source_id="t2", target_id="c1",
                   relation_type=RelationType.RELATES_TO, weight=0.8)
    engine.add_relation(rel)

    pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
    engine.merge_alias_pair(pair)

    # Edge should now be kubernetes → devops
    neighbors = engine.get_entity_neighbors("t1")
    assert any(n.name == "devops" for n in neighbors)

def test_merge_alias_pair_removes_alias(self, engine):
    """Alias entity is removed from graph after merge."""
    e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
    e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
    engine.add_entity(e1)
    engine.add_entity(e2)

    pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
    engine.merge_alias_pair(pair)

    assert "t2" not in engine._entities
    assert not engine._graph.has_node("t2")
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`

**Step 3: Implement merge_alias_pair()**

```python
def merge_alias_pair(self, pair: AliasPair) -> None:
    """Merge alias entity into canonical entity.

    Accumulates aliases, mention_count, source_docs. Redirects all edges
    from the alias to the canonical. Removes the alias from the graph.
    """
    canonical = self._entities.get(pair.canonical.id)
    alias = self._entities.get(pair.alias.id)
    if not canonical or not alias:
        return

    # Accumulate fields
    if alias.name not in canonical.aliases and alias.name != canonical.name:
        canonical.aliases.append(alias.name)
    for a in alias.aliases:
        if a not in canonical.aliases and a != canonical.name:
            canonical.aliases.append(a)
    canonical.mention_count += alias.mention_count
    for doc in alias.source_docs:
        if doc not in canonical.source_docs:
            canonical.source_docs.append(doc)
    if alias.description and not canonical.description:
        canonical.description = alias.description

    # Redirect edges
    if self._graph.has_node(alias.id):
        for predecessor in list(self._graph.predecessors(alias.id)):
            if predecessor != canonical.id:
                edge_data = self._graph.edges[predecessor, alias.id]
                if not self._graph.has_edge(predecessor, canonical.id):
                    self._graph.add_edge(predecessor, canonical.id, **edge_data)
        for successor in list(self._graph.successors(alias.id)):
            if successor != canonical.id:
                edge_data = self._graph.edges[alias.id, successor]
                if not self._graph.has_edge(canonical.id, successor):
                    self._graph.add_edge(canonical.id, successor, **edge_data)
        self._graph.remove_node(alias.id)

    # Remove from entities dict
    del self._entities[alias.id]

    # Update canonical in graph
    self._graph.nodes[canonical.id].update(canonical.to_dict())
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/knowledge/graph_engine.py tests/test_graph_engine.py
git commit -m "FEAT: Add merge_alias_pair() with edge redirection"
```

---

### Task 4: deduplicate_aliases() orchestrator + pipeline integration

**Files:**
- Modify: `src/knowledge/graph_engine.py`
- Modify: `src/knowledge/extraction/pipeline.py`
- Test: `tests/test_graph_engine.py`

**Step 1: Write failing tests**

```python
def test_deduplicate_aliases_full(self, engine):
    """End-to-end: detect + merge + stats."""
    e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
    e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
    engine.add_entity(e1)
    engine.add_entity(e2)

    embeddings = {"t1": [0.9, 0.1], "t2": [0.88, 0.12]}
    stats = engine.deduplicate_aliases(embeddings, threshold=0.85, dry_run=False)

    assert stats["candidates"] == 1
    assert stats["merged"] == 1
    assert "t2" not in engine._entities

def test_deduplicate_aliases_dry_run(self, engine):
    """Dry run detects but does not merge."""
    e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
    e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
    engine.add_entity(e1)
    engine.add_entity(e2)

    embeddings = {"t1": [0.9, 0.1], "t2": [0.88, 0.12]}
    stats = engine.deduplicate_aliases(embeddings, threshold=0.85, dry_run=True)

    assert stats["candidates"] == 1
    assert stats["merged"] == 0
    assert "t2" in engine._entities  # NOT removed
```

**Step 2: Run to verify fail**

**Step 3: Implement deduplicate_aliases()**

```python
def deduplicate_aliases(
    self,
    embeddings: dict[str, list[float]],
    threshold: float = 0.85,
    dry_run: bool = True,
) -> dict:
    """Detect and optionally merge alias entities.

    Returns stats dict with candidates found and merges applied.
    """
    pairs = self.detect_aliases(embeddings, threshold)

    stats = {
        "candidates": len(pairs),
        "merged": 0,
        "pairs": [(p.canonical.name, p.alias.name, p.similarity) for p in pairs],
    }

    if dry_run:
        return stats

    for pair in pairs:
        # Skip if alias was already merged in a previous iteration
        if pair.alias.id not in self._entities:
            continue
        self.merge_alias_pair(pair)
        stats["merged"] += 1

    return stats
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_graph_engine.py::TestAliasDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/knowledge/graph_engine.py tests/test_graph_engine.py
git commit -m "FEAT: Add deduplicate_aliases() orchestrator"
```

---

### Task 5: CLI command + pipeline integration

**Files:**
- Modify: `src/interfaces/cli.py` (graph subcommand)
- Modify: `src/knowledge/extraction/pipeline.py`
- Test: `tests/test_cli.py`

**Step 1: Add `aliases` subcommand to CLI**

In the `graph()` command handler, add handling for `aliases` action:

```python
elif action == "aliases":
    from src.knowledge.rag import RAGEngine
    # ... get threshold and dry_run from args
    # Compute embeddings for all entity names
    # Call engine.deduplicate_aliases()
    # Display results table
```

CLI flags:
- `--threshold FLOAT` (default 0.85)
- `--dry-run` (default: True — must pass `--apply` to actually merge)

Use `--apply` flag instead of negating `--dry-run` for safety.

**Step 2: Write CLI tests**

```python
class TestGraphAliases:
    def test_graph_aliases_dry_run(self):
        # Mock graph engine with entities, mock embeddings
        # Verify output contains candidate table
        pass

    def test_graph_aliases_apply(self):
        # Mock graph engine, verify merge is called
        pass
```

**Step 3: Pipeline integration**

In `pipeline.py`, after decay step, add optional alias deduplication.
This should be gated by a flag (not every build needs deduplication).

**Step 4: Run full suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/interfaces/cli.py src/knowledge/extraction/pipeline.py tests/test_cli.py
git commit -m "FEAT: Add graph aliases CLI command and pipeline integration"
```
