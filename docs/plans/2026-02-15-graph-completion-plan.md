# Graph Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add transitive inference to the knowledge graph, materializing implicit DEPENDS_ON and PART_OF chains.

**Architecture:** New `complete()` method on `GraphEngine` iterates all edges of transitive types, finds A->B->C chains, and creates A->C edges with weight=0.3. Pipeline calls it as step 7 after transcripts. CLI shows stats and exposes standalone action.

**Tech Stack:** Python 3.12, NetworkX, pytest, Typer/Rich

---

### Task 1: Write failing tests for GraphEngine.complete()

**Files:**
- Modify: `tests/test_extraction.py` (append new test class at end)

**Step 1: Write the 8 failing tests**

Add this class at the end of `tests/test_extraction.py`:

```python
# --- Graph Completion Tests ---


class TestGraphCompletion:
    """Tests for GraphEngine.complete() - transitive inference."""

    def _add_chain(self, engine, a_name, b_name, c_name, relation_type):
        """Helper: add A->B->C chain of given relation_type."""
        from src.knowledge.graph_schema import Entity, Relation, make_entity_id
        a_id = make_entity_id("technology", a_name)
        b_id = make_entity_id("technology", b_name)
        c_id = make_entity_id("technology", c_name)

        engine.add_entity(Entity(id=a_id, name=a_name, entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=b_id, name=b_name, entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=c_id, name=c_name, entity_type=EntityType.TECHNOLOGY))

        engine.add_relation(Relation(
            source_id=a_id, target_id=b_id,
            relation_type=relation_type, weight=0.7,
            source_docs=["test:1"],
        ))
        engine.add_relation(Relation(
            source_id=b_id, target_id=c_id,
            relation_type=relation_type, weight=0.6,
            source_docs=["test:2"],
        ))
        return a_id, b_id, c_id

    def test_complete_depends_on(self, tmp_dir):
        """A->B->C with DEPENDS_ON -> infers A->C."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "fastapi", "starlette", "uvicorn", RelationType.DEPENDS_ON)

        stats = engine.complete()

        assert engine._graph.has_edge(a_id, c_id)
        edge = engine._graph.edges[a_id, c_id]
        assert edge["relation_type"] == RelationType.DEPENDS_ON.value
        assert stats["inferred_count"] > 0
        assert stats["depends_on_inferred"] > 0

    def test_complete_part_of(self, tmp_dir):
        """A->B->C with PART_OF -> infers A->C."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "router", "api-layer", "backend", RelationType.PART_OF)

        stats = engine.complete()

        assert engine._graph.has_edge(a_id, c_id)
        edge = engine._graph.edges[a_id, c_id]
        assert edge["relation_type"] == RelationType.PART_OF.value
        assert stats["part_of_inferred"] > 0

    def test_complete_no_duplicate(self, tmp_dir):
        """If A->C already exists, complete() does NOT create a duplicate or modify it."""
        from src.knowledge.graph_schema import Entity, Relation, make_entity_id
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "x", "y", "z", RelationType.DEPENDS_ON)

        # Add direct A->C edge with weight=0.8
        engine.add_relation(Relation(
            source_id=a_id, target_id=c_id,
            relation_type=RelationType.DEPENDS_ON, weight=0.8,
            source_docs=["direct:1"],
        ))
        original_weight = engine._graph.edges[a_id, c_id]["weight"]

        stats = engine.complete()

        # Edge should still exist with original weight (not overwritten)
        assert engine._graph.edges[a_id, c_id]["weight"] == original_weight
        assert stats["inferred_count"] == 0

    def test_complete_different_types_no_inference(self, tmp_dir):
        """A depends_on B, B part_of C -> NO inference (different relation types)."""
        from src.knowledge.graph_schema import Entity, Relation, make_entity_id
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")

        a_id = make_entity_id("technology", "svc-a")
        b_id = make_entity_id("technology", "svc-b")
        c_id = make_entity_id("technology", "svc-c")

        engine.add_entity(Entity(id=a_id, name="svc-a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=b_id, name="svc-b", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=c_id, name="svc-c", entity_type=EntityType.TECHNOLOGY))

        engine.add_relation(Relation(
            source_id=a_id, target_id=b_id,
            relation_type=RelationType.DEPENDS_ON, weight=0.7,
        ))
        engine.add_relation(Relation(
            source_id=b_id, target_id=c_id,
            relation_type=RelationType.PART_OF, weight=0.6,
        ))

        stats = engine.complete()

        assert not engine._graph.has_edge(a_id, c_id)
        assert stats["inferred_count"] == 0

    def test_complete_uses_not_transitive(self, tmp_dir):
        """A uses B, B uses C -> NO inference (USES is not transitive)."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "app", "lib", "core", RelationType.USES)

        stats = engine.complete()

        assert not engine._graph.has_edge(a_id, c_id)
        assert stats["inferred_count"] == 0

    def test_complete_stats(self, tmp_dir):
        """Stats return correct counts for multiple inferences."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        # Chain 1: DEPENDS_ON
        self._add_chain(engine, "a1", "b1", "c1", RelationType.DEPENDS_ON)
        # Chain 2: PART_OF
        self._add_chain(engine, "a2", "b2", "c2", RelationType.PART_OF)

        stats = engine.complete()

        assert stats["inferred_count"] == 2
        assert stats["depends_on_inferred"] == 1
        assert stats["part_of_inferred"] == 1

    def test_complete_inferred_metadata(self, tmp_dir):
        """Inferred edges have weight=0.3, source_docs=["inferred:transitive"], metadata={"inferred": true}."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")
        a_id, _, c_id = self._add_chain(engine, "p", "q", "r", RelationType.DEPENDS_ON)

        engine.complete()

        edge = engine._graph.edges[a_id, c_id]
        assert edge["weight"] == 0.3
        assert edge["source_docs"] == ["inferred:transitive"]
        assert edge["metadata"] == {"inferred": True}

    def test_complete_empty_graph(self, tmp_dir):
        """Empty graph -> stats with all 0s, no errors."""
        engine = GraphEngine(data_dir=tmp_dir / "graphdb")

        stats = engine.complete()

        assert stats["inferred_count"] == 0
        assert stats["depends_on_inferred"] == 0
        assert stats["part_of_inferred"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_extraction.py::TestGraphCompletion -v`
Expected: FAIL with `AttributeError: 'GraphEngine' object has no attribute 'complete'`

**Step 3: Commit failing tests**

```bash
git add tests/test_extraction.py
git commit -m "TEST: Añadir 8 tests para GraphEngine.complete()"
```

---

### Task 2: Implement GraphEngine.complete()

**Files:**
- Modify: `src/knowledge/graph_engine.py` (add constants + method before `# --- Persistence ---` section)

**Step 1: Add constants after imports**

After line 20 (`logger = structlog.get_logger()`), add:

```python
TRANSITIVE_RELATIONS = [RelationType.DEPENDS_ON, RelationType.PART_OF]
INFERRED_CONFIDENCE = 0.3
```

**Step 2: Add complete() method**

Add this method to `GraphEngine` class, right before the `# --- Persistence ---` comment (before `save()`, around line 299):

```python
    def complete(self) -> dict:
        """Infer transitive relations for DEPENDS_ON and PART_OF.

        For each transitive relation type, finds A->B->C chains where both
        edges share the same type, and creates A->C if it doesn't exist.

        Only creates new edges (never modifies existing ones).
        Single level of transitivity only (no deeper chains).

        Returns:
            Stats dict with inferred_count, depends_on_inferred, part_of_inferred.
        """
        stats = {
            "inferred_count": 0,
            "depends_on_inferred": 0,
            "part_of_inferred": 0,
        }

        for rel_type in TRANSITIVE_RELATIONS:
            # Collect all edges of this type: source -> [targets]
            edges_by_source: dict[str, list[str]] = {}
            for src, tgt, data in self._graph.edges(data=True):
                if data.get("relation_type") == rel_type.value:
                    edges_by_source.setdefault(src, []).append(tgt)

            # Find A->B->C chains and infer A->C
            inferred = 0
            for a_id, b_ids in edges_by_source.items():
                for b_id in b_ids:
                    c_ids = edges_by_source.get(b_id, [])
                    for c_id in c_ids:
                        if a_id == c_id:
                            continue
                        if self._graph.has_edge(a_id, c_id):
                            continue

                        self._graph.add_edge(
                            a_id,
                            c_id,
                            source_id=a_id,
                            target_id=c_id,
                            relation_type=rel_type.value,
                            weight=INFERRED_CONFIDENCE,
                            source_docs=["inferred:transitive"],
                            metadata={"inferred": True},
                        )
                        inferred += 1

            stats["inferred_count"] += inferred
            stat_key = f"{rel_type.value}_inferred"
            stats[stat_key] = inferred

        if stats["inferred_count"] > 0:
            logger.info(
                "graph_completion_done",
                **stats,
            )

        return stats
```

**Step 3: Run the 8 tests to verify they pass**

Run: `python3 -m pytest tests/test_extraction.py::TestGraphCompletion -v`
Expected: 8 PASSED

**Step 4: Run all tests to verify no regressions**

Run: `python3 -m pytest tests/test_extraction.py -v`
Expected: 96 PASSED (88 existing + 8 new)

**Step 5: Commit**

```bash
git add src/knowledge/graph_engine.py
git commit -m "FEAT: Añadir GraphEngine.complete() para inferencia transitiva"
```

---

### Task 3: Write failing test for Pipeline integration + implement

**Files:**
- Modify: `tests/test_extraction.py` (add 1 integration test to TestGraphCompletion)
- Modify: `src/knowledge/extraction/pipeline.py` (add step 7)

**Step 1: Write the failing integration test**

Add this test to the `TestGraphCompletion` class:

```python
    def test_pipeline_runs_completion(self, tmp_dir):
        """Pipeline build() runs completion, stats include inferred_triples."""
        import asyncio
        from src.knowledge.extraction.pipeline import ExtractionPipeline

        # Create datalake with training pair that creates a DEPENDS_ON chain
        training_dir = tmp_dir / "datalake" / "02-processed" / "training-pairs"
        training_dir.mkdir(parents=True)
        pairs = [
            {
                "instruction": "FastAPI depends on Starlette which depends on uvicorn",
                "output": "FastAPI is built on top of Starlette, and Starlette depends on uvicorn as ASGI server.",
                "category": "api",
            },
        ]
        jsonl_file = training_dir / "test-deps.jsonl"
        with open(jsonl_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        graph_dir = tmp_dir / "graphdb"
        engine = GraphEngine(data_dir=graph_dir)

        with patch("src.knowledge.extraction.pipeline.settings") as mock_settings:
            mock_settings.datalake_path = tmp_dir / "datalake"

            pipeline = ExtractionPipeline(engine=engine, use_llm=False)
            stats = asyncio.run(pipeline.build(force=True))

            # Pipeline should have run completion
            assert "inferred_triples" in stats
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_extraction.py::TestGraphCompletion::test_pipeline_runs_completion -v`
Expected: FAIL with `AssertionError: assert 'inferred_triples' in stats`

**Step 3: Implement pipeline step 7**

In `src/knowledge/extraction/pipeline.py`, in the `build()` method, add step 7 between the transcripts step (step 6) and the save step. After line 178 (`await self._process_transcripts(transcripts_dir, processed_files)`), and before line 181 (`self.engine.save()`), add:

```python
        # 7. Graph completion (transitive inference)
        completion_stats = self.engine.complete()
        self._stats["inferred_triples"] = completion_stats["inferred_count"]
```

Also add `"inferred_triples": 0` to the `_stats` dict initialization in both `__init__` and `build()`.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_extraction.py::TestGraphCompletion::test_pipeline_runs_completion -v`
Expected: PASS

**Step 5: Run all tests**

Run: `python3 -m pytest tests/test_extraction.py -v`
Expected: 97 PASSED (88 + 9 new)

**Step 6: Commit**

```bash
git add src/knowledge/extraction/pipeline.py tests/test_extraction.py
git commit -m "FEAT: Integrar graph completion como paso 7 del pipeline"
```

---

### Task 4: CLI changes - show inferred_triples + standalone complete action

**Files:**
- Modify: `src/interfaces/cli.py` (graph command)

**Step 1: Add inferred_triples to build output table**

In the `graph` command's `run_build()` function, after the existing transcript stats line (around line 448), add:

```python
            if stats.get("inferred_triples", 0) > 0:
                table.add_row(
                    "Triples inferidos",
                    str(stats["inferred_triples"]),
                )
```

**Step 2: Add standalone `complete` action**

In the `graph` command, add a new `elif action == "complete"` block before the final `else` (before line 524):

```python
    elif action == "complete":
        if not engine.load():
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")
            return

        stats = engine.complete()
        engine.save()

        console.print(Panel.fit("[bold green]Graph Completion Done[/bold green]"))
        table = Table()
        table.add_column("Metrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Total inferidos", str(stats["inferred_count"]))
        table.add_row("DEPENDS_ON inferidos", str(stats["depends_on_inferred"]))
        table.add_row("PART_OF inferidos", str(stats["part_of_inferred"]))
        console.print(table)
```

Also update the final `else` help text to include `complete`:

```python
        console.print("[yellow]Uso: graph build | graph search -q 'query' | graph stats | graph complete[/yellow]")
```

**Step 3: Manual smoke test**

Run: `python3 -m fabrik graph complete`
Expected: Either "No hay Knowledge Graph construido" or completion stats table.

**Step 4: Commit**

```bash
git add src/interfaces/cli.py
git commit -m "FEAT: Mostrar triples inferidos en build + accion graph complete"
```

---

### Task 5: Final verification + commit

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/test_extraction.py -v`
Expected: 97 PASSED

**Step 2: Verify no regressions in other test files**

Run: `python3 -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Final commit (if any remaining changes)**

```bash
git status
# If all committed, skip. Otherwise add and commit remaining.
```
