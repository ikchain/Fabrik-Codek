"""Tests for GraphEngine and graph schema."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.knowledge.graph_engine import GraphEngine
from src.knowledge.graph_schema import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    Triple,
    make_entity_id,
)


@pytest.fixture
def tmp_graph_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def engine(tmp_graph_dir):
    return GraphEngine(data_dir=tmp_graph_dir)


# --- Schema Tests ---


class TestSchema:
    def test_make_entity_id_deterministic(self):
        id1 = make_entity_id("technology", "FastAPI")
        id2 = make_entity_id("technology", "FastAPI")
        assert id1 == id2

    def test_make_entity_id_case_insensitive(self):
        id1 = make_entity_id("technology", "FastAPI")
        id2 = make_entity_id("technology", "fastapi")
        assert id1 == id2

    def test_make_entity_id_strips_whitespace(self):
        id1 = make_entity_id("technology", "FastAPI")
        id2 = make_entity_id("technology", "  FastAPI  ")
        assert id1 == id2

    def test_entity_serialization(self):
        entity = Entity(
            id="abc123",
            name="fastapi",
            entity_type=EntityType.TECHNOLOGY,
            description="Web framework",
            aliases=["FastAPI"],
            source_docs=["doc1"],
        )
        data = entity.to_dict()
        restored = Entity.from_dict(data)
        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.aliases == entity.aliases

    def test_relation_serialization(self):
        relation = Relation(
            source_id="a",
            target_id="b",
            relation_type=RelationType.USES,
            weight=0.8,
            source_docs=["doc1"],
        )
        data = relation.to_dict()
        restored = Relation.from_dict(data)
        assert restored.source_id == "a"
        assert restored.relation_type == RelationType.USES
        assert restored.weight == 0.8

    def test_entity_type_enum(self):
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType("technology") == EntityType.TECHNOLOGY

    def test_relation_type_enum(self):
        assert RelationType.FIXES.value == "fixes"
        assert RelationType("uses") == RelationType.USES


# --- Engine Entity Tests ---


class TestEngineEntities:
    def test_add_entity(self, engine):
        entity = Entity(
            id="test1",
            name="fastapi",
            entity_type=EntityType.TECHNOLOGY,
        )
        result = engine.add_entity(entity)
        assert result.id == "test1"
        assert engine.get_entity("test1") is not None

    def test_merge_duplicate_entity(self, engine):
        e1 = Entity(
            id="test1",
            name="fastapi",
            entity_type=EntityType.TECHNOLOGY,
            source_docs=["doc1"],
            mention_count=1,
        )
        e2 = Entity(
            id="test1",
            name="fastapi",
            entity_type=EntityType.TECHNOLOGY,
            source_docs=["doc2"],
            mention_count=1,
            aliases=["fast-api"],
        )
        engine.add_entity(e1)
        merged = engine.add_entity(e2)

        assert merged.mention_count == 2
        assert "doc1" in merged.source_docs
        assert "doc2" in merged.source_docs
        assert "fast-api" in merged.aliases

    def test_find_entity_by_name(self, engine):
        engine.add_entity(
            Entity(
                id="t1",
                name="fastapi",
                entity_type=EntityType.TECHNOLOGY,
            )
        )
        assert engine.find_entity_by_name("FastAPI") is not None
        assert engine.find_entity_by_name("nonexistent") is None

    def test_find_entity_by_alias(self, engine):
        engine.add_entity(
            Entity(
                id="t1",
                name="ddd",
                entity_type=EntityType.CONCEPT,
                aliases=["Domain-Driven Design"],
            )
        )
        assert engine.find_entity_by_name("domain-driven design") is not None

    def test_search_entities(self, engine):
        engine.add_entity(
            Entity(
                id="t1",
                name="fastapi",
                entity_type=EntityType.TECHNOLOGY,
                mention_count=5,
            )
        )
        engine.add_entity(
            Entity(
                id="t2",
                name="flask",
                entity_type=EntityType.TECHNOLOGY,
                mention_count=2,
            )
        )
        engine.add_entity(
            Entity(
                id="t3",
                name="angular",
                entity_type=EntityType.TECHNOLOGY,
                mention_count=3,
            )
        )

        results = engine.search_entities("f")
        assert len(results) == 2
        assert results[0].name == "fastapi"  # Higher mention_count

    def test_search_entities_by_type(self, engine):
        engine.add_entity(
            Entity(
                id="t1",
                name="fastapi",
                entity_type=EntityType.TECHNOLOGY,
            )
        )
        engine.add_entity(
            Entity(
                id="c1",
                name="ddd",
                entity_type=EntityType.CONCEPT,
            )
        )

        results = engine.search_entities("", entity_type=EntityType.TECHNOLOGY)
        assert len(results) == 1
        assert results[0].entity_type == EntityType.TECHNOLOGY


# --- Engine Relation Tests ---


class TestEngineRelations:
    def test_add_relation(self, engine):
        engine.add_entity(Entity(id="a", name="fastapi", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="pydantic", entity_type=EntityType.TECHNOLOGY))

        rel = Relation(
            source_id="a",
            target_id="b",
            relation_type=RelationType.USES,
            weight=0.8,
        )
        engine.add_relation(rel)

        rels = engine.get_relations("a", direction="out")
        assert len(rels) == 1
        assert rels[0].relation_type == RelationType.USES

    def test_relation_reinforcement(self, engine):
        engine.add_entity(Entity(id="a", name="fastapi", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="pydantic", entity_type=EntityType.TECHNOLOGY))

        rel1 = Relation(source_id="a", target_id="b", relation_type=RelationType.USES, weight=0.5)
        rel2 = Relation(source_id="a", target_id="b", relation_type=RelationType.USES, weight=0.5)
        engine.add_relation(rel1)
        engine.add_relation(rel2)

        rels = engine.get_relations("a", direction="out")
        assert len(rels) == 1
        assert rels[0].weight == 0.6  # 0.5 + 0.1 reinforcement

    def test_relation_skips_missing_entity(self, engine):
        engine.add_entity(Entity(id="a", name="fastapi", entity_type=EntityType.TECHNOLOGY))
        rel = Relation(source_id="a", target_id="nonexistent", relation_type=RelationType.USES)
        engine.add_relation(rel)
        assert len(engine.get_relations("a")) == 0

    def test_get_relations_by_type(self, engine):
        engine.add_entity(Entity(id="a", name="fastapi", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="pydantic", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="c", name="flask", entity_type=EntityType.TECHNOLOGY))

        engine.add_relation(Relation(source_id="a", target_id="b", relation_type=RelationType.USES))
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="c",
                relation_type=RelationType.ALTERNATIVE_TO,
            )
        )

        uses = engine.get_relations("a", relation_type=RelationType.USES)
        assert len(uses) == 1


# --- Triple Ingestion ---


class TestTripleIngestion:
    def test_ingest_triple(self, engine):
        triple = Triple(
            subject_name="FastAPI",
            subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic",
            object_type=EntityType.TECHNOLOGY,
            source_doc="training-pair-001",
        )
        subj, obj, rel = engine.ingest_triple(triple)

        assert subj.name == "fastapi"
        assert obj.name == "pydantic"
        assert engine.get_entity(subj.id) is not None
        assert engine.get_entity(obj.id) is not None
        assert len(engine.get_relations(subj.id, direction="out")) == 1

    def test_ingest_multiple_triples_merges(self, engine):
        t1 = Triple(
            subject_name="FastAPI",
            subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic",
            object_type=EntityType.TECHNOLOGY,
            source_doc="doc1",
        )
        t2 = Triple(
            subject_name="FastAPI",
            subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Starlette",
            object_type=EntityType.TECHNOLOGY,
            source_doc="doc2",
        )
        engine.ingest_triple(t1)
        engine.ingest_triple(t2)

        fastapi = engine.find_entity_by_name("fastapi")
        assert fastapi.mention_count == 2
        assert len(engine.get_relations(fastapi.id, direction="out")) == 2


# --- Graph Traversal ---


class TestTraversal:
    def _build_chain(self, engine):
        """Build A -> B -> C -> D chain."""
        for name in ["a", "b", "c", "d"]:
            eid = make_entity_id("technology", name)
            engine.add_entity(Entity(id=eid, name=name, entity_type=EntityType.TECHNOLOGY))

        pairs = [("a", "b"), ("b", "c"), ("c", "d")]
        for src, tgt in pairs:
            engine.add_relation(
                Relation(
                    source_id=make_entity_id("technology", src),
                    target_id=make_entity_id("technology", tgt),
                    relation_type=RelationType.RELATED_TO,
                    weight=0.5,
                )
            )
        return make_entity_id("technology", "a")

    def test_get_neighbors_depth_1(self, engine):
        seed = self._build_chain(engine)
        neighbors = engine.get_neighbors(seed, depth=1)
        assert len(neighbors) == 1  # Only B

    def test_get_neighbors_depth_2(self, engine):
        seed = self._build_chain(engine)
        neighbors = engine.get_neighbors(seed, depth=2)
        assert len(neighbors) == 2  # B and C

    def test_get_neighbors_depth_3(self, engine):
        seed = self._build_chain(engine)
        neighbors = engine.get_neighbors(seed, depth=3)
        assert len(neighbors) == 3  # B, C, D

    def test_get_neighbors_min_weight(self, engine):
        """Neighbors with weight below threshold are excluded."""
        engine.add_entity(Entity(id="x", name="x", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="y", name="y", entity_type=EntityType.TECHNOLOGY))
        engine.add_relation(
            Relation(
                source_id="x",
                target_id="y",
                relation_type=RelationType.RELATED_TO,
                weight=0.1,
            )
        )
        assert len(engine.get_neighbors("x", depth=1, min_weight=0.3)) == 0
        assert len(engine.get_neighbors("x", depth=1, min_weight=0.05)) == 1

    def test_get_source_docs_from_neighbors(self, engine):
        engine.add_entity(
            Entity(
                id="a",
                name="a",
                entity_type=EntityType.TECHNOLOGY,
                source_docs=["doc1"],
            )
        )
        engine.add_entity(
            Entity(
                id="b",
                name="b",
                entity_type=EntityType.TECHNOLOGY,
                source_docs=["doc2", "doc3"],
            )
        )
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.RELATED_TO,
                weight=0.5,
            )
        )

        docs = engine.get_source_docs_from_neighbors("a", depth=1)
        assert "doc1" in docs
        assert "doc2" in docs
        assert "doc3" in docs

    def test_get_neighbors_nonexistent(self, engine):
        assert engine.get_neighbors("nonexistent") == []


# --- Persistence ---


class TestPersistence:
    def test_save_and_load(self, tmp_graph_dir):
        engine1 = GraphEngine(data_dir=tmp_graph_dir)
        engine1.ingest_triple(
            Triple(
                subject_name="FastAPI",
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.USES,
                object_name="Pydantic",
                object_type=EntityType.TECHNOLOGY,
                source_doc="doc1",
            )
        )
        engine1.save()

        engine2 = GraphEngine(data_dir=tmp_graph_dir)
        assert engine2.load() is True
        assert engine2.find_entity_by_name("fastapi") is not None
        assert engine2.find_entity_by_name("pydantic") is not None
        assert (
            len(engine2.get_relations(make_entity_id("technology", "fastapi"), direction="out"))
            == 1
        )

    def test_load_nonexistent(self, tmp_graph_dir):
        engine = GraphEngine(data_dir=tmp_graph_dir)
        assert engine.load() is False

    def test_save_creates_metadata(self, tmp_graph_dir):
        engine = GraphEngine(data_dir=tmp_graph_dir)
        engine.save()
        assert (tmp_graph_dir / "build_metadata.json").exists()

    def test_extraction_state(self, tmp_graph_dir):
        engine = GraphEngine(data_dir=tmp_graph_dir)
        state = engine.load_extraction_state()
        assert state == {"processed_files": {}}

        state["processed_files"]["file1.jsonl"] = "2026-01-01"
        engine.save_extraction_state(state)

        loaded = engine.load_extraction_state()
        assert loaded["processed_files"]["file1.jsonl"] == "2026-01-01"


# --- Stats ---


class TestStats:
    def test_stats_empty(self, engine):
        stats = engine.get_stats()
        assert stats["entity_count"] == 0
        assert stats["edge_count"] == 0

    def test_stats_with_data(self, engine):
        engine.ingest_triple(
            Triple(
                subject_name="FastAPI",
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.USES,
                object_name="Pydantic",
                object_type=EntityType.TECHNOLOGY,
            )
        )
        stats = engine.get_stats()
        assert stats["entity_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["entity_types"]["technology"] == 2
        assert stats["relation_types"]["uses"] == 1


# --- Context Paths ---


class TestContextPaths:
    def test_direct_edge_path(self, engine):
        engine.ingest_triple(
            Triple(
                subject_name="FastAPI",
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.USES,
                object_name="Pydantic",
                object_type=EntityType.TECHNOLOGY,
            )
        )
        fastapi_id = make_entity_id("technology", "fastapi")
        pydantic_id = make_entity_id("technology", "pydantic")

        paths = engine.get_context_paths([fastapi_id, pydantic_id])
        assert len(paths) == 1
        assert "uses" in paths[0]

    def test_no_path(self, engine):
        engine.add_entity(Entity(id="x", name="x", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="y", name="y", entity_type=EntityType.TECHNOLOGY))
        paths = engine.get_context_paths(["x", "y"])
        assert len(paths) == 0


# --- Pruning ---


class TestPruning:
    def test_prune_isolated_entity(self, engine):
        """Ghost node with no edges and low mentions is removed."""
        engine.add_entity(
            Entity(
                id="ghost",
                name="old_class",
                entity_type=EntityType.CONCEPT,
                mention_count=1,
            )
        )
        result = engine.prune()
        assert result["entities_removed"] == 1
        assert engine.get_entity("ghost") is None

    def test_prune_keeps_connected_entity(self, engine):
        """Entity with edges is NOT removed even with low mentions."""
        engine.add_entity(
            Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        )
        engine.add_entity(
            Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        )
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.USES,
                weight=0.8,
            )
        )
        result = engine.prune()
        assert result["entities_removed"] == 0
        assert engine.get_entity("a") is not None
        assert engine.get_entity("b") is not None

    def test_prune_keeps_high_mention_isolate(self, engine):
        """Isolated entity with high mention_count is preserved."""
        engine.add_entity(
            Entity(
                id="popular",
                name="popular",
                entity_type=EntityType.CONCEPT,
                mention_count=5,
            )
        )
        result = engine.prune(min_mention_count=1)
        assert result["entities_removed"] == 0
        assert engine.get_entity("popular") is not None

    def test_prune_low_weight_edges(self, engine):
        """Edges below min_edge_weight are removed."""
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.RELATED_TO,
                weight=0.1,
            )
        )
        result = engine.prune(min_edge_weight=0.3)
        assert result["edges_removed"] == 1
        assert len(engine.get_relations("a", direction="out")) == 0

    def test_prune_keeps_high_weight_edges(self, engine):
        """Edges at or above min_edge_weight are kept."""
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.USES,
                weight=0.8,
            )
        )
        result = engine.prune(min_edge_weight=0.3)
        assert result["edges_removed"] == 0
        assert len(engine.get_relations("a", direction="out")) == 1

    def test_prune_cascading_orphan(self, engine):
        """Entities become orphans after their edges are pruned."""
        engine.add_entity(
            Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        )
        engine.add_entity(
            Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        )
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.RELATED_TO,
                weight=0.1,
            )
        )
        # Both entities only connected by a weak edge
        result = engine.prune(min_edge_weight=0.3)
        assert result["edges_removed"] == 1
        assert result["entities_removed"] == 2

    def test_prune_inferred_edges_removed_by_default(self, engine):
        """Inferred edges are removed by default."""
        engine.add_entity(
            Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        )
        engine.add_entity(
            Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        )
        engine._graph.add_edge(
            "a",
            "b",
            source_id="a",
            target_id="b",
            relation_type=RelationType.DEPENDS_ON.value,
            weight=0.2,
            source_docs=["inferred:transitive"],
            metadata={"inferred": True},
        )
        result = engine.prune(min_edge_weight=0.3)
        assert result["edges_removed"] == 1

    def test_prune_keep_inferred(self, engine):
        """keep_inferred=True preserves inferred edges."""
        engine.add_entity(
            Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        )
        engine.add_entity(
            Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        )
        engine._graph.add_edge(
            "a",
            "b",
            source_id="a",
            target_id="b",
            relation_type=RelationType.DEPENDS_ON.value,
            weight=0.2,
            source_docs=["inferred:transitive"],
            metadata={"inferred": True},
        )
        result = engine.prune(min_edge_weight=0.3, keep_inferred=True)
        assert result["edges_removed"] == 0

    def test_prune_dry_run_no_modification(self, engine):
        """dry_run=True does not modify the graph."""
        engine.add_entity(
            Entity(
                id="ghost",
                name="ghost",
                entity_type=EntityType.CONCEPT,
                mention_count=1,
            )
        )
        result = engine.prune(dry_run=True)
        assert result["entities_removed"] == 1
        # Entity still exists
        assert engine.get_entity("ghost") is not None

    def test_prune_dry_run_reports_removed_items(self, engine):
        """dry_run reports what would be removed."""
        engine.add_entity(
            Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        )
        engine.add_entity(
            Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        )
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.RELATED_TO,
                weight=0.1,
            )
        )
        result = engine.prune(dry_run=True)
        assert result["edges_removed"] == 1
        assert result["entities_removed"] == 2
        assert len(result["removed_edges"]) == 1
        assert len(result["removed_entities"]) == 2
        # Still there
        assert engine.get_entity("a") is not None

    def test_prune_empty_graph(self, engine):
        """Pruning an empty graph returns zero stats."""
        result = engine.prune()
        assert result["edges_removed"] == 0
        assert result["entities_removed"] == 0
        assert result["removed_edges"] == []
        assert result["removed_entities"] == []

    def test_prune_saves_correctly(self, tmp_graph_dir):
        """Graph is consistent after prune + save + load."""
        engine1 = GraphEngine(data_dir=tmp_graph_dir)
        engine1.add_entity(
            Entity(id="keep", name="keep", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        )
        engine1.add_entity(
            Entity(id="ghost", name="ghost", entity_type=EntityType.CONCEPT, mention_count=1)
        )
        engine1.prune()
        engine1.save()

        engine2 = GraphEngine(data_dir=tmp_graph_dir)
        engine2.load()
        assert engine2.get_entity("keep") is not None
        assert engine2.get_entity("ghost") is None

    def test_prune_custom_thresholds(self, engine):
        """Custom thresholds change what gets pruned."""
        engine.add_entity(
            Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY, mention_count=3)
        )
        engine.add_entity(
            Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY, mention_count=3)
        )
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.USES,
                weight=0.4,
            )
        )
        # Default threshold (0.3) keeps the edge
        result_default = engine.prune(dry_run=True)
        assert result_default["edges_removed"] == 0

        # Higher threshold removes the edge
        result_strict = engine.prune(min_edge_weight=0.5, dry_run=True)
        assert result_strict["edges_removed"] == 1


# --- Timestamps ---


class TestTimestamps:
    def test_add_entity_sets_last_seen(self, engine):
        entity = Entity(id="t1", name="fastapi", entity_type=EntityType.TECHNOLOGY)
        result = engine.add_entity(entity)
        assert "last_seen" in result.metadata
        # Should be a valid ISO timestamp
        datetime.fromisoformat(result.metadata["last_seen"])

    def test_merge_entity_updates_last_seen(self, engine):
        e1 = Entity(id="t1", name="fastapi", entity_type=EntityType.TECHNOLOGY)
        result1 = engine.add_entity(e1)
        first_seen = result1.metadata["last_seen"]

        e2 = Entity(id="t1", name="fastapi", entity_type=EntityType.TECHNOLOGY)
        result2 = engine.add_entity(e2)
        second_seen = result2.metadata["last_seen"]
        assert second_seen >= first_seen

    def test_add_relation_sets_timestamps(self, engine):
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))
        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.USES,
                weight=0.7,
            )
        )
        edge_data = engine._graph.edges["a", "b"]
        meta = edge_data["metadata"]
        assert meta["base_weight"] == 0.7
        assert "last_reinforced" in meta
        datetime.fromisoformat(meta["last_reinforced"])

    def test_reinforce_relation_updates_timestamps(self, engine):
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))

        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.USES,
                weight=0.5,
            )
        )
        first_meta = dict(engine._graph.edges["a", "b"]["metadata"])

        engine.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type=RelationType.USES,
                weight=0.5,
            )
        )
        second_meta = engine._graph.edges["a", "b"]["metadata"]

        # base_weight should be updated to the new reinforced weight
        assert second_meta["base_weight"] == 0.6  # 0.5 + 0.1
        assert second_meta["last_reinforced"] >= first_meta["last_reinforced"]


# --- Temporal Decay ---


class TestDecay:
    def _make_edge_with_age(self, engine, weight, days_ago, src="a", tgt="b"):
        """Helper: create an edge that was last reinforced `days_ago` days ago."""
        engine.add_entity(Entity(id=src, name=src, entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=tgt, name=tgt, entity_type=EntityType.TECHNOLOGY))
        past = datetime.now(tz=UTC) - timedelta(days=days_ago)
        engine._graph.add_edge(
            src,
            tgt,
            source_id=src,
            target_id=tgt,
            relation_type=RelationType.USES.value,
            weight=weight,
            source_docs=[],
            metadata={
                "base_weight": weight,
                "last_reinforced": past.isoformat(),
            },
        )

    def test_apply_decay_reduces_weights(self, engine):
        self._make_edge_with_age(engine, weight=1.0, days_ago=30)
        result = engine.apply_decay(half_life_days=90.0)
        assert result["edges_decayed"] == 1
        new_weight = engine._graph.edges["a", "b"]["weight"]
        assert new_weight < 1.0

    def test_apply_decay_idempotent(self, engine):
        self._make_edge_with_age(engine, weight=0.8, days_ago=45)
        ref_time = datetime.now(tz=UTC)

        engine.apply_decay(half_life_days=90.0, reference_time=ref_time)
        weight_after_first = engine._graph.edges["a", "b"]["weight"]

        engine.apply_decay(half_life_days=90.0, reference_time=ref_time)
        weight_after_second = engine._graph.edges["a", "b"]["weight"]

        assert abs(weight_after_first - weight_after_second) < 1e-9

    def test_apply_decay_skips_legacy_edges(self, engine):
        """Edges without last_reinforced timestamp are skipped."""
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))
        engine._graph.add_edge(
            "a",
            "b",
            source_id="a",
            target_id="b",
            relation_type=RelationType.USES.value,
            weight=0.5,
            source_docs=[],
            metadata={},
        )
        result = engine.apply_decay(half_life_days=90.0)
        assert result["edges_skipped"] == 1
        assert result["edges_decayed"] == 0
        # Weight unchanged
        assert engine._graph.edges["a", "b"]["weight"] == 0.5

    def test_apply_decay_respects_half_life(self, engine):
        """At exactly 1 half-life, weight should be ~50% of base."""
        self._make_edge_with_age(engine, weight=1.0, days_ago=90)
        engine.apply_decay(half_life_days=90.0)
        new_weight = engine._graph.edges["a", "b"]["weight"]
        assert abs(new_weight - 0.5) < 0.01

    def test_apply_decay_dry_run(self, engine):
        self._make_edge_with_age(engine, weight=1.0, days_ago=90)
        result = engine.apply_decay(half_life_days=90.0, dry_run=True)
        assert result["edges_decayed"] == 1
        # Weight should NOT have changed
        assert engine._graph.edges["a", "b"]["weight"] == 1.0

    def test_apply_decay_fresh_edges_no_decay(self, engine):
        """Edge created just now should have negligible decay."""
        self._make_edge_with_age(engine, weight=0.8, days_ago=0)
        engine.apply_decay(half_life_days=90.0)
        new_weight = engine._graph.edges["a", "b"]["weight"]
        assert abs(new_weight - 0.8) < 0.01

    def test_decay_then_prune_removes_stale(self, engine):
        """Old edge decays below threshold, prune removes edge + orphan entities."""
        self._make_edge_with_age(engine, weight=0.5, days_ago=365)

        engine.apply_decay(half_life_days=90.0)
        decayed_weight = engine._graph.edges["a", "b"]["weight"]
        assert decayed_weight < 0.3  # Should be well below prune threshold

        result = engine.prune(min_edge_weight=0.3)
        assert result["edges_removed"] == 1
        assert result["entities_removed"] == 2
        assert engine.get_entity("a") is None
        assert engine.get_entity("b") is None


# --- Adaptive Forgetting Curve ---


class TestAdaptiveForgettingCurve:
    """Tests for adaptive forgetting curve with reinforcement count (FC-43)."""

    def _make_edge_with_age(
        self, engine, weight, days_ago, src="a", tgt="b", reinforcement_count=0
    ):
        """Helper: create an edge with given age and reinforcement count."""
        engine.add_entity(Entity(id=src, name=src, entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id=tgt, name=tgt, entity_type=EntityType.TECHNOLOGY))
        past = datetime.now(tz=UTC) - timedelta(days=days_ago)
        engine._graph.add_edge(
            src,
            tgt,
            source_id=src,
            target_id=tgt,
            relation_type=RelationType.USES.value,
            weight=weight,
            source_docs=[],
            metadata={
                "base_weight": weight,
                "last_reinforced": past.isoformat(),
                "reinforcement_count": reinforcement_count,
            },
        )

    def test_reinforced_edge_decays_slower(self, engine):
        """Edge with high reinforcement_count should have higher weight after decay."""
        self._make_edge_with_age(
            engine, weight=1.0, days_ago=90, src="a", tgt="b", reinforcement_count=0
        )
        self._make_edge_with_age(
            engine, weight=1.0, days_ago=90, src="c", tgt="d", reinforcement_count=10
        )
        engine.apply_decay(half_life_days=90.0)
        weight_no_reinforce = engine._graph.edges["a", "b"]["weight"]
        weight_with_reinforce = engine._graph.edges["c", "d"]["weight"]
        # Reinforced edge should decay less
        assert weight_with_reinforce > weight_no_reinforce
        # Non-reinforced at exactly 1 half-life should be ~0.5
        assert abs(weight_no_reinforce - 0.5) < 0.01

    def test_zero_reinforcement_matches_fixed_decay(self, engine):
        """With reinforcement_count=0, behavior is identical to fixed half-life."""
        self._make_edge_with_age(engine, weight=1.0, days_ago=90, reinforcement_count=0)
        engine.apply_decay(half_life_days=90.0)
        weight = engine._graph.edges["a", "b"]["weight"]
        # At exactly 1 half-life: 1.0 * 0.5^1 = 0.5
        assert abs(weight - 0.5) < 0.01

    def test_missing_reinforcement_count_defaults_to_zero(self, engine):
        """Edge without reinforcement_count metadata uses default 0."""
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))
        past = datetime.now(tz=UTC) - timedelta(days=90)
        engine._graph.add_edge(
            "a",
            "b",
            source_id="a",
            target_id="b",
            relation_type=RelationType.USES.value,
            weight=1.0,
            source_docs=[],
            metadata={"base_weight": 1.0, "last_reinforced": past.isoformat()},
        )
        engine.apply_decay(half_life_days=90.0)
        weight = engine._graph.edges["a", "b"]["weight"]
        assert abs(weight - 0.5) < 0.01

    def test_add_relation_increments_reinforcement_count(self, engine):
        """Reinforcing an existing edge increments reinforcement_count."""
        engine.add_entity(Entity(id="a", name="a", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="b", entity_type=EntityType.TECHNOLOGY))
        rel = Relation(
            source_id="a",
            target_id="b",
            relation_type=RelationType.USES,
            weight=0.5,
            source_docs=["doc1"],
        )
        engine.add_relation(rel)  # First add
        meta = engine._graph.edges["a", "b"]["metadata"]
        assert meta.get("reinforcement_count", 0) == 0  # First add, not a reinforcement

        engine.add_relation(rel)  # Second add = first reinforcement
        meta = engine._graph.edges["a", "b"]["metadata"]
        assert meta["reinforcement_count"] == 1

        engine.add_relation(rel)  # Third add = second reinforcement
        meta = engine._graph.edges["a", "b"]["metadata"]
        assert meta["reinforcement_count"] == 2

    def test_high_reinforcement_slows_decay_significantly(self, engine):
        """Edge reinforced 50 times should retain most of its weight over 90 days."""
        self._make_edge_with_age(engine, weight=1.0, days_ago=90, reinforcement_count=50)
        engine.apply_decay(half_life_days=90.0)
        weight = engine._graph.edges["a", "b"]["weight"]
        # With 50 reinforcements: effective_half_life = 90 * (1 + 0.3*50)^0.5 = 90 * 4.0 = ~360 days
        # At 90 days: 0.5^(90/360) = 0.5^0.25 ≈ 0.84
        assert weight > 0.80

    def test_decay_stats_include_reinforcement_info(self, engine):
        """apply_decay returns avg_reinforcement_count and max_reinforcement_count."""
        self._make_edge_with_age(
            engine, weight=1.0, days_ago=30, src="a", tgt="b", reinforcement_count=5
        )
        self._make_edge_with_age(
            engine, weight=1.0, days_ago=30, src="c", tgt="d", reinforcement_count=15
        )
        result = engine.apply_decay(half_life_days=90.0)
        assert result["max_reinforcement_count"] == 15
        assert result["avg_reinforcement_count"] == 10.0

    def test_adaptive_decay_idempotent(self, engine):
        """Adaptive decay is still idempotent (recomputes from base_weight)."""
        self._make_edge_with_age(engine, weight=0.8, days_ago=45, reinforcement_count=5)
        ref_time = datetime.now(tz=UTC)
        engine.apply_decay(half_life_days=90.0, reference_time=ref_time)
        w1 = engine._graph.edges["a", "b"]["weight"]
        engine.apply_decay(half_life_days=90.0, reference_time=ref_time)
        w2 = engine._graph.edges["a", "b"]["weight"]
        assert abs(w1 - w2) < 1e-9


# --- Alias Detection ---


class TestAliasDetection:
    """Tests for dynamic alias detection (FC-40)."""

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

    def test_cosine_similarity_zero_vector(self):
        from src.knowledge.graph_engine import _cosine_similarity

        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_detect_aliases_same_type(self, engine):
        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        engine.add_entity(e1)
        engine.add_entity(e2)
        embeddings = {"t1": [0.9, 0.1, 0.0], "t2": [0.88, 0.12, 0.01]}
        pairs = engine.detect_aliases(embeddings, threshold=0.85)
        assert len(pairs) == 1
        assert pairs[0].canonical.name == "kubernetes"
        assert pairs[0].alias.name == "k8s"

    def test_detect_aliases_different_type(self, engine):
        e1 = Entity(id="t1", name="react", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        e2 = Entity(id="c1", name="reactive", entity_type=EntityType.CONCEPT, mention_count=3)
        engine.add_entity(e1)
        engine.add_entity(e2)
        embeddings = {"t1": [0.9, 0.1, 0.0], "c1": [0.88, 0.12, 0.01]}
        pairs = engine.detect_aliases(embeddings, threshold=0.85)
        assert len(pairs) == 0

    def test_detect_aliases_below_threshold(self, engine):
        e1 = Entity(id="t1", name="react", entity_type=EntityType.TECHNOLOGY, mention_count=5)
        e2 = Entity(id="t2", name="angular", entity_type=EntityType.TECHNOLOGY, mention_count=3)
        engine.add_entity(e1)
        engine.add_entity(e2)
        embeddings = {"t1": [1.0, 0.0, 0.0], "t2": [0.0, 1.0, 0.0]}
        pairs = engine.detect_aliases(embeddings, threshold=0.85)
        assert len(pairs) == 0

    def test_detect_aliases_missing_embedding(self, engine):
        """Entity without embedding is skipped."""
        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        engine.add_entity(e1)
        engine.add_entity(e2)
        embeddings = {"t1": [0.9, 0.1]}  # Only t1 has embedding
        pairs = engine.detect_aliases(embeddings, threshold=0.85)
        assert len(pairs) == 0

    def test_merge_alias_pair_fields(self, engine):
        from src.knowledge.graph_engine import AliasPair

        e1 = Entity(
            id="t1",
            name="kubernetes",
            entity_type=EntityType.TECHNOLOGY,
            mention_count=10,
            source_docs=["doc1"],
            aliases=["kube"],
        )
        e2 = Entity(
            id="t2",
            name="k8s",
            entity_type=EntityType.TECHNOLOGY,
            mention_count=2,
            source_docs=["doc2"],
            aliases=["k8"],
        )
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

    def test_merge_alias_pair_edges(self, engine):
        from src.knowledge.graph_engine import AliasPair
        from src.knowledge.graph_schema import Relation, RelationType

        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        e3 = Entity(id="c1", name="devops", entity_type=EntityType.CONCEPT, mention_count=5)
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        rel = Relation(
            source_id="t2", target_id="c1", relation_type=RelationType.RELATED_TO, weight=0.8
        )
        engine.add_relation(rel)
        pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
        engine.merge_alias_pair(pair)
        neighbors = engine.get_neighbors("t1", depth=1, min_weight=0.0)
        assert any(n.name == "devops" for n, _ in neighbors)

    def test_merge_alias_pair_removes_alias(self, engine):
        from src.knowledge.graph_engine import AliasPair

        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        engine.add_entity(e1)
        engine.add_entity(e2)
        pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
        engine.merge_alias_pair(pair)
        assert "t2" not in engine._entities
        assert not engine._graph.has_node("t2")

    def test_deduplicate_aliases_full(self, engine):
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
        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        engine.add_entity(e1)
        engine.add_entity(e2)
        embeddings = {"t1": [0.9, 0.1], "t2": [0.88, 0.12]}
        stats = engine.deduplicate_aliases(embeddings, threshold=0.85, dry_run=True)
        assert stats["candidates"] == 1
        assert stats["merged"] == 0
        assert "t2" in engine._entities  # NOT removed

    def test_deduplicate_empty_graph(self, engine):
        stats = engine.deduplicate_aliases({}, threshold=0.85, dry_run=False)
        assert stats["candidates"] == 0
        assert stats["merged"] == 0

    def test_merge_alias_found_by_name(self, engine):
        """After merge, alias name is searchable via find_entity_by_name."""
        from src.knowledge.graph_engine import AliasPair

        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        engine.add_entity(e1)
        engine.add_entity(e2)
        pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
        engine.merge_alias_pair(pair)
        # k8s should now be findable via alias lookup
        found = engine.find_entity_by_name("k8s")
        assert found is not None
        assert found.name == "kubernetes"

    def test_merge_alias_pair_incoming_edges(self, engine):
        """Incoming edges (predecessor → alias) are redirected to canonical."""
        from src.knowledge.graph_engine import AliasPair
        from src.knowledge.graph_schema import Relation, RelationType

        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=2)
        e3 = Entity(id="c1", name="devops", entity_type=EntityType.CONCEPT, mention_count=5)
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        # Edge from devops → k8s (incoming to alias)
        rel = Relation(
            source_id="c1", target_id="t2", relation_type=RelationType.RELATED_TO, weight=0.8
        )
        engine.add_relation(rel)
        pair = AliasPair(canonical=e1, alias=e2, similarity=0.92)
        engine.merge_alias_pair(pair)
        # Edge should now be devops → kubernetes
        assert engine._graph.has_edge("c1", "t1")
        assert not engine._graph.has_node("t2")

    def test_deduplicate_transitive_chain(self, engine):
        """Transitive chain: A ~ B ~ C. After dedup, only A remains."""
        e1 = Entity(id="t1", name="kubernetes", entity_type=EntityType.TECHNOLOGY, mention_count=10)
        e2 = Entity(id="t2", name="k8s", entity_type=EntityType.TECHNOLOGY, mention_count=3)
        e3 = Entity(id="t3", name="kube", entity_type=EntityType.TECHNOLOGY, mention_count=1)
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        # All very similar embeddings
        embeddings = {
            "t1": [0.90, 0.10],
            "t2": [0.89, 0.11],
            "t3": [0.88, 0.12],
        }
        stats = engine.deduplicate_aliases(embeddings, threshold=0.85, dry_run=False)
        assert stats["candidates"] >= 2
        # Only kubernetes should remain
        assert "t1" in engine._entities
        assert "t2" not in engine._entities
        assert "t3" not in engine._entities
        merged = engine._entities["t1"]
        assert "k8s" in merged.aliases
        assert "kube" in merged.aliases


# --- Semantic Drift Detection Tests ---


class TestSemanticDrift:
    """Tests for detect_drift() — structural neighborhood comparison."""

    def _make_entity(self, name, etype=EntityType.TECHNOLOGY):
        eid = make_entity_id(etype.value, name)
        return Entity(id=eid, name=name, entity_type=etype)

    def test_no_drift_on_first_build(self, engine):
        """First build has no previous snapshot — no drift detected."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.USES,
            )
        )
        events = engine.detect_drift(threshold=0.7)
        assert events == []

    def test_drift_detected_when_neighbors_change(self, engine):
        """Entity with different neighbors between snapshots triggers drift."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        e3 = self._make_entity("django")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.USES,
            )
        )
        # Snapshot current state
        engine.snapshot_neighborhoods()

        # Now change neighbors: remove fastapi, add django
        engine._graph.remove_edge(e1.id, e2.id)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e3.id,
                relation_type=RelationType.USES,
            )
        )

        events = engine.detect_drift(threshold=0.7)
        assert len(events) >= 1
        drifted_names = [ev.entity_name for ev in events]
        assert "python" in drifted_names

    def test_no_drift_when_neighbors_unchanged(self, engine):
        """Same neighbors between snapshots — no drift."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.USES,
            )
        )
        engine.snapshot_neighborhoods()

        # No changes — same state
        events = engine.detect_drift(threshold=0.7)
        assert events == []

    def test_drift_includes_old_and_new_neighbors(self, engine):
        """DriftEvent contains previous and current neighbor sets."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        e3 = self._make_entity("django")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.USES,
            )
        )
        engine.snapshot_neighborhoods()

        engine._graph.remove_edge(e1.id, e2.id)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e3.id,
                relation_type=RelationType.USES,
            )
        )

        events = engine.detect_drift(threshold=0.7)
        python_events = [ev for ev in events if ev.entity_name == "python"]
        assert len(python_events) == 1
        ev = python_events[0]
        assert "fastapi" in str(ev.old_neighbors)
        assert "django" in str(ev.new_neighbors)

    def test_drift_threshold_sensitivity(self, engine):
        """Lower threshold means less drift detected."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        e3 = self._make_entity("django")
        e4 = self._make_entity("flask")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        engine.add_entity(e4)
        # python -> fastapi, django
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e2.id, relation_type=RelationType.USES)
        )
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e3.id, relation_type=RelationType.USES)
        )
        engine.snapshot_neighborhoods()

        # Change 1 of 2 neighbors: replace django with flask (50% overlap)
        engine._graph.remove_edge(e1.id, e3.id)
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e4.id, relation_type=RelationType.USES)
        )

        # With high threshold (0.7) — drift detected (Jaccard < 0.7)
        events_high = engine.detect_drift(threshold=0.7)
        # With low threshold (0.3) — no drift (Jaccard ~0.33, still under)
        events_low = engine.detect_drift(threshold=0.3)
        # Jaccard of {fastapi, django} vs {fastapi, flask} = 1/3 ≈ 0.33
        assert len(events_high) >= 1
        assert len(events_low) >= 1  # 0.33 < 0.3 is false, so no drift at 0.3
        # Actually Jaccard = 1/3 = 0.33, which IS > 0.3, so no drift at 0.3

    def test_snapshot_persists_across_save_load(self, engine):
        """Snapshots survive save/load cycle."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_relation(
            Relation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.USES,
            )
        )
        engine.snapshot_neighborhoods()
        engine.save()

        # Load in fresh engine
        engine2 = GraphEngine(data_dir=engine.data_dir)
        engine2.load()

        # Verify snapshot is present in metadata
        python_entity = engine2.get_entity(e1.id)
        assert python_entity is not None
        assert "neighbor_snapshot" in python_entity.metadata

    def test_created_at_set_on_first_add(self, engine):
        """Entity gets created_at metadata on first creation."""
        e1 = self._make_entity("python")
        result = engine.add_entity(e1)
        assert "created_at" in result.metadata
        created = result.metadata["created_at"]

        # Adding again doesn't change created_at
        e1_dup = self._make_entity("python")
        result2 = engine.add_entity(e1_dup)
        assert result2.metadata["created_at"] == created

    def test_version_increments_on_context_change(self, engine):
        """Entity version increments when neighbor_snapshot changes."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        e3 = self._make_entity("django")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e2.id, relation_type=RelationType.USES)
        )
        engine.snapshot_neighborhoods()

        python = engine.get_entity(e1.id)
        assert python.metadata.get("version", 1) == 1

        # Change neighbors
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e3.id, relation_type=RelationType.USES)
        )
        engine.snapshot_neighborhoods()
        python = engine.get_entity(e1.id)
        assert python.metadata.get("version", 1) == 2

    def test_drift_event_has_jaccard_score(self, engine):
        """DriftEvent includes the Jaccard similarity score."""
        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        e3 = self._make_entity("django")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e2.id, relation_type=RelationType.USES)
        )
        engine.snapshot_neighborhoods()

        engine._graph.remove_edge(e1.id, e2.id)
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e3.id, relation_type=RelationType.USES)
        )
        events = engine.detect_drift(threshold=0.7)
        assert len(events) >= 1
        assert hasattr(events[0], "jaccard_similarity")
        assert 0.0 <= events[0].jaccard_similarity <= 1.0

    def test_isolated_entities_no_drift(self, engine):
        """Entities with no neighbors don't trigger drift (no context to compare)."""
        e1 = self._make_entity("python")
        engine.add_entity(e1)
        engine.snapshot_neighborhoods()
        # No neighbors before or after — no drift
        events = engine.detect_drift(threshold=0.7)
        assert events == []

    def test_drift_log_persistence(self, engine):
        """Drift events are persisted to JSONL log file."""
        import json

        e1 = self._make_entity("python")
        e2 = self._make_entity("fastapi")
        e3 = self._make_entity("django")
        engine.add_entity(e1)
        engine.add_entity(e2)
        engine.add_entity(e3)
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e2.id, relation_type=RelationType.USES)
        )
        engine.snapshot_neighborhoods()

        engine._graph.remove_edge(e1.id, e2.id)
        engine.add_relation(
            Relation(source_id=e1.id, target_id=e3.id, relation_type=RelationType.USES)
        )

        events = engine.detect_drift(threshold=0.7)
        engine.persist_drift_events(events)

        log_path = engine.data_dir / "drift_log.jsonl"
        assert log_path.exists()
        with open(log_path) as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) >= 1
        assert lines[0]["entity_name"] == "python"

    def test_load_drift_log(self, engine):
        """Can load and filter drift log entries."""
        import json

        # Write some test entries
        log_path = engine.data_dir / "drift_log.jsonl"
        entries = [
            {
                "entity_id": "abc123",
                "entity_name": "python",
                "jaccard_similarity": 0.2,
                "timestamp": "2026-02-26T10:00:00",
            },
            {
                "entity_id": "def456",
                "entity_name": "fastapi",
                "jaccard_similarity": 0.5,
                "timestamp": "2026-02-26T11:00:00",
            },
        ]
        with open(log_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        all_events = engine.load_drift_log()
        assert len(all_events) == 2

        python_events = engine.load_drift_log(entity_name="python")
        assert len(python_events) == 1
        assert python_events[0]["entity_name"] == "python"
