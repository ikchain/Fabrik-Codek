"""Tests for GraphEngine and graph schema."""

import tempfile
from pathlib import Path

import pytest

from src.knowledge.graph_schema import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    Triple,
    make_entity_id,
)
from src.knowledge.graph_engine import GraphEngine


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
        engine.add_entity(Entity(
            id="t1", name="fastapi", entity_type=EntityType.TECHNOLOGY,
        ))
        assert engine.find_entity_by_name("FastAPI") is not None
        assert engine.find_entity_by_name("nonexistent") is None

    def test_find_entity_by_alias(self, engine):
        engine.add_entity(Entity(
            id="t1", name="ddd", entity_type=EntityType.CONCEPT,
            aliases=["Domain-Driven Design"],
        ))
        assert engine.find_entity_by_name("domain-driven design") is not None

    def test_search_entities(self, engine):
        engine.add_entity(Entity(
            id="t1", name="fastapi", entity_type=EntityType.TECHNOLOGY, mention_count=5,
        ))
        engine.add_entity(Entity(
            id="t2", name="flask", entity_type=EntityType.TECHNOLOGY, mention_count=2,
        ))
        engine.add_entity(Entity(
            id="t3", name="angular", entity_type=EntityType.TECHNOLOGY, mention_count=3,
        ))

        results = engine.search_entities("f")
        assert len(results) == 2
        assert results[0].name == "fastapi"  # Higher mention_count

    def test_search_entities_by_type(self, engine):
        engine.add_entity(Entity(
            id="t1", name="fastapi", entity_type=EntityType.TECHNOLOGY,
        ))
        engine.add_entity(Entity(
            id="c1", name="ddd", entity_type=EntityType.CONCEPT,
        ))

        results = engine.search_entities("", entity_type=EntityType.TECHNOLOGY)
        assert len(results) == 1
        assert results[0].entity_type == EntityType.TECHNOLOGY


# --- Engine Relation Tests ---


class TestEngineRelations:
    def test_add_relation(self, engine):
        engine.add_entity(Entity(id="a", name="fastapi", entity_type=EntityType.TECHNOLOGY))
        engine.add_entity(Entity(id="b", name="pydantic", entity_type=EntityType.TECHNOLOGY))

        rel = Relation(
            source_id="a", target_id="b",
            relation_type=RelationType.USES, weight=0.8,
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
        engine.add_relation(Relation(
            source_id="a", target_id="c", relation_type=RelationType.ALTERNATIVE_TO,
        ))

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
            subject_name="FastAPI", subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic", object_type=EntityType.TECHNOLOGY,
            source_doc="doc1",
        )
        t2 = Triple(
            subject_name="FastAPI", subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Starlette", object_type=EntityType.TECHNOLOGY,
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
            engine.add_relation(Relation(
                source_id=make_entity_id("technology", src),
                target_id=make_entity_id("technology", tgt),
                relation_type=RelationType.RELATED_TO,
                weight=0.5,
            ))
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
        engine.add_relation(Relation(
            source_id="x", target_id="y",
            relation_type=RelationType.RELATED_TO, weight=0.1,
        ))
        assert len(engine.get_neighbors("x", depth=1, min_weight=0.3)) == 0
        assert len(engine.get_neighbors("x", depth=1, min_weight=0.05)) == 1

    def test_get_source_docs_from_neighbors(self, engine):
        engine.add_entity(Entity(
            id="a", name="a", entity_type=EntityType.TECHNOLOGY, source_docs=["doc1"],
        ))
        engine.add_entity(Entity(
            id="b", name="b", entity_type=EntityType.TECHNOLOGY, source_docs=["doc2", "doc3"],
        ))
        engine.add_relation(Relation(
            source_id="a", target_id="b",
            relation_type=RelationType.RELATED_TO, weight=0.5,
        ))

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
        engine1.ingest_triple(Triple(
            subject_name="FastAPI", subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic", object_type=EntityType.TECHNOLOGY,
            source_doc="doc1",
        ))
        engine1.save()

        engine2 = GraphEngine(data_dir=tmp_graph_dir)
        assert engine2.load() is True
        assert engine2.find_entity_by_name("fastapi") is not None
        assert engine2.find_entity_by_name("pydantic") is not None
        assert len(engine2.get_relations(
            make_entity_id("technology", "fastapi"), direction="out"
        )) == 1

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
        engine.ingest_triple(Triple(
            subject_name="FastAPI", subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic", object_type=EntityType.TECHNOLOGY,
        ))
        stats = engine.get_stats()
        assert stats["entity_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["entity_types"]["technology"] == 2
        assert stats["relation_types"]["uses"] == 1


# --- Context Paths ---


class TestContextPaths:
    def test_direct_edge_path(self, engine):
        engine.ingest_triple(Triple(
            subject_name="FastAPI", subject_type=EntityType.TECHNOLOGY,
            relation_type=RelationType.USES,
            object_name="Pydantic", object_type=EntityType.TECHNOLOGY,
        ))
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
