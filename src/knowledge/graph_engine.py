"""Knowledge Graph Engine - NetworkX-based graph with persistence."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import structlog

from src.config import settings
from src.knowledge.graph_schema import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    Triple,
    make_entity_id,
)

logger = structlog.get_logger()

TRANSITIVE_RELATIONS = [RelationType.DEPENDS_ON, RelationType.PART_OF]
INFERRED_CONFIDENCE = 0.3


class GraphEngine:
    """Core knowledge graph engine backed by NetworkX."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.data_dir / "graphdb"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "knowledge_graph.json"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "extraction_state.json"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "build_metadata.json"

    # --- Entity Management ---

    def add_entity(self, entity: Entity) -> Entity:
        """Add or merge an entity into the graph."""
        existing = self._entities.get(entity.id)
        if existing:
            # Merge: combine source_docs, aliases, increment count
            existing.mention_count += entity.mention_count
            for doc in entity.source_docs:
                if doc not in existing.source_docs:
                    existing.source_docs.append(doc)
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            if entity.description and not existing.description:
                existing.description = entity.description
            self._graph.nodes[existing.id].update(existing.to_dict())
            return existing

        self._entities[entity.id] = entity
        self._graph.add_node(entity.id, **entity.to_dict())
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Entity | None:
        """Find entity by name or alias (case-insensitive)."""
        name_lower = name.strip().lower()
        for entity in self._entities.values():
            if entity.name.lower() == name_lower:
                return entity
            if name_lower in [a.lower() for a in entity.aliases]:
                return entity
        return None

    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by substring match on name/aliases."""
        query_lower = query.strip().lower()
        results = []
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            # Match on name or aliases
            if query_lower in entity.name.lower():
                results.append(entity)
            elif any(query_lower in a.lower() for a in entity.aliases):
                results.append(entity)

        results.sort(key=lambda e: e.mention_count, reverse=True)
        return results[:limit]

    # --- Relation Management ---

    def add_relation(self, relation: Relation) -> None:
        """Add or reinforce a relation."""
        if relation.source_id not in self._entities or relation.target_id not in self._entities:
            return

        edge_key = (relation.source_id, relation.target_id)
        if self._graph.has_edge(*edge_key):
            edge_data = self._graph.edges[edge_key]
            # Reinforce weight (cap at 1.0)
            current_weight = edge_data.get("weight", 0.5)
            edge_data["weight"] = min(1.0, current_weight + 0.1)
            # Merge source_docs
            existing_docs = edge_data.get("source_docs", [])
            for doc in relation.source_docs:
                if doc not in existing_docs:
                    existing_docs.append(doc)
            edge_data["source_docs"] = existing_docs
        else:
            self._graph.add_edge(
                relation.source_id,
                relation.target_id,
                **relation.to_dict(),
            )

    def get_relations(
        self,
        entity_id: str,
        relation_type: RelationType | None = None,
        direction: str = "both",
    ) -> list[Relation]:
        """Get relations for an entity."""
        relations = []

        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(entity_id, data=True):
                rel = Relation.from_dict(data)
                if relation_type is None or rel.relation_type == relation_type:
                    relations.append(rel)

        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(entity_id, data=True):
                rel = Relation.from_dict(data)
                if relation_type is None or rel.relation_type == relation_type:
                    relations.append(rel)

        return relations

    # --- Triple Ingestion ---

    def ingest_triple(self, triple: Triple) -> tuple[Entity, Entity, Relation]:
        """Ingest a triple, creating/merging entities and adding relation."""
        # Create/get subject entity
        subj_id = make_entity_id(triple.subject_type.value, triple.subject_name)
        subject = self.add_entity(Entity(
            id=subj_id,
            name=triple.subject_name.strip().lower(),
            entity_type=triple.subject_type,
            source_docs=[triple.source_doc] if triple.source_doc else [],
        ))

        # Create/get object entity
        obj_id = make_entity_id(triple.object_type.value, triple.object_name)
        obj = self.add_entity(Entity(
            id=obj_id,
            name=triple.object_name.strip().lower(),
            entity_type=triple.object_type,
            source_docs=[triple.source_doc] if triple.source_doc else [],
        ))

        # Create relation
        relation = Relation(
            source_id=subj_id,
            target_id=obj_id,
            relation_type=triple.relation_type,
            weight=triple.confidence,
            source_docs=[triple.source_doc] if triple.source_doc else [],
        )
        self.add_relation(relation)

        return subject, obj, relation

    # --- Graph Traversal ---

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
        min_weight: float = 0.3,
    ) -> list[tuple[Entity, float]]:
        """BFS traversal from entity, returning neighbors with distance score.

        Returns list of (entity, score) where score decreases with distance.
        """
        if entity_id not in self._graph:
            return []

        visited: dict[str, float] = {}
        queue: list[tuple[str, int]] = [(entity_id, 0)]

        while queue:
            node_id, dist = queue.pop(0)
            if node_id in visited:
                continue
            if dist > depth:
                continue

            score = 1.0 / (1.0 + dist)
            visited[node_id] = score

            # Explore neighbors (both directions)
            for neighbor in self._graph.successors(node_id):
                edge = self._graph.edges[node_id, neighbor]
                if edge.get("weight", 0) >= min_weight and neighbor not in visited:
                    queue.append((neighbor, dist + 1))

            for neighbor in self._graph.predecessors(node_id):
                edge = self._graph.edges[neighbor, node_id]
                if edge.get("weight", 0) >= min_weight and neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        # Remove seed entity from results
        visited.pop(entity_id, None)

        results = []
        for nid, score in visited.items():
            entity = self._entities.get(nid)
            if entity:
                results.append((entity, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_source_docs_from_neighbors(
        self,
        entity_id: str,
        depth: int = 2,
        min_weight: float = 0.3,
    ) -> list[str]:
        """Get all source_docs from entity neighborhood (for RAG retrieval)."""
        neighbors = self.get_neighbors(entity_id, depth=depth, min_weight=min_weight)
        doc_ids: set[str] = set()

        # Include seed entity's source_docs
        seed = self._entities.get(entity_id)
        if seed:
            doc_ids.update(seed.source_docs)

        for entity, _score in neighbors:
            doc_ids.update(entity.source_docs)

        return list(doc_ids)

    def get_context_paths(
        self,
        entity_ids: list[str],
        max_paths: int = 5,
    ) -> list[str]:
        """Get readable relationship paths between entities for context injection."""
        paths = []
        for i, src_id in enumerate(entity_ids):
            for tgt_id in entity_ids[i + 1:]:
                src = self._entities.get(src_id)
                tgt = self._entities.get(tgt_id)
                if not src or not tgt:
                    continue

                # Check direct edge
                if self._graph.has_edge(src_id, tgt_id):
                    edge = self._graph.edges[src_id, tgt_id]
                    rel_type = edge.get("relation_type", "related_to")
                    paths.append(f"{src.name} --[{rel_type}]--> {tgt.name}")
                elif self._graph.has_edge(tgt_id, src_id):
                    edge = self._graph.edges[tgt_id, src_id]
                    rel_type = edge.get("relation_type", "related_to")
                    paths.append(f"{tgt.name} --[{rel_type}]--> {src.name}")
                else:
                    # Try shortest path (undirected)
                    try:
                        undirected = self._graph.to_undirected()
                        path = nx.shortest_path(undirected, src_id, tgt_id)
                        if len(path) <= 4:
                            names = [
                                self._entities[n].name for n in path if n in self._entities
                            ]
                            paths.append(" -> ".join(names))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

                if len(paths) >= max_paths:
                    break
            if len(paths) >= max_paths:
                break

        return paths

    # --- Graph Completion ---

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

    # --- Pruning ---

    def prune(
        self,
        min_mention_count: int = 1,
        min_edge_weight: float = 0.3,
        keep_inferred: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Remove ghost nodes and low-quality edges from the graph.

        Three-step process:
        1. Find edges with weight < min_edge_weight
        2. Find isolated entities (0 remaining edges) with mention_count <= min_mention_count
        3. Remove them (unless dry_run)

        Args:
            min_mention_count: Entities with mention_count <= this AND 0 edges are removed.
            min_edge_weight: Edges with weight below this are removed.
            keep_inferred: If True, preserve inferred (transitive) edges.
            dry_run: If True, compute stats without modifying the graph.

        Returns:
            Stats dict with edges_removed, entities_removed, removed_edges, removed_entities.
        """
        # Step 1: Find edges to remove
        edges_to_remove = []
        for src, tgt, data in self._graph.edges(data=True):
            if data.get("weight", 0.5) < min_edge_weight:
                if keep_inferred and data.get("metadata", {}).get("inferred"):
                    continue
                edges_to_remove.append((src, tgt))

        # Step 2: Find isolated entities after edge removal
        # Build a set of edges that will survive
        surviving_edges: set[tuple[str, str]] = set()
        for src, tgt in self._graph.edges():
            if (src, tgt) not in set(edges_to_remove):
                surviving_edges.add((src, tgt))

        # Compute degree for each entity excluding removed edges
        degree: dict[str, int] = defaultdict(int)
        for src, tgt in surviving_edges:
            degree[src] += 1
            degree[tgt] += 1

        entities_to_remove = []
        for eid, entity in self._entities.items():
            if degree[eid] == 0 and entity.mention_count <= min_mention_count:
                entities_to_remove.append(eid)

        # Build result
        removed_edges_info = [
            {"source": src, "target": tgt, "weight": self._graph.edges[src, tgt].get("weight", 0)}
            for src, tgt in edges_to_remove
        ]
        removed_entities_info = [
            {"id": eid, "name": self._entities[eid].name, "type": self._entities[eid].entity_type.value}
            for eid in entities_to_remove
        ]

        stats = {
            "edges_removed": len(edges_to_remove),
            "entities_removed": len(entities_to_remove),
            "removed_edges": removed_edges_info,
            "removed_entities": removed_entities_info,
        }

        # Step 3: Apply changes
        if not dry_run:
            for src, tgt in edges_to_remove:
                self._graph.remove_edge(src, tgt)
            for eid in entities_to_remove:
                self._graph.remove_node(eid)
                del self._entities[eid]

            if edges_to_remove or entities_to_remove:
                logger.info(
                    "graph_pruned",
                    edges_removed=len(edges_to_remove),
                    entities_removed=len(entities_to_remove),
                )

        return stats

    # --- Persistence ---

    def save(self) -> None:
        """Save graph to JSON."""
        data = nx.node_link_data(self._graph)
        self.graph_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        # Save build metadata
        metadata = {
            "last_build": datetime.now().isoformat(),
            "entity_count": len(self._entities),
            "edge_count": self._graph.number_of_edges(),
            "entity_types": dict(self._count_entity_types()),
            "relation_types": dict(self._count_relation_types()),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        logger.info(
            "graph_saved",
            entities=len(self._entities),
            edges=self._graph.number_of_edges(),
        )

    def load(self) -> bool:
        """Load graph from JSON. Returns True if loaded, False if no file."""
        if not self.graph_path.exists():
            return False

        try:
            data = json.loads(self.graph_path.read_text())
            self._graph = nx.node_link_graph(data, directed=True)

            # Rebuild entity index
            self._entities.clear()
            for node_id, node_data in self._graph.nodes(data=True):
                if "entity_type" in node_data:
                    # node_link_graph strips 'id' from node data (uses it as key)
                    node_data_with_id = {**node_data, "id": node_id}
                    self._entities[node_id] = Entity.from_dict(node_data_with_id)

            logger.info(
                "graph_loaded",
                entities=len(self._entities),
                edges=self._graph.number_of_edges(),
            )
            return True

        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.error("graph_load_error", error=str(e))
            return False

    # --- Stats ---

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "entity_count": len(self._entities),
            "edge_count": self._graph.number_of_edges(),
            "entity_types": dict(self._count_entity_types()),
            "relation_types": dict(self._count_relation_types()),
            "connected_components": nx.number_weakly_connected_components(self._graph)
            if len(self._graph) > 0
            else 0,
            "graph_path": str(self.graph_path),
            "graph_exists": self.graph_path.exists(),
        }

    def _count_entity_types(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            counts[entity.entity_type.value] += 1
        return dict(counts)

    def _count_relation_types(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for _, _, data in self._graph.edges(data=True):
            rel_type = data.get("relation_type", "unknown")
            counts[rel_type] += 1
        return dict(counts)

    # --- Extraction State ---

    def load_extraction_state(self) -> dict:
        """Load tracking of processed files."""
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"processed_files": {}}

    def save_extraction_state(self, state: dict) -> None:
        """Save extraction state."""
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))
