"""Knowledge Graph Engine - NetworkX-based graph with persistence."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass
class DriftEvent:
    """A detected semantic drift event for an entity."""

    entity_id: str
    entity_name: str
    entity_type: str
    jaccard_similarity: float
    old_neighbors: list[str]
    new_neighbors: list[str]
    added: list[str]
    removed: list[str]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "jaccard_similarity": round(self.jaccard_similarity, 4),
            "old_neighbors": self.old_neighbors,
            "new_neighbors": self.new_neighbors,
            "added": self.added,
            "removed": self.removed,
            "timestamp": self.timestamp,
        }


@dataclass
class AliasPair:
    """A pair of entities detected as likely aliases."""

    canonical: Entity  # Keep this one (higher mention_count)
    alias: Entity  # Merge into canonical
    similarity: float  # Cosine similarity score


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
        now_iso = datetime.now(tz=UTC).isoformat()
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
            existing.metadata["last_seen"] = now_iso
            self._graph.nodes[existing.id].update(existing.to_dict())
            return existing

        entity.metadata["last_seen"] = now_iso
        entity.metadata.setdefault("created_at", now_iso)
        entity.metadata.setdefault("version", 1)
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

        now_iso = datetime.now(tz=UTC).isoformat()
        edge_key = (relation.source_id, relation.target_id)
        if self._graph.has_edge(*edge_key):
            edge_data = self._graph.edges[edge_key]
            # Reinforce weight (cap at 1.0)
            current_weight = edge_data.get("weight", 0.5)
            new_weight = min(1.0, current_weight + 0.1)
            edge_data["weight"] = new_weight
            # Update decay timestamps
            edge_meta = edge_data.get("metadata", {})
            edge_meta["base_weight"] = new_weight
            edge_meta["last_reinforced"] = now_iso
            edge_meta["reinforcement_count"] = edge_meta.get("reinforcement_count", 0) + 1
            edge_data["metadata"] = edge_meta
            # Merge source_docs
            existing_docs = edge_data.get("source_docs", [])
            for doc in relation.source_docs:
                if doc not in existing_docs:
                    existing_docs.append(doc)
            edge_data["source_docs"] = existing_docs
        else:
            relation.metadata["base_weight"] = relation.weight
            relation.metadata["last_reinforced"] = now_iso
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
        subject = self.add_entity(
            Entity(
                id=subj_id,
                name=triple.subject_name.strip().lower(),
                entity_type=triple.subject_type,
                source_docs=[triple.source_doc] if triple.source_doc else [],
            )
        )

        # Create/get object entity
        obj_id = make_entity_id(triple.object_type.value, triple.object_name)
        obj = self.add_entity(
            Entity(
                id=obj_id,
                name=triple.object_name.strip().lower(),
                entity_type=triple.object_type,
                source_docs=[triple.source_doc] if triple.source_doc else [],
            )
        )

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
            for tgt_id in entity_ids[i + 1 :]:
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
                            names = [self._entities[n].name for n in path if n in self._entities]
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
            {
                "id": eid,
                "name": self._entities[eid].name,
                "type": self._entities[eid].entity_type.value,
            }
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

    # --- Temporal Decay ---

    REINFORCEMENT_FACTOR: float = 0.3
    REINFORCEMENT_EXPONENT: float = 0.5

    def apply_decay(
        self,
        half_life_days: float = 90.0,
        reference_time: datetime | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Apply exponential decay to edge weights based on time since last reinforcement.

        Computes weight = base_weight * 0.5^(days_elapsed / half_life_days).
        Idempotent: always recomputes from base_weight, never compounds.

        Args:
            half_life_days: Days for weight to halve.
            reference_time: Reference point for age calculation (default: now UTC).
            dry_run: If True, compute stats without modifying weights.

        Returns:
            Stats dict with edges_decayed, edges_skipped, min/max weight.
        """
        ref = reference_time or datetime.now(tz=UTC)
        edges_decayed = 0
        edges_skipped = 0
        min_weight = float("inf")
        max_weight = 0.0
        total_reinforcement_count = 0
        max_reinforcement_count = 0

        for _src, _tgt, data in self._graph.edges(data=True):
            meta = data.get("metadata", {})
            last_reinforced = meta.get("last_reinforced")
            if not last_reinforced:
                edges_skipped += 1
                continue

            lr_dt = datetime.fromisoformat(last_reinforced)
            days_elapsed = (ref - lr_dt).total_seconds() / 86400.0
            if days_elapsed < 0:
                days_elapsed = 0.0

            reinforcement_count = meta.get("reinforcement_count", 0)
            effective_half_life = (
                half_life_days
                * (1 + self.REINFORCEMENT_FACTOR * reinforcement_count)
                ** self.REINFORCEMENT_EXPONENT
            )
            decay_factor = 0.5 ** (days_elapsed / effective_half_life)
            base_weight = meta.get("base_weight", data.get("weight", 0.5))
            new_weight = base_weight * decay_factor

            if not dry_run:
                data["weight"] = new_weight

            min_weight = min(min_weight, new_weight)
            max_weight = max(max_weight, new_weight)
            total_reinforcement_count += reinforcement_count
            max_reinforcement_count = max(max_reinforcement_count, reinforcement_count)
            edges_decayed += 1

        if edges_decayed == 0:
            min_weight = 0.0

        return {
            "edges_decayed": edges_decayed,
            "edges_skipped": edges_skipped,
            "min_weight_after": round(min_weight, 6),
            "max_weight_after": round(max_weight, 6),
            "avg_reinforcement_count": (
                round(total_reinforcement_count / edges_decayed, 2) if edges_decayed else 0.0
            ),
            "max_reinforcement_count": max_reinforcement_count,
        }

    # --- Alias Detection ---

    def detect_aliases(
        self,
        embeddings: dict[str, list[float]],
        threshold: float = 0.85,
    ) -> list[AliasPair]:
        """Detect entity pairs that are likely aliases based on embedding similarity.

        Only compares entities of the same type. The entity with higher
        mention_count is treated as canonical.
        """
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
                        pairs.append(
                            AliasPair(
                                canonical=canonical,
                                alias=alias,
                                similarity=sim,
                            )
                        )

        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs

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
                    edge_data = dict(self._graph.edges[predecessor, alias.id])
                    if not self._graph.has_edge(predecessor, canonical.id):
                        self._graph.add_edge(predecessor, canonical.id, **edge_data)
            for successor in list(self._graph.successors(alias.id)):
                if successor != canonical.id:
                    edge_data = dict(self._graph.edges[alias.id, successor])
                    if not self._graph.has_edge(canonical.id, successor):
                        self._graph.add_edge(canonical.id, successor, **edge_data)
            self._graph.remove_node(alias.id)

        # Remove from entities dict
        del self._entities[alias.id]

        # Update canonical in graph
        self._graph.nodes[canonical.id].update(canonical.to_dict())

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

        if stats["merged"] > 0:
            logger.info(
                "aliases_deduplicated",
                candidates=stats["candidates"],
                merged=stats["merged"],
            )

        return stats

    # --- Semantic Drift Detection ---

    def _get_neighbor_set(self, entity_id: str) -> set[str]:
        """Get the set of neighbor names (both directions) for an entity."""
        neighbors: set[str] = set()
        if entity_id not in self._graph:
            return neighbors
        for succ in self._graph.successors(entity_id):
            ent = self._entities.get(succ)
            if ent:
                neighbors.add(ent.name)
        for pred in self._graph.predecessors(entity_id):
            ent = self._entities.get(pred)
            if ent:
                neighbors.add(ent.name)
        return neighbors

    def snapshot_neighborhoods(self) -> int:
        """Store current neighbor set in each entity's metadata.

        Compares with previous snapshot and increments version if changed.
        Returns number of entities whose context changed.
        """
        changed = 0
        for entity in self._entities.values():
            current = sorted(self._get_neighbor_set(entity.id))
            previous = entity.metadata.get("neighbor_snapshot", [])
            if current != previous:
                if previous:  # Only count as change if there was a prior snapshot
                    entity.metadata["version"] = entity.metadata.get("version", 1) + 1
                    changed += 1
                entity.metadata["neighbor_snapshot"] = current
            # Update graph node
            self._graph.nodes[entity.id]["metadata"] = entity.metadata
        return changed

    def detect_drift(self, threshold: float = 0.7) -> list[DriftEvent]:
        """Detect entities whose neighborhood changed beyond threshold.

        Compares current neighbor set against stored snapshot using Jaccard
        similarity. Entities without a previous snapshot are skipped.

        Args:
            threshold: Jaccard similarity below this triggers drift event.

        Returns:
            List of DriftEvent for entities with significant context change.
        """
        events: list[DriftEvent] = []
        for entity in self._entities.values():
            old_snapshot = entity.metadata.get("neighbor_snapshot", [])
            if not old_snapshot:
                continue  # No previous snapshot — skip

            current = sorted(self._get_neighbor_set(entity.id))
            old_set = set(old_snapshot)
            new_set = set(current)

            # Skip isolated entities (no neighbors then or now)
            if not old_set and not new_set:
                continue

            # Jaccard similarity
            intersection = old_set & new_set
            union = old_set | new_set
            jaccard = len(intersection) / len(union) if union else 1.0

            if jaccard < threshold:
                events.append(
                    DriftEvent(
                        entity_id=entity.id,
                        entity_name=entity.name,
                        entity_type=entity.entity_type.value,
                        jaccard_similarity=jaccard,
                        old_neighbors=sorted(old_set),
                        new_neighbors=sorted(new_set),
                        added=sorted(new_set - old_set),
                        removed=sorted(old_set - new_set),
                    )
                )

        events.sort(key=lambda e: e.jaccard_similarity)
        return events

    def persist_drift_events(self, events: list[DriftEvent]) -> Path:
        """Append drift events to JSONL log file.

        Returns path to the log file.
        """
        log_path = self.data_dir / "drift_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        if events:
            logger.info("drift_events_persisted", count=len(events), path=str(log_path))
        return log_path

    def load_drift_log(self, entity_name: str | None = None) -> list[dict]:
        """Load drift log entries, optionally filtered by entity name."""
        log_path = self.data_dir / "drift_log.jsonl"
        if not log_path.exists():
            return []

        entries = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entity_name and entry.get("entity_name") != entity_name:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

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
            "connected_components": (
                nx.number_weakly_connected_components(self._graph) if len(self._graph) > 0 else 0
            ),
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
