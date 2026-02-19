"""Knowledge Graph Schema - Entity and Relation definitions."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    PATTERN = "pattern"
    ERROR_TYPE = "error_type"
    STRATEGY = "strategy"
    CATEGORY = "category"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    USES = "uses"
    RELATED_TO = "related_to"
    FIXES = "fixes"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    ALTERNATIVE_TO = "alternative_to"
    LEARNED_FROM = "learned_from"


def make_entity_id(entity_type: str, name: str) -> str:
    """Generate deterministic entity ID from type + normalized name."""
    normalized = name.strip().lower()
    return hashlib.md5(f"{entity_type}:{normalized}".encode()).hexdigest()[:12]


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    mention_count: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "aliases": self.aliases,
            "source_docs": self.source_docs,
            "mention_count": self.mention_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            source_docs=data.get("source_docs", []),
            mention_count=data.get("mention_count", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relation:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 0.5
    source_docs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "source_docs": self.source_docs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data.get("weight", 0.5),
            source_docs=data.get("source_docs", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Triple:
    """An extracted triple (subject, predicate, object) before resolution."""
    subject_name: str
    subject_type: EntityType
    relation_type: RelationType
    object_name: str
    object_type: EntityType
    source_doc: str = ""
    confidence: float = 1.0
