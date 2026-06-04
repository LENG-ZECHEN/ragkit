"""Graph storage adapters.

The CLI talks to GraphStore; concrete implementations plug behind it.
Today: NetworkXGraphStore with JSON file persistence. Future: Neo4j adapter
would implement the same interface.

To swap backends, edit `default_store()` at the bottom.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import networkx as nx

from ragkit.core.graph.types import Community, Entity, Finding, Relation
from ragkit.logger import logger


class GraphStore(ABC):
    """Abstract graph backend."""

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def upsert_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def get_entity(self, name: str) -> Entity | None: ...

    @abstractmethod
    def neighbors(self, name: str, depth: int = 1) -> list[Entity]: ...

    @abstractmethod
    def entity_count(self) -> int: ...

    @abstractmethod
    def relation_count(self) -> int: ...

    @abstractmethod
    def all_entities(self) -> Iterable[Entity]: ...

    @abstractmethod
    def all_relations(self) -> Iterable[Relation]: ...

    @abstractmethod
    def set_communities(self, communities: list[Community]) -> None: ...

    @abstractmethod
    def all_communities(self) -> list[Community]: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class NetworkXGraphStore(GraphStore):
    """In-memory graph backed by NetworkX, persisted to one JSON file.

    Entity name lookup is case-insensitive (we always lowercase on input).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.g: nx.Graph = nx.Graph()
        self.communities: list[Community] = []
        self._load_if_exists()

    # ---- entities -------------------------------------------------------

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def upsert_entity(self, entity: Entity) -> None:
        key = self._key(entity.name)
        normalized = Entity(
            name=key,
            type=entity.type,
            description=entity.description,
            source_chunks=list(entity.source_chunks),
        )
        if key in self.g:
            existing = self._node_to_entity(key)
            existing.merge(normalized)
            self.g.nodes[key].update(self._entity_to_node(existing))
        else:
            self.g.add_node(key, **self._entity_to_node(normalized))

    def get_entity(self, name: str) -> Entity | None:
        key = self._key(name)
        if key not in self.g:
            return None
        return self._node_to_entity(key)

    def all_entities(self) -> Iterable[Entity]:
        for name in self.g.nodes:
            yield self._node_to_entity(name)

    def entity_count(self) -> int:
        return self.g.number_of_nodes()

    # ---- relations ------------------------------------------------------

    def upsert_relation(self, relation: Relation) -> None:
        src = self._key(relation.source)
        tgt = self._key(relation.target)
        if src == tgt:
            return  # Skip self-loops — they hurt community detection.

        # Auto-create endpoint entities so we never have dangling edges.
        for endpoint in (src, tgt):
            if endpoint not in self.g:
                self.g.add_node(endpoint, **self._entity_to_node(
                    Entity(name=endpoint, type="unknown")
                ))

        normalized = Relation(
            source=src,
            target=tgt,
            description=relation.description,
            weight=relation.weight,
            source_chunks=list(relation.source_chunks),
        )
        if self.g.has_edge(src, tgt):
            existing = self._edge_to_relation(src, tgt)
            existing.merge(normalized)
            self.g.edges[src, tgt].update(self._relation_to_edge(existing))
        else:
            self.g.add_edge(src, tgt, **self._relation_to_edge(normalized))

    def all_relations(self) -> Iterable[Relation]:
        for u, v in self.g.edges:
            yield self._edge_to_relation(u, v)

    def relation_count(self) -> int:
        return self.g.number_of_edges()

    # ---- direct description overrides (used by LLM consolidator) --------
    #
    # These bypass merge() so the consolidator can REPLACE (not concatenate)
    # the description with an LLM-summarized version.

    def replace_entity_description(self, name: str, new_description: str) -> None:
        """Overwrite an existing entity's description in place.

        Used by description_merger after LLM consolidation. Going through
        upsert_entity would trigger merge() and re-concatenate text — the
        opposite of what we want here.
        """
        key = self._key(name)
        if key not in self.g:
            logger.warning(f"replace_entity_description: '{name}' not in graph")
            return
        self.g.nodes[key]["description"] = new_description

    def replace_relation_description(
        self, source: str, target: str, new_description: str
    ) -> None:
        """Overwrite an existing relation's description in place."""
        src = self._key(source)
        tgt = self._key(target)
        if not self.g.has_edge(src, tgt):
            logger.warning(
                f"replace_relation_description: edge {source}↔{target} not in graph"
            )
            return
        self.g.edges[src, tgt]["description"] = new_description

    # ---- traversal ------------------------------------------------------

    def neighbors(self, name: str, depth: int = 1) -> list[Entity]:
        """BFS neighbors up to `depth` hops, excluding the start node."""
        key = self._key(name)
        if key not in self.g:
            return []
        if depth <= 0:
            return []
        visited = {key}
        frontier = {key}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for nb in self.g.neighbors(node):
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(key)
        return [self._node_to_entity(n) for n in visited]

    # ---- communities ----------------------------------------------------

    def set_communities(self, communities: list[Community]) -> None:
        self.communities = list(communities)

    def all_communities(self) -> list[Community]:
        return list(self.communities)

    # ---- persistence ----------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": [self._serialize_entity(self._node_to_entity(n)) for n in self.g.nodes],
            "relations": [self._serialize_relation(self._edge_to_relation(u, v)) for u, v in self.g.edges],
            "communities": [self._serialize_community(c) for c in self.communities],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved → {self.path} ({self.entity_count()} entities, {self.relation_count()} relations)")

    def _load_if_exists(self) -> None:
        """Load the graph from disk. If the file is corrupt, rename it aside
        so the next save() doesn't blow away what might still be recoverable.
        """
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                self.path.rename(backup)
                logger.error(
                    f"Graph file {self.path} is unreadable ({e}). "
                    f"Moved aside to {backup}. Starting fresh."
                )
            except OSError as ren_err:
                logger.error(
                    f"Graph file {self.path} is unreadable ({e}) and could not be moved aside ({ren_err}). "
                    "Starting fresh — next save will overwrite the corrupt file."
                )
            return

        for e in data.get("entities", []):
            self.g.add_node(e["name"], **{
                "type": e.get("type", "unknown"),
                "description": e.get("description", ""),
                "source_chunks": e.get("source_chunks", []),
            })
        for r in data.get("relations", []):
            self.g.add_edge(r["source"], r["target"], **{
                "description": r.get("description", ""),
                "weight": float(r.get("weight", 1.0)),
                "source_chunks": r.get("source_chunks", []),
            })
        self.communities = [
            Community(
                id=c["id"],
                entity_names=c["entity_names"],
                summary=c.get("summary", ""),
                level=c.get("level", 0),
                extra=c.get("extra", {}),
                # Newer fields — defaulted for backward compatibility with
                # graphs saved before task #23.
                title=c.get("title", ""),
                rank=float(c.get("rank", 0.0)),
                rank_explanation=c.get("rank_explanation", ""),
                findings=[
                    Finding(
                        summary=f.get("summary", ""),
                        explanation=f.get("explanation", ""),
                    )
                    for f in c.get("findings", [])
                ],
            )
            for c in data.get("communities", [])
        ]

    def clear(self) -> None:
        self.g.clear()
        self.communities = []
        if self.path.exists():
            self.path.unlink()

    # ---- internal serializers ------------------------------------------

    @staticmethod
    def _entity_to_node(e: Entity) -> dict:
        return {
            "type": e.type,
            "description": e.description,
            "source_chunks": list(e.source_chunks),
        }

    def _node_to_entity(self, name: str) -> Entity:
        n = self.g.nodes[name]
        return Entity(
            name=name,
            type=n.get("type", "unknown"),
            description=n.get("description", ""),
            source_chunks=list(n.get("source_chunks", [])),
        )

    @staticmethod
    def _relation_to_edge(r: Relation) -> dict:
        return {
            "description": r.description,
            "weight": r.weight,
            "source_chunks": list(r.source_chunks),
        }

    def _edge_to_relation(self, u: str, v: str) -> Relation:
        e = self.g.edges[u, v]
        return Relation(
            source=u,
            target=v,
            description=e.get("description", ""),
            weight=float(e.get("weight", 1.0)),
            source_chunks=list(e.get("source_chunks", [])),
        )

    @staticmethod
    def _serialize_entity(e: Entity) -> dict:
        return {
            "name": e.name,
            "type": e.type,
            "description": e.description,
            "source_chunks": e.source_chunks,
        }

    @staticmethod
    def _serialize_relation(r: Relation) -> dict:
        return {
            "source": r.source,
            "target": r.target,
            "description": r.description,
            "weight": r.weight,
            "source_chunks": r.source_chunks,
        }

    @staticmethod
    def _serialize_community(c: Community) -> dict:
        return {
            "id": c.id,
            "entity_names": c.entity_names,
            "summary": c.summary,
            "level": c.level,
            "extra": c.extra,
            # Structured report fields (empty defaults for legacy data).
            "title": c.title,
            "rank": c.rank,
            "rank_explanation": c.rank_explanation,
            "findings": [
                {"summary": f.summary, "explanation": f.explanation}
                for f in c.findings
            ],
        }


# --------------------------------------------------------------------------
# Factory — change here to swap backends
# --------------------------------------------------------------------------


def default_store_path(kb_name: str) -> Path:
    from ragkit.config import get_config
    return get_config().storage_dir / "graphs" / f"{kb_name}.json"


def open_store(kb_name: str) -> GraphStore:
    """Return the store for a knowledge base. Change this function to swap backends."""
    return NetworkXGraphStore(path=default_store_path(kb_name))
