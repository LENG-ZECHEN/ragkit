"""Graph RAG data types.

All graph-related data flows through these three dataclasses. Mutability is
intentional for builder/summarizer to attach derived fields incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """A node in the knowledge graph.

    Attributes:
        name: Canonical lowercase name used as the graph key.
        type: Free-form category ("person", "company", "concept", ...).
        description: Aggregated description from all chunks that mention it.
        source_chunks: doc_id/chunk_id pairs where this entity appeared.
    """

    name: str
    type: str
    description: str = ""
    source_chunks: list[str] = field(default_factory=list)

    def merge(self, other: "Entity") -> None:
        """Merge another mention of the same entity into this one."""
        if other.type and other.type != self.type:
            # Keep both types separated by "/" rather than picking one.
            existing = set(self.type.split("/"))
            existing.add(other.type)
            self.type = "/".join(sorted(existing))
        if other.description and other.description not in self.description:
            self.description = (
                self.description + " " + other.description
            ).strip() if self.description else other.description
        for sc in other.source_chunks:
            if sc not in self.source_chunks:
                self.source_chunks.append(sc)


@dataclass
class Relation:
    """An edge between two entities."""

    source: str
    target: str
    description: str = ""
    weight: float = 1.0
    source_chunks: list[str] = field(default_factory=list)

    def merge(self, other: "Relation") -> None:
        """Merge another observation of the same edge (undirected match)."""
        if other.description and other.description not in self.description:
            self.description = (
                self.description + " " + other.description
            ).strip() if self.description else other.description
        self.weight += other.weight
        for sc in other.source_chunks:
            if sc not in self.source_chunks:
                self.source_chunks.append(sc)


@dataclass
class Finding:
    """One key fact discovered about a community.

    Findings are surfaced individually so the global retriever can show
    them as bullet points to the LLM, instead of burying them in one
    long summary paragraph.
    """

    summary: str        # one-line conclusion
    explanation: str    # detail (~50-100 chars)


@dataclass
class Community:
    """A cluster of entities discovered by community detection.

    Now carries a structured "report" (title + summary + rank + findings)
    in addition to the legacy ``summary`` field, mirroring Microsoft
    GraphRAG's community report output.
    """

    id: int
    entity_names: list[str]
    summary: str = ""  # legacy single-string summary, kept for back-compat
    level: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    # Structured report fields (filled by summarizer; default empties so
    # old JSON files still load cleanly).
    title: str = ""
    rank: float = 0.0
    rank_explanation: str = ""
    findings: list[Finding] = field(default_factory=list)
