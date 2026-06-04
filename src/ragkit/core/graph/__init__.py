"""Graph RAG — entity/relation extraction, knowledge graph, community-based retrieval.

Module layout (each file owns ONE concern — easy to swap):

    types.py        Entity / Relation / Community dataclasses
    extractor.py    LLM extracts entities + relations from a chunk
    store.py        Graph storage (NetworkX + JSON file) — adapter interface
    community.py    Cluster the graph into communities (Louvain)
    summarizer.py   LLM summarizes each community
    builder.py      Orchestrates extract → store → detect → summarize
    retriever.py    Local / global / hybrid query strategies
"""

from ragkit.core.graph.types import Community, Entity, Finding, Relation

__all__ = ["Entity", "Relation", "Community", "Finding"]
