"""Graph RAG retrieval strategies.

Three modes:

  local   - identify entities in the question → BFS neighborhood → fetch
            the source chunks that mentioned those entities. Best for
            "what is X" / "how does X relate to Y" style questions.

  global  - return community summaries as context. Best for thematic
            ("what does this corpus say overall") questions.

  hybrid  - combine local results with vector retrieval, dedupe by content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragkit.core.graph.store import GraphStore, open_store
from ragkit.core.graph.types import Entity
from ragkit.core.retriever import RetrievedChunk, retrieve as vector_retrieve
from ragkit.logger import logger

# Safety bound on BFS — much beyond 3 hops loses semantic coherence
# and grows quadratically in cost.
MAX_DEPTH = 5


def _validate_args(question: str, kb_name: str, depth: int | None = None) -> None:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not kb_name or not kb_name.strip():
        raise ValueError("kb_name must be a non-empty string")
    if depth is not None and (depth < 0 or depth > MAX_DEPTH):
        raise ValueError(f"depth must be in [0, {MAX_DEPTH}], got {depth}")


@dataclass(frozen=True)
class GraphHit:
    """A retrieved evidence item — uniform shape across local/global modes.

    Local hits carry the entity/relation evidence text; global hits carry
    community summary text. Both can be flattened into the LLM prompt as
    numbered references.
    """

    rank: int
    kind: str  # "entity" | "community" | "chunk"
    title: str
    content: str
    extra: dict


def _find_mentioned_entities(question: str, store: GraphStore) -> list[Entity]:
    """Match entity names from the graph against the question text.

    Simple substring match — works well for short/medium graphs and
    avoids an extra LLM call. For very large graphs we could swap in
    embedding-based entity disambiguation.
    """
    q_lower = question.lower()
    matches: list[Entity] = []
    for entity in store.all_entities():
        if not entity.name:
            continue
        # Skip super short names (<2 chars) that would match too eagerly.
        if len(entity.name) < 2:
            continue
        # Word-boundary-ish match: works for both English tokens and Chinese.
        if re.search(re.escape(entity.name), q_lower):
            matches.append(entity)
    return matches


def retrieve_local(
    question: str,
    kb_name: str,
    *,
    depth: int = 1,
    top_k: int = 5,
    store: GraphStore | None = None,
) -> list[GraphHit]:
    """Local graph retrieval.

    Find entities mentioned in the question, expand BFS to `depth` hops,
    and return them with their descriptions and connecting relations.
    """
    _validate_args(question, kb_name, depth)
    store = store or open_store(kb_name)
    seed_entities = _find_mentioned_entities(question, store)
    if not seed_entities:
        logger.info(f"Local retrieval: no entities in question matched the graph")
        return []

    # Collect all entities in the neighborhood (deduped by name).
    collected: dict[str, Entity] = {e.name: e for e in seed_entities}
    for seed in seed_entities:
        for nb in store.neighbors(seed.name, depth=depth):
            collected.setdefault(nb.name, nb)

    # Build hits: render each entity + its outgoing edges as one passage.
    # Cap edges per entity to keep prompts short.
    MAX_EDGES_PER_ENTITY = 5
    hits: list[GraphHit] = []
    relations = list(store.all_relations())
    for i, entity in enumerate(collected.values(), start=1):
        rel_lines: list[str] = []
        for r in relations:
            if r.source == entity.name or r.target == entity.name:
                other = r.target if r.source == entity.name else r.source
                rel_lines.append(f"{entity.name} ↔ {other}: {r.description}")
        # Assemble content from non-empty parts only — avoids stray leading "\n"
        # when description is empty (common for stub entities auto-created on
        # dangling edges).
        parts: list[str] = []
        if entity.description:
            parts.append(entity.description)
        if rel_lines:
            parts.append("关系：\n" + "\n".join(rel_lines[:MAX_EDGES_PER_ENTITY]))
        content = "\n".join(parts)
        hits.append(GraphHit(
            rank=i,
            kind="entity",
            title=f"{entity.name} [{entity.type}]",
            content=content,
            extra={"source_chunks": entity.source_chunks},
        ))

    # Rank by number of relations (rough centrality proxy).
    hits.sort(
        key=lambda h: len(h.extra.get("source_chunks", [])),
        reverse=True,
    )
    return hits[:top_k]


_MAX_FINDINGS_IN_HIT = 3


def _render_community_hit_content(community) -> str:
    """Render a Community's structured report into a single passage the
    LLM can consume.

    Includes title, summary, and the top N findings (cap to keep prompts
    short). Falls back gracefully when the report is incomplete (e.g.
    legacy data with only ``summary``).
    """
    parts: list[str] = []
    if community.title:
        parts.append(community.title)
    if community.summary:
        parts.append(community.summary)
    if community.findings:
        lines = ["关键发现："]
        for f in community.findings[:_MAX_FINDINGS_IN_HIT]:
            head = f.summary.strip()
            body = f.explanation.strip()
            if head and body:
                lines.append(f"- {head}: {body}")
            elif head:
                lines.append(f"- {head}")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def retrieve_global(
    question: str,
    kb_name: str,
    *,
    top_k: int = 3,
    store: GraphStore | None = None,
) -> list[GraphHit]:
    """Global retrieval — return the top community reports.

    Uses a lightweight token-overlap proxy on title + summary. Once task
    #24/#25 lands, this will be replaced by ES vector search.
    """
    _validate_args(question, kb_name)
    store = store or open_store(kb_name)
    # A community is "globally retrievable" if it has any report content —
    # title, summary, or findings.
    communities = [
        c for c in store.all_communities()
        if c.summary or c.title or c.findings
    ]
    if not communities:
        return []

    q_tokens = set(re.findall(r"\w+", question.lower()))

    def score(community) -> int:
        text = community.title + " " + community.summary
        for f in community.findings:
            text += " " + f.summary
        s_tokens = set(re.findall(r"\w+", text.lower()))
        return len(q_tokens & s_tokens)

    ranked = sorted(
        communities,
        key=lambda c: (score(c), c.rank, len(c.entity_names)),
        reverse=True,
    )
    return [
        GraphHit(
            rank=i,
            kind="community",
            title=c.title or f"Community {c.id}",
            content=_render_community_hit_content(c),
            extra={
                "entity_names": c.entity_names,
                "level": c.level,
                "rank": c.rank,
            },
        )
        for i, c in enumerate(ranked[:top_k], start=1)
    ]


def retrieve_hybrid(
    question: str,
    kb_name: str,
    *,
    top_k: int = 5,
    vector_weight: float = 0.6,
    store: GraphStore | None = None,
) -> list[GraphHit]:
    """Hybrid: vector retrieval + local graph retrieval, deduped by content.

    Vector hits are returned first (they tend to be more specific); graph
    hits add cross-entity context that pure vector misses.

    Note on vector failures: we log loudly and continue with graph-only
    results. The caller can detect this by counting hits where `kind=='chunk'`
    (zero means the vector half didn't contribute).
    """
    _validate_args(question, kb_name)
    store = store or open_store(kb_name)

    try:
        vector_chunks = vector_retrieve(
            question,
            kb_name=kb_name,
            top_k=top_k,
            vector_similarity_weight=vector_weight,
        )
    except Exception as e:
        # Loud log — this is degraded mode, not normal operation.
        logger.error(
            f"Vector retrieval FAILED in hybrid mode ({e}). "
            "Returning graph-only results. Check ES connection."
        )
        vector_chunks = []

    vector_hits = [
        GraphHit(
            rank=i,
            kind="chunk",
            title=c.document_name,
            content=c.content,
            extra={"similarity": c.similarity, "document_id": c.document_id},
        )
        for i, c in enumerate(vector_chunks, start=1)
    ]

    local_hits = retrieve_local(question, kb_name, top_k=top_k, store=store)

    # Dedupe by full-content hash (xxhash is already a project dep).
    # Using a prefix would collide on documents with shared boilerplate headers.
    import xxhash

    seen: set[str] = set()
    merged: list[GraphHit] = []
    for hit in vector_hits + local_hits:
        normalized = hit.content.strip()
        if not normalized:
            continue
        key = xxhash.xxh64(normalized.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)

    # Re-number ranks.
    return [
        GraphHit(rank=i, kind=h.kind, title=h.title, content=h.content, extra=h.extra)
        for i, h in enumerate(merged[:top_k * 2], start=1)
    ]


def graph_hits_to_chunks(hits: list[GraphHit]) -> list[RetrievedChunk]:
    """Convert GraphHits to RetrievedChunks so the existing generator can
    consume them through its current interface.
    """
    return [
        RetrievedChunk(
            rank=h.rank,
            document_id=h.extra.get("document_id", h.kind),
            document_name=h.title,
            content=h.content,
            similarity=float(h.extra.get("similarity", 0.0)),
            vector_similarity=0.0,
            term_similarity=0.0,
        )
        for h in hits
    ]
