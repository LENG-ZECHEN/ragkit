"""Index graph artifacts (entities and community reports) into a dedicated
Elasticsearch index ``{kb_name}_graph``.

This is the bridge between the in-memory NetworkX graph (kept in JSON) and
the vector-search infrastructure (Elasticsearch). After this step runs,
graph-based retrieval (task #25) can use ES dense_vector + BM25 instead
of substring matching / token overlap.

Two document types live in ``{kb_name}_graph``:
  type_kwd="entity"     — one doc per entity (description-derived embedding)
  type_kwd="community"  — one doc per community report (all levels, summary-derived)

Notes
-----
- Relations are NOT indexed (per design — they're surfaced via their
  endpoint entities).
- Entity ID is stable (xxhash of name); a ``desc_hash_kwd`` field lets
  us detect when an entity's description has changed and re-embed only
  the affected entries.
- Community IDs are NOT stable across rebuilds (Louvain partitions
  shift). So every build deletes-then-rewrites all community docs.
"""

from __future__ import annotations

from typing import Callable, Iterable

import xxhash

from ragkit.core.embedder import embed_batch
from ragkit.core.graph.store import GraphStore
from ragkit.core.graph.types import Community, Entity
from ragkit.core.rag.utils.es_conn import ESConnection
from ragkit.logger import logger


# Re-embed is aborted if more than this fraction of embedding calls fail.
EMBED_FAILURE_ABORT_RATIO = 0.1


# --------------------------------------------------------------------------
# ID and hash helpers
# --------------------------------------------------------------------------


def _entity_doc_id(entity_name: str) -> str:
    """Stable per-entity ES document ID (deterministic from name)."""
    return f"ent-{xxhash.xxh64(entity_name.encode('utf-8')).hexdigest()}"


def _community_doc_id(community: Community) -> str:
    """Community doc IDs encode (level, id). Not stable across builds —
    that's why we delete+rewrite all community docs each build."""
    return f"com-{community.level}-{community.id}"


def _entity_desc_hash(entity: Entity) -> str:
    """Short hash of the text used to embed an entity. Lets us detect
    description changes without re-embedding everything."""
    text = entity.name + "\x00" + entity.description
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Render text for embedding and for human-readable content_with_weight
# --------------------------------------------------------------------------


def _entity_embed_text(entity: Entity) -> str:
    """Text fed to the embedding model. Putting the name first gives the
    vector a strong identifying signal so semantic matches work even when
    the user paraphrases (e.g. '通义千问 plus' → entity 'qwen-plus')."""
    return f"{entity.name}. {entity.description or ''}".strip()


def _entity_display_text(entity: Entity) -> str:
    """Human-readable text stored on the doc (shown to LLM in retrieval)."""
    return f"{entity.name} [{entity.type}]: {entity.description or ''}".rstrip(": ").rstrip()


def _community_embed_text(community: Community) -> str:
    """Text fed to the embedding model for a community.

    Includes title + summary + finding HEADS (not full explanations) — long
    explanations dilute the embedding's specificity.
    """
    parts: list[str] = []
    if community.title:
        parts.append(community.title)
    if community.summary:
        parts.append(community.summary)
    if community.findings:
        finding_heads = "; ".join(
            f.summary for f in community.findings if f.summary
        )
        if finding_heads:
            parts.append(finding_heads)
    return ". ".join(parts) if parts else f"Community {community.id}"


def _community_display_text(community: Community) -> str:
    """Human-readable text shown to the LLM — full report including
    findings explanations (top 3 to keep prompts short)."""
    parts: list[str] = []
    if community.title:
        parts.append(community.title)
    if community.summary:
        parts.append(community.summary)
    if community.findings:
        lines = ["关键发现："]
        for f in community.findings[:3]:
            head = (f.summary or "").strip()
            body = (f.explanation or "").strip()
            if head and body:
                lines.append(f"- {head}: {body}")
            elif head:
                lines.append(f"- {head}")
        parts.append("\n".join(lines))
    return "\n".join(parts) if parts else f"Community {community.id}"


# --------------------------------------------------------------------------
# Document builders (pure functions — easy to unit-test)
# --------------------------------------------------------------------------


def _build_entity_doc(entity: Entity, embedding: list[float], kb_name: str) -> dict:
    """Assemble one ES document for an entity. Pure function."""
    display = _entity_display_text(entity)
    return {
        "id": _entity_doc_id(entity.name),
        "type_kwd": "entity",
        "content_with_weight": display,
        "content_ltks": display,  # whitespace-tokenized by ES mapping
        f"q_{len(embedding)}_vec": embedding,
        "kb_id": kb_name,
        "entity_name_kwd": entity.name,
        "entity_type_kwd": entity.type or "unknown",
        "source_chunks_kwd": list(entity.source_chunks),
        "desc_hash_kwd": _entity_desc_hash(entity),
    }


def _build_community_doc(
    community: Community, embedding: list[float], kb_name: str
) -> dict:
    """Assemble one ES document for a community report. Pure function."""
    display = _community_display_text(community)
    return {
        "id": _community_doc_id(community),
        "type_kwd": "community",
        "content_with_weight": display,
        "content_ltks": display,
        f"q_{len(embedding)}_vec": embedding,
        "kb_id": kb_name,
        "community_level_int": community.level,
        "community_id_int": community.id,
        "community_rank_flt": community.rank,
        "community_entity_names_kwd": list(community.entity_names),
    }


# --------------------------------------------------------------------------
# Diff logic — which entities need (re-)embedding?
# --------------------------------------------------------------------------


def _fetch_existing_entity_hashes(kb_name: str, es) -> dict[str, str]:
    """Pull (entity_name, desc_hash) for every existing entity doc in the
    {kb_name}_graph index. Returns {} if the index doesn't exist yet.
    """
    index = f"{kb_name}_graph"
    if not es.indices.exists(index=index):
        return {}

    out: dict[str, str] = {}
    sid = None
    # Scroll over all entity docs (could be thousands).
    # ROBUSTNESS (ISS-006): wrap the initial search in try/except too — any
    # transient ES failure must degrade gracefully (treat as empty, re-embed
    # everything) rather than crash the whole index pipeline before the
    # community deletion step.
    try:
        resp = es.search(
            index=index,
            # Modern elasticsearch-py 8.x kwargs (ISS-015: replaces deprecated body=).
            query={"term": {"type_kwd": "entity"}},
            source=["entity_name_kwd", "desc_hash_kwd"],
            size=1000,
            scroll="2m",
        )
        sid = resp.get("_scroll_id")
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                name = src.get("entity_name_kwd")
                desc_hash = src.get("desc_hash_kwd")
                if name and desc_hash:
                    out[name] = desc_hash
            resp = es.scroll(scroll_id=sid, scroll="2m")
            sid = resp.get("_scroll_id", sid)
    except Exception as e:
        logger.warning(
            f"_fetch_existing_entity_hashes: ES error ({e}). "
            "Treating as empty index — all entities will be re-embedded."
        )
        return {}
    finally:
        if sid:
            try:
                es.clear_scroll(scroll_id=sid)
            except Exception as e:
                logger.debug(f"clear_scroll failed: {e}")
    return out


def _filter_entities_to_embed(
    store: GraphStore, existing_hashes: dict[str, str]
) -> list[Entity]:
    """Return only the entities whose current desc_hash differs from what
    ES has (or aren't in ES at all). These need re-embedding."""
    out: list[Entity] = []
    for entity in store.all_entities():
        current_hash = _entity_desc_hash(entity)
        if existing_hashes.get(entity.name) != current_hash:
            out.append(entity)
    return out


def _delete_community_docs(kb_name: str, es) -> None:
    """Wipe all community docs for this KB. Called before bulk-inserting
    the new community set — Louvain partition IDs aren't stable across
    rebuilds so we can't rely on doc upserts."""
    index = f"{kb_name}_graph"
    if not es.indices.exists(index=index):
        return
    try:
        # ISS-015: modern elasticsearch-py 8.x kwargs (replaces deprecated body=).
        es.delete_by_query(
            index=index,
            query={
                "bool": {
                    "must": [
                        {"term": {"type_kwd": "community"}},
                        {"term": {"kb_id": kb_name}},
                    ]
                }
            },
            refresh=True,
        )
    except Exception as e:
        logger.warning(f"delete_community_docs failed for {kb_name}: {e}")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------


def _embed_in_batches(
    items: list,
    text_fn: Callable,
    batch_size: int,
) -> list[list[float] | None]:
    """Embed items via text_fn(item). Returns per-item vector or None on
    failure for that batch."""
    vectors: list[list[float] | None] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        texts = [text_fn(it) for it in chunk]
        try:
            batch_vecs = embed_batch(texts)
        except Exception as e:
            logger.error(f"Embedding batch {i // batch_size + 1} failed: {e}")
            vectors.extend([None] * len(chunk))
            continue
        if len(batch_vecs) != len(chunk):
            # ISS-035: keep aligned vectors when possible — only mark the
            # missing positions as failed, not the whole batch.
            logger.error(
                f"Embedding returned {len(batch_vecs)} vectors for {len(chunk)} inputs "
                f"— keeping first {len(batch_vecs)}, marking rest as failed"
            )
            vectors.extend(batch_vecs[: len(chunk)])
            vectors.extend([None] * (len(chunk) - len(batch_vecs)))
            continue
        vectors.extend(batch_vecs)
    return vectors


def index_graph_to_es(
    store: GraphStore,
    kb_name: str,
    *,
    batch_size: int = 10,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Bring the {kb_name}_graph ES index in sync with the in-memory graph.

    Steps:
      1. Diff entities by desc_hash → embed only the changed/new ones
      2. Bulk insert new/changed entity docs
      3. delete_by_query all community docs (their IDs are unstable)
      4. Embed and bulk insert all current community docs (skipping any
         with empty reports — title+summary+findings all blank)

    Returns a dict with counts: {entity_embedded, community_embedded, ...}

    Failure policy: an individual embedding failure logs and skips that
    item. If >EMBED_FAILURE_ABORT_RATIO (10%) of an entire phase fails,
    raise RuntimeError to surface the underlying problem (API quota etc).
    """
    es = ESConnection()
    raw_es = es.es
    es.ensure_index(f"{kb_name}_graph")

    # ----- Entities -----------------------------------------------------
    existing = _fetch_existing_entity_hashes(kb_name, raw_es)
    to_embed = _filter_entities_to_embed(store, existing)

    if progress_cb:
        progress_cb("embedding_entities", 0, len(to_embed))

    entity_embedded = 0
    entity_failed = 0
    if to_embed:
        vectors = _embed_in_batches(to_embed, _entity_embed_text, batch_size)
        entity_docs = []
        for entity, vec in zip(to_embed, vectors):
            if vec is None:
                entity_failed += 1
                continue
            entity_docs.append(_build_entity_doc(entity, vec, kb_name))

        if entity_docs:
            es.insert(entity_docs, f"{kb_name}_graph")
            entity_embedded = len(entity_docs)

        # Abort hard if too many failures (a quota issue would silently
        # produce a half-indexed graph otherwise). ISS-036: we're already
        # inside `if to_embed:` so the prefix guard was redundant.
        if entity_failed / len(to_embed) > EMBED_FAILURE_ABORT_RATIO:
            raise RuntimeError(
                f"Entity embedding failed for {entity_failed}/{len(to_embed)} "
                f"items in {kb_name}_graph (>{EMBED_FAILURE_ABORT_RATIO:.0%}). "
                "Check API key / quota / network."
            )

        if progress_cb:
            progress_cb("embedding_entities", len(to_embed), len(to_embed))
    else:
        logger.info(f"No entity changes — skipping entity embedding for {kb_name}_graph")

    # ----- Communities (full refresh) -----------------------------------
    _delete_community_docs(kb_name, raw_es)

    # Skip communities with no usable content (no LLM report and no title).
    communities = [
        c for c in store.all_communities()
        if c.title or c.summary or c.findings
    ]

    if progress_cb:
        progress_cb("embedding_communities", 0, len(communities))

    community_embedded = 0
    community_failed = 0
    if communities:
        vectors = _embed_in_batches(communities, _community_embed_text, batch_size)
        community_docs = []
        for c, vec in zip(communities, vectors):
            if vec is None:
                community_failed += 1
                continue
            community_docs.append(_build_community_doc(c, vec, kb_name))

        if community_docs:
            es.insert(community_docs, f"{kb_name}_graph")
            community_embedded = len(community_docs)

        # ISS-036: already inside `if communities:` — prefix guard was redundant.
        if community_failed / len(communities) > EMBED_FAILURE_ABORT_RATIO:
            raise RuntimeError(
                f"Community embedding failed for {community_failed}/{len(communities)} "
                f"items in {kb_name}_graph (>{EMBED_FAILURE_ABORT_RATIO:.0%})."
            )

        if progress_cb:
            progress_cb("embedding_communities", len(communities), len(communities))

    summary = {
        "entity_embedded": entity_embedded,
        "entity_skipped": store.entity_count() - len(to_embed),
        "entity_failed": entity_failed,
        "community_embedded": community_embedded,
        "community_failed": community_failed,
    }
    logger.info(f"index_graph_to_es {kb_name}: {summary}")
    return summary
