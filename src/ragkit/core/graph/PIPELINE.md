# Graph RAG Pipeline — End-to-End Trace

> **What this is**: An engineering walkthrough of what happens when you run
> `rag index ./file.pdf --kb mykb --build-graph` (or `rag graph build --kb mykb`),
> tracing every code path, decision point, side-effect, and abort threshold.
>
> **Audience**: Anyone who needs to modify the pipeline, debug a corrupt graph,
> or estimate cost/latency. Read alongside the source files in this directory.

---

## Pipeline at a Glance

```
ENTRY:  cli/commands.py:cmd_index
          ↓ if --build-graph
        core/indexer.py:index_file
          │
          ├── (1) chunk_file ─────────────── core/chunker.py
          ├── (2) embed_batch ─────────────── core/embedder.py
          ├── (3) ESConnection.insert ─────── _ragflow/rag/utils/es_conn.py
          │                                   ⇒ writes ES index {kb}
          │
          └── build_graph ──────────────────── core/graph/builder.py
                ├── (4) open_store ─────────── core/graph/store.py
                │                              ⇐ reads storage/graphs/{kb}.json
                ├── (5) extract_from_text ──── core/graph/extractor.py    [LLM × N]
                ├── (5.5) consolidate_all ──── core/graph/description_merger.py [LLM × ≤20]
                ├── (6) detect_communities ─── core/graph/community.py
                ├── (7) summarize_all ──────── core/graph/summarizer.py   [LLM × ≤20]
                ├── (8) store.save ─────────── core/graph/store.py
                │                              ⇒ writes storage/graphs/{kb}.json
                └── (9) index_graph_to_es ──── core/graph/es_indexer.py
                                               ⇒ writes ES index {kb}_graph
```

**Three side-effects sinks**, all KB-scoped:
1. ES index `{kb}` — chunks with BM25 + dense vector (step 3)
2. ES index `{kb}_graph` — entity docs + community report docs (step 9)
3. File `storage/graphs/{kb}.json` — the graph's source-of-truth (step 8)

---

## Step 1 — Chunk the file

**File**: `core/chunker.py:chunk_file`
**Triggered by**: Always.
**Input**: file path
**Output**: `list[dict]`, each `dict` has at least `content_with_weight`, plus `title_tks`, `important_kwd`, etc.

**Behavior**:
- Dispatches by file extension to one of `_ragflow/deepdoc/parser/*.py`
- PDFs go through OCR + layout + table recognition (deepdoc)
- Output goes through `_ragflow/rag/app/naive.py` for token-budgeted merging (default `chunk_token_num=128`)

**Side-effects**:
- 🌐 First PDF triggers `huggingface_hub.snapshot_download` of OCR models (~340 MB to `~/.cache/huggingface/`)
- 📁 First tokenizer use builds `_ragflow/rag/res/huqie.txt.trie` (~30 s, ~52 MB)

**Failure modes**:
- 0 chunks produced → `observe.show_chunks_produced` warns; `index_file` returns early without writing anything

---

## Step 2 — Embed chunks

**File**: `core/embedder.py:embed_batch`
**Triggered by**: Always (unless 0 chunks).
**Input**: `list[str]` (chunk contents)
**Output**: `list[list[float] | None]` — same length as input; `None` means that chunk's embedding API call failed.

**Behavior**:
- Slices input into batches of `_MAX_BATCH = 10` (DashScope hard limit)
- Calls `client.embeddings.create(model=cfg.embedding_model, ...)` per batch
- Aligns output by position

**Decision point — abort threshold** (`indexer.py:91-99`):
```python
ratio = none_count / len(vectors)
if ratio > 0.1:
    raise RuntimeError(...)    # >10% failures → hard abort
logger.warning(...)              # ≤10% → continue, log
```

**Side-effects**:
- 💰 DashScope embedding tokens billed

---

## Step 3 — Write chunks to ES

**File**: `_ragflow/rag/utils/es_conn.py:ESConnection.insert`
**Triggered by**: Always.
**Input**: `list[dict]` (chunk documents)

**Behavior**:
- For each chunk, computes `id = xxhash64(content + kb_name).hexdigest()`
- Calls ES `_bulk` with all docs
- Same `id` → ES upsert **overwrites** existing doc

**Side-effects**:
- 🗂️ ES index `{kb}` gets/updates chunk docs containing:
  - `q_1024_vec` (dense vector)
  - `content_ltks`, `title_tks` (tokenized BM25 fields)
  - `tag_feas`, `pagerank_fea` (rank features)
  - `kb_id`, `doc_id`, `docnm_kwd` (metadata)

**Failure modes**:
- `es.insert` returns non-empty error list → `RuntimeError` (hard abort)

> If `--build-graph` is **not** set, execution returns here.

---

## Step 4 — Load existing graph

**File**: `core/graph/store.py:NetworkXGraphStore.__init__` → `_load_if_exists`
**Triggered by**: `--build-graph` (via `open_store(kb_name)`)
**Input**: KB name (used to compute path)
**Output**: An in-memory `NetworkXGraphStore` with whatever was on disk.

**Decision point — 3 cases**:
| Disk state | Resulting in-memory store |
|---|---|
| `storage/graphs/{kb}.json` does **not exist** | empty graph |
| Valid JSON | fully loaded (entities + relations + communities) |
| Corrupt JSON | file renamed to `.corrupt`; empty graph created; warned |

**Why this matters**: This is the single decision that makes `rag index --build-graph` **incremental by default**. There is no "fresh build" flag — to start fresh you must `rag graph clear NAME` first.

**Side-effects**: Only reads JSON (no writes yet).

---

## Step 5 — Extract entities and relations per chunk

**File**: `core/graph/extractor.py:extract_from_text` (called from `builder.py:54-70`)
**Triggered by**: `--build-graph`
**Input**: chunk text + chunk_id
**Output**: `ExtractionResult(entities=[...], relations=[...])`

**Behavior**:
- One LLM chat call per chunk
- Prompt asks for structured JSON: `{"entities": [...], "relations": [...]}`
- Output is parsed defensively (handles code fences, missing keys, dangling edges)

**Per-chunk loop** in `builder.py`:
```python
for i, chunk in enumerate(chunks):
    result = extract_from_text(text, chunk_id)
    for entity in result.entities:
        store.upsert_entity(entity)       # ← merge semantics
    for relation in result.relations:
        store.upsert_relation(relation)
```

**`upsert_entity` merge logic** (`store.py:81-130`):
- **Case-insensitive name** matching → existing entity with same lowercase name found
- If found:
  - `type`: union of old + new
  - `description`: **string concatenation** (`old + " | " + new`)
  - `source_chunks`: set union (no duplicates)
  - `weight`: += 1
- If not found: new entity added

**Decision point — abort threshold** (`builder.py:73-79`):
```python
failure_ratio = extraction_failures / total
if failure_ratio > 0.5 and store.entity_count() == 0:
    raise RuntimeError("Refusing to save an empty graph")
```

**The AND is important**: incremental builds with a loaded existing graph won't trigger this even if the current batch fully fails — because `store.entity_count() > 0` from the loaded data.

**Side-effects**:
- 💰 N LLM chat calls (N = chunk count). **Largest cost driver**.
- 🧠 In-memory `store` mutated

---

## Step 5.5 — Consolidate long descriptions

**File**: `core/graph/description_merger.py:consolidate_all`
**Triggered by**: `--build-graph` AND `consolidate_descriptions=True` (CLI default; `--no-consolidate` disables)
**Input**: store
**Output**: stats dict (consolidated count, LLM call count, failures)

**Behavior**:
- Scans every entity (and every relation)
- For each, checks **trigger conditions**:
  ```python
  if len(entity.description) > CONSOLIDATION_TRIGGER (250) and \
     entity.weight > MIN_SOURCES_FOR_CONSOLIDATION (3):
      # description has grown long AND entity is seen in >3 chunks
      → LLM rewrite to ≤ 180 chars
  ```
- 70-char buffer between trigger (250) and target (180) avoids "summary of summary" loops
- Shared cap: at most `max_calls=20` LLM calls per build for entities + relations combined

**Why the cap exists**: Without it, a build of a large corpus could trigger hundreds of consolidation calls.

**Side-effects**:
- 💰 Up to 20 LLM chat calls
- 🧠 `entity.description` / `relation.description` rewritten in place

---

## Step 6 — Detect communities

**File**: `core/graph/community.py:detect_communities`
**Triggered by**: `--build-graph` (always; no flag to disable)
**Input**: store
**Output**: `list[Community]` (entity-name groupings + level metadata)

**Behavior**:
- Builds an undirected `networkx.Graph` from the store's relation edges
- Calls `community_louvain.generate_dendrogram(g, ...)` for hierarchical Louvain
- Walks the dendrogram up to `MAX_LEVELS=3` levels, producing `Community` objects at each level
- Tiny communities (< `MIN_COMMUNITY_SIZE=3`) pooled into a "misc" bucket per level
- Single-node graph or no-edge graph: early return `[]` (ISS-018 fix)

**Decision point — replacement semantics**:
```python
communities = detect_communities(store)
store.set_communities(communities)    # ⚠️ total replacement, not merge
```

Communities are **rebuilt from scratch every time**. Louvain output is not deterministic (depends on node insertion order), so community IDs are not stable across builds.

**Implication**: `rag graph report mykb 3` may refer to a different community after rebuild. Use `community.title` as the human-stable identifier, not ID.

**Side-effects**: Only mutates in-memory `store._communities`.

---

## Step 7 — Generate community reports

**File**: `core/graph/summarizer.py:summarize_all` → `generate_community_report`
**Triggered by**: `--build-graph` AND `summarize=True` (CLI default; `--no-summarize` disables)
**Input**: store
**Output**: failure count (the community objects are mutated in place)

**Behavior** (`summarizer.py`):
```python
for community in store.all_communities()[:max_communities]:
    if not community.entity_names:
        continue
    report = generate_community_report(community, store)   # 1 LLM call
    community.title = report.title
    community.summary = report.summary
    community.rank = report.rank
    community.rank_explanation = report.rank_explanation
    community.findings = report.findings
```

**Per-community LLM call** generates structured JSON:
```json
{
  "title": "...",
  "summary": "...",
  "rank": 0-10,
  "rank_explanation": "...",
  "findings": [{"summary": "...", "explanation": "..."}, ...]
}
```

**Decision point — `max_communities` cap**:
- Default `max_communities=20`
- Only the first 20 communities (by current insertion order, no sorting) get reports
- Communities beyond 20 stay in `store._communities` but with empty title/summary/findings — they will be **skipped during ES indexing** (step 9, filtered by `if c.title or c.summary or c.findings`)

**Failure isolation**:
- Per-community LLM failures don't abort the loop
- A failed community keeps its previous (or empty) fields
- `failures` counter is returned for visibility

**Side-effects**:
- 💰 Up to `max_communities` LLM chat calls (default 20)
- 🧠 Each community gets `title`/`summary`/`rank`/`findings` filled in

---

## Step 8 — Persist graph to JSON

**File**: `core/graph/store.py:NetworkXGraphStore.save`
**Triggered by**: `--build-graph` (always)
**Input**: store
**Output**: file write

**Behavior**:
```python
data = {
    "entities": [e.to_dict() for e in self._entities()],
    "relations": [r.to_dict() for r in self._edges()],
    "communities": [c.to_dict() for c in self._communities],
}
tmp_path = self.path.with_suffix(".tmp")
tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
tmp_path.rename(self.path)   # atomic rename (POSIX guarantee on same filesystem)
```

**Key**: The write is **atomic** via temp-file + rename. If the process is killed mid-write, the old file remains intact (or doesn't exist — never half-written).

**Side-effects**:
- 📂 `storage/graphs/{kb}.json` is overwritten with current in-memory state

> This is the **single source of truth** for graph data. ES `{kb}_graph` is treated as a derived cache.

---

## Step 9 — Synchronize graph to ES

**File**: `core/graph/es_indexer.py:index_graph_to_es`
**Triggered by**: `--build-graph` AND `index_to_es=True` (always set true by `builder.py`)
**Input**: store, kb_name
**Output**: stats dict `{entity_embedded, entity_skipped, entity_failed, community_embedded, community_failed}`

This step has **two distinct phases** with very different incremental behavior:

### Phase A — Entities (incremental via `desc_hash`)

```python
existing = _fetch_existing_entity_hashes(kb_name, raw_es)
# → {entity_name: desc_hash} for all entity docs currently in ES

to_embed = [
    e for e in store.all_entities()
    if _entity_desc_hash(e) != existing.get(e.name)
]
# Only entities whose name+description has changed need re-embedding
```

**`desc_hash` formula**:
```python
xxhash64(entity.name + "\x00" + entity.description).hexdigest()[:16]
```

- Stable input (no change) → same hash → entity is skipped
- Description changed (merge or consolidation) → hash changed → re-embed

**Abort threshold**:
```python
if entity_failed / len(to_embed) > EMBED_FAILURE_ABORT_RATIO (0.1):
    raise RuntimeError(...)   # >10% embed failures aborts
```

**Robustness fallback** (ISS-006 fix):
- If `_fetch_existing_entity_hashes` itself fails (e.g., ES transient error), it returns `{}`
- Then every entity is treated as new → all re-embedded
- This degrades gracefully rather than crashing the whole build

### Phase B — Communities (full delete + reinsert)

```python
_delete_community_docs(kb_name, raw_es)
# DELETE BY QUERY: type_kwd=community AND kb_id=kb_name

# Skip communities with no usable content
communities = [c for c in store.all_communities()
               if c.title or c.summary or c.findings]

# Embed + bulk insert all communities
```

**Why no incremental for communities?** Louvain IDs aren't stable across builds. There's no way to know whether `community_id_int=3` today is the same conceptual group as `community_id_int=3` last week.

**Abort threshold**: same as entities (10% community embed failures aborts).

### Error isolation across steps 8 and 9

```python
# In builder.py:
store.save()    # step 8 — JSON written first

if index_to_es:
    try:
        es_stats = index_graph_to_es(store, kb_name, ...)
    except Exception as e:
        logger.error("Graph ES indexing failed; JSON graph is still saved; "
                     "rerun `rag graph build` after fixing ES.")
```

**Critical**: JSON write happens **before** ES sync. If ES is down, you still have the graph — just not searchable via ES. Re-run `rag graph build` once ES recovers; it will retry the ES sync without re-extracting (because chunks are unchanged and `_load_if_exists` will reload the graph).

**Side-effects**:
- 💰 N + M embeddings (N = entities changed, M = communities with content)
- 🗂️ ES index `{kb}_graph` updated:
  - Entity docs (`type_kwd=entity`): name, type, description, source_chunks, embedding, desc_hash
  - Community docs (`type_kwd=community`): level, id, rank, entity_names, embedding, full report content

---

## Decision Matrix — When Do Things Get Recomputed?

| Operation | First-time build | Re-build (incremental) | Different `kb_name` |
|---|---|---|---|
| Chunk parse | Yes | Only new files | Yes |
| Chunk embed | Yes | Only new chunks (deduped by ID) | Yes |
| ES `{kb}` write | Yes | Upsert by chunk_id | New index |
| Load existing graph | No (empty) | Yes (from JSON) | No (empty) |
| Entity extraction | Yes, all chunks | All chunks passed to builder | Yes, all chunks |
| Entity merge | New entities only | Same-name entities merged | New entities only |
| Description consolidation | Triggered if desc > 250 | Same trigger | Same trigger |
| Community detection | Full Louvain | **Full Louvain (replaces all)** | Full Louvain |
| Community summarization | First 20 | First 20 (new IDs!) | First 20 |
| JSON save | Overwrite | Overwrite | New file |
| ES entity sync | All new | Only `desc_hash` changed | All new (new index) |
| ES community sync | All inserted | **Full delete + insert** | All inserted |

---

## Failure Modes and Recovery

| Failure | Detection | Recovery |
|---|---|---|
| 0 chunks parsed | `observe.show_chunks_produced` warns | Returns early; no writes |
| >10% chunk embeddings fail | `RuntimeError` raised in `index_file` | Investigate API/key; nothing written yet |
| ES bulk insert returns errors | `RuntimeError` raised | Investigate ES; chunks NOT written |
| Corrupt graph JSON on load | File renamed `.corrupt` | Empty store starts; old file preserved |
| >50% extractions fail AND no entities | `RuntimeError`; refuses to save empty graph | API down; no JSON write |
| Description consolidation fails | Per-entity try/except; logged | Continues; description stays long |
| Community detection fails | Unlikely (pure NetworkX) | — |
| Per-community summary fails | Counted in `failures`; community stays empty | Survives; affected community gets no report |
| Graph JSON write fails mid-way | Atomic rename — old file intact | Re-run rebuild |
| ES graph indexing fails | Caught in `builder.py`; logged | JSON saved; re-run `rag graph build` for ES retry |

---

## Cost Anatomy of a Single Build

For a corpus of `C` chunks producing `E` entities organized into `K` communities:

| Phase | LLM chat calls | LLM embedding calls |
|---|---|---|
| (2) chunk embedding | 0 | `C / 10` (batched) |
| (5) entity extraction | `C` | 0 |
| (5.5) consolidation | up to 20 | 0 |
| (7) community summaries | `min(K, 20)` | 0 |
| (9A) entity ES sync | 0 | `changed_E / 10` (incremental) |
| (9B) community ES sync | 0 | `K / 10` (full refresh) |
| **Total per build** | **`C + 40` max** | **`(C + changed_E + K) / 10`** |

For the audit-PDF case (`C=25`, `E=167`, `K=20`):
- Chat: ~45 calls (25 extract + ~0 consolidation + 20 summary)
- Embedding: ~22 batches (3 chunks + 17 entities + 2 communities)
- Wall time: ~5 minutes

For a rebuild without source changes:
- Chat: ~45 calls again (extraction and summary always re-run)
- Embedding: ~5 batches (changed entities + all communities); chunks are already in ES, skipped
- **Conclusion**: rebuilds are not free. If you don't need to change the graph, don't run `--build-graph`.

---

## Quick Navigation — "I want to change..."

| Goal | Edit |
|---|---|
| Chunking parameters (size, overlap) | `_ragflow/rag/app/naive.py` |
| Entity extraction prompt | `core/graph/extractor.py:EXTRACT_PROMPT` |
| Consolidation thresholds (250 / 180 / 3 / 20) | `core/graph/description_merger.py` constants |
| Community algorithm (Louvain → Leiden) | `core/graph/community.py:detect_communities` |
| Community report prompt / fields | `core/graph/summarizer.py:SUMMARIZE_PROMPT` |
| Default `max_summary_communities` | `core/graph/builder.py:build_graph` signature |
| Embed-failure abort ratio | `core/graph/es_indexer.py:EMBED_FAILURE_ABORT_RATIO` |
| Storage location | `core/graph/store.py:default_store_path` |
| ES index naming convention | `core/graph/es_indexer.py:172` (`f"{kb_name}_graph"`) |
| `desc_hash` algorithm | `core/graph/es_indexer.py:_entity_desc_hash` |
| LLM model used for build | `RAG_LLM_MODEL` env var (used by all graph LLM callers) |

---

## Design Decisions and Their Reasons

| Decision | Why |
|---|---|
| `chunk_id = hash(content + kb_name)` | Same content within a KB dedupes naturally; cross-KB indices stay independent |
| JSON is source-of-truth, ES is cache | Survives ES outage; cheap to rebuild ES from JSON |
| `_load_if_exists` runs automatically | Increment-by-default is the common case; explicit "fresh" requires `graph clear` |
| `upsert_entity` merges by lowercased name | "Tesla" / "tesla" / "TESLA" same entity; "特斯拉" vs "Tesla" stay separate (CJK matching is out-of-scope) |
| Description merge concatenates rather than replaces | Multi-document corpus accumulates information |
| Consolidation triggers at `>250` + `>3 sources` | Prevent runaway concat AND avoid wasteful rewrites on single-source entities |
| Communities fully replaced each build | Louvain IDs not stable; trying to incrementally update them would corrupt cluster membership |
| Entity ES sync incremental via `desc_hash` | Most entities are unchanged across rebuilds; saves embedding cost |
| 10% / 50% abort thresholds | Loud failure beats silent partial data |
| Atomic JSON write (tmp + rename) | Process kill mid-write can't corrupt the file |
| ES failure isolated from JSON write | Don't lose graph just because ES had a hiccup |
| Per-chunk extraction failure tolerated | One bad chunk shouldn't fail the whole build |
| Per-community summary failure tolerated | One bad community shouldn't fail the whole build |

---

## Related Documentation

- `core/_ragflow/README.md` — what's vendored from RAGFlow vs ragkit-original
- `core/graph/types.py` — `Entity` / `Relation` / `Community` / `Finding` dataclasses
- `core/graph/global_search.py` (top comment) — Map-Reduce retrieval architecture
- `cli/observe.py` — what gets traced where in this pipeline (search for `show_*` and `trace_*`)
- Test suite: `tests/test_graph_*.py` covers each step in isolation; `tests/test_graph_builder_cli.py` covers the full pipeline end-to-end
