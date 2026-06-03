# ragkit

A minimal **Retrieval-Augmented Generation (RAG)** toolkit with a CLI interface.
Supports both **vector RAG** and **Graph RAG**.

Extracted and refactored from a larger backend project — the heavy frontend, auth,
session-management, and HTTP layers are gone; what's left is the RAG pipeline itself.

## Features

### Vector RAG
- **Hybrid retrieval** — BM25 (Elasticsearch) + dense vectors in one query
- **DeepDoc parsing** — OCR + layout + table recognition for complex PDFs
- **Multi-format ingest** — PDF, DOCX, XLSX, PPTX, MD, HTML, JSON, TXT, source code
- **DashScope-powered** — Qwen LLM, text-embedding-v3, gte-rerank

### Graph RAG
- **Entity / relation extraction** — LLM (Qwen) with structured JSON output
- **Knowledge graph storage** — NetworkX + JSON persistence (swappable backend)
- **Community detection** — Louvain clustering
- **Community summarization** — LLM-generated topic summaries
- **Three retrieval modes** — local (entity-traversal), global (community summaries), hybrid (both + vector)

### CLI
- **Interactive REPL** — typer + rich + prompt_toolkit
- **Streaming answers** — token-by-token with reference citations
- **Single-file commands** — each functional area has its own module (easy to swap)

## Architecture

```
   ┌─────────────────────────────────────────────────────┐
   │  CLI (typer)                                        │
   │  ├─ rag index    → indexer [+ optional graph build] │
   │  ├─ rag ask      → retriever | graph.retriever      │
   │  │                  → generator                     │
   │  ├─ rag chat     → REPL (prompt_toolkit)            │
   │  ├─ rag retrieve → retriever (no LLM)               │
   │  ├─ rag kb …     → kb_manager                       │
   │  ├─ rag graph …  → graph.builder / store            │
   │  └─ rag doctor   → health checks                    │
   └────────────────────┬────────────────────────────────┘
                        │
   ┌────────────────────▼────────────────────────────────┐
   │  Core RAG (each file owns one concern)              │
   │  ├─ chunker.py     parse + split                    │
   │  ├─ embedder.py    DashScope text-embedding-v3      │
   │  ├─ indexer.py     parse → chunk → embed → ES       │
   │  ├─ retriever.py   BM25 + dense + rerank            │
   │  ├─ reranker.py    DashScope gte-rerank             │
   │  ├─ generator.py   Qwen streaming answer            │
   │  ├─ kb_manager.py  list/info/delete                 │
   │  └─ graph/                                          │
   │     ├─ types.py        Entity / Relation / Community│
   │     ├─ extractor.py    LLM-based extraction         │
   │     ├─ store.py        NetworkX + JSON adapter      │
   │     ├─ community.py    Louvain clustering           │
   │     ├─ summarizer.py   LLM community summaries      │
   │     ├─ builder.py      orchestrator                 │
   │     └─ retriever.py    local / global / hybrid      │
   └────────────────────┬────────────────────────────────┘
                        │
   ┌────────────────────▼────────────────────────────────┐
   │  Storage                                            │
   │  ├─ Elasticsearch        vectors + BM25 index       │
   │  └─ storage/graphs/*.json knowledge graph           │
   └─────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Prerequisites

- Python 3.10+ (tested on 3.11)
- Docker (for Elasticsearch)
- DashScope API key — get one at [bailian.console.aliyun.com](https://bailian.console.aliyun.com)

### 2. Setup

```bash
# Clone and enter the project
cd ragkit

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your DASHSCOPE_API_KEY

# Start Elasticsearch
docker compose up -d es

# Verify
rag doctor
```

### 3. Index a document

```bash
rag index ./my-report.pdf --kb finance
rag index ./docs/         --kb finance --recursive
```

### 4. Ask questions

```bash
# Single question
rag ask "What were the Q3 revenue drivers?" --kb finance

# Interactive REPL
rag chat --kb finance
> What were the Q3 revenue drivers?
[streaming answer with ##1$$ ##2$$ citations]
> /show 1
[full text of reference 1]
> /kb personal
[switches KB]
> /exit
```

### 5. Manage knowledge bases

```bash
rag kb list
rag kb info finance
rag kb delete finance --yes
```

## CLI reference

### Indexing
| Command | Purpose |
|---|---|
| `rag index <path> [--kb NAME] [--recursive]` | Parse, embed, index files into ES |
| `rag index <path> --build-graph` | Same, plus extract entities/relations into the knowledge graph |

### Asking questions
| Command | Purpose |
|---|---|
| `rag ask "Q" [--kb NAME] [--mode MODE]` | Single question. Mode: `vector` (default), `local`, `global`, `hybrid` |
| `rag chat [--kb NAME] [--top-k N] [--thinking]` | Interactive REPL |
| `rag retrieve "Q" [--kb NAME]` | Retrieval only (no LLM) — useful for tuning |

### Retrieval modes (for `ask`)
- **vector** — Original BM25 + dense vector. Fastest. Best for "find me the chunk that says X".
- **local** — Entity-neighborhood graph traversal. Best for "what is X" / "how does X relate to Y".
- **global** — Community summaries. Best for thematic ("what does this corpus discuss") questions.
- **hybrid** — vector + local graph + dedupe. Best general default once the graph is built.

### Knowledge-base management
| Command | Purpose |
|---|---|
| `rag kb list` | List KBs |
| `rag kb info NAME` | Show stats + document list |
| `rag kb delete NAME` | Drop the ES index |

### Knowledge-graph management
| Command | Purpose |
|---|---|
| `rag graph build --kb NAME` | Build a graph from an already-indexed KB (1 LLM call/chunk) |
| `rag graph info NAME` | Stats: entities, relations, communities, types |
| `rag graph show NAME ENTITY [--depth N]` | Inspect one entity and its neighborhood |
| `rag graph clear NAME` | Delete the graph file (vector index untouched) |

### Diagnostics
| Command | Purpose |
|---|---|
| `rag doctor` | Verify ES connection, API key, dictionaries |

REPL slash commands: `/kb`, `/top`, `/thinking`, `/show <i>`, `/clear`, `/help`, `/exit`.

## Configuration (.env)

| Variable | Default | Notes |
|---|---|---|
| `DASHSCOPE_API_KEY` | — | **Required** |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint |
| `RAG_LLM_MODEL` | `qwen-plus` | Any DashScope chat model |
| `RAG_EMBEDDING_MODEL` | `text-embedding-v3` | DashScope embedding model |
| `RAG_EMBEDDING_DIM` | `1024` | Must match the model |
| `ES_HOST` | `http://localhost:9200` | |
| `ES_USER` / `ES_PASSWORD` | `elastic` / `infini_rag_flow` | |
| `HF_ENDPOINT` | `https://hf-mirror.com` | For OCR model downloads in mainland China |
| `RAG_STORAGE_DIR` | `./storage` | Where parsed files / model cache go |

## Testing

```bash
# All tests (no external services needed — DashScope + ES are mocked)
pytest

# With coverage
pytest --cov=ragkit --cov-report=term-missing

# Only unit tests (fast)
pytest -m unit
```

The test suite covers (121 tests total):

**Vector pipeline**
- **config** — env-var precedence, missing-key contract
- **chunker** — supported-format dispatch, file-not-found, real Chinese TXT parsing
- **embedder** — batch-splitting at DashScope's 10-item cap, empty input
- **retriever** — query validation, ES→dataclass mapping, weight passthrough
- **indexer** — full pipeline + chunk-id determinism + cross-KB isolation + sparse embedding failure abort
- **kb_manager** — list/info/delete + aggregation shape
- **generator** — prompt format, content/thinking/done event stream, error handling
- **repl** — slash commands, immutability of state on errors
- **CLI** — command registration, exit codes, destructive-op confirmation

**Graph pipeline**
- **graph types** — Entity/Relation merge semantics (type union, weight accumulation, dedup)
- **graph store** — case-insensitive lookup, self-loop rejection, BFS depth, save/load roundtrip, double roundtrip, corrupt-file recovery
- **graph extractor** — JSON code-fence stripping, dangling-edge dropping, case-insensitive dedup, LLM-failure tolerance
- **graph community** — cluster separation, deterministic with seed, isolated-node bundling, misc-bucket flag, size-sorted IDs
- **graph retriever** — entity matching, BFS expansion, hybrid dedup via xxhash, vector-failure visibility, input validation
- **graph builder** — cross-chunk aggregation, abort-on-mass-failure, persistence, progress callbacks
- **summarizer** — data-loss guard (preserves communities beyond `max_communities`), per-community failure isolation

## Project layout

```
ragkit/
├── src/ragkit/
│   ├── cli/                CLI layer (typer + rich + prompt_toolkit)
│   │   ├── app.py
│   │   ├── commands.py
│   │   ├── repl.py
│   │   └── ui.py
│   ├── core/               RAG pipeline
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── reranker.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   ├── kb_manager.py
│   │   ├── deepdoc/        OCR + layout + table parsing (third-party)
│   │   ├── graph/          Graph RAG (each file = one swap point)
│   │   │   ├── types.py
│   │   │   ├── extractor.py   ⇆ swap entity-extraction model/prompt
│   │   │   ├── store.py       ⇆ swap graph backend (NetworkX → Neo4j)
│   │   │   ├── community.py   ⇆ swap clustering algorithm (Louvain → Leiden)
│   │   │   ├── summarizer.py  ⇆ swap summary model/prompt
│   │   │   ├── builder.py     orchestrator
│   │   │   └── retriever.py   local / global / hybrid
│   │   ├── rag/            tokenizer + search engine (third-party)
│   │   ├── api/utils/      project base-path helper
│   │   └── conf/           ES mapping
│   ├── config.py
│   └── logger.py
├── tests/                  121 behavior-focused tests
├── docker-compose.yml      Elasticsearch only
├── pyproject.toml
└── .env.example
```

### How to swap a component
Each `⇆`-marked file owns exactly one swap point — change there only:

| To swap... | Edit this | What changes |
|---|---|---|
| LLM provider | `generator.py:_client()` + `extractor.py:_llm_client()` + `summarizer.py:_client()` | Point `OpenAI(base_url=…)` at any OpenAI-compatible endpoint |
| Embedding model | `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_DIM` env vars | No code change |
| Vector backend | implement `DocStoreConnection`, change `retriever._get_dealer()` | Drop in Milvus/Qdrant/pgvector |
| Graph backend | implement `GraphStore`, change `store.open_store()` | Drop in Neo4j/Memgraph |
| Clustering algorithm | body of `community.detect_communities()` | Swap Louvain → Leiden, Girvan-Newman, etc. |
| Document parsing | `core/deepdoc/parser/*` | Replace per-format parser |

## Credits

The `deepdoc/` (parsers, OCR, layout/table recognition) and `rag/` (tokenizer, hybrid
search engine) directories are extracted and adapted from
[RAGFlow / InfiniFlow](https://github.com/infiniflow/ragflow), Apache 2.0.

## License

Apache-2.0
