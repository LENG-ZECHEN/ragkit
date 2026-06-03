# ragkit

A minimal **Retrieval-Augmented Generation (RAG)** toolkit with a CLI interface.

Extracted and refactored from a larger backend project — the heavy frontend, auth,
session-management, and HTTP layers are gone; what's left is the RAG pipeline itself.

## Features

- **Hybrid retrieval** — BM25 (Elasticsearch) + dense vectors in one query
- **DeepDoc parsing** — OCR + layout + table recognition for complex PDFs
- **Multi-format ingest** — PDF, DOCX, XLSX, PPTX, MD, HTML, JSON, TXT, source code
- **DashScope-powered** — Qwen LLM, text-embedding-v3, gte-rerank
- **Interactive CLI** — typer + rich + prompt_toolkit
- **Streaming answers** — token-by-token with reference citations

## Architecture

```
   ┌─────────────────────────────────────────────────────┐
   │  CLI  (typer)                                       │
   │  ├─ rag index    → indexer                          │
   │  ├─ rag ask      → retriever → generator            │
   │  ├─ rag chat     → REPL (prompt_toolkit)            │
   │  ├─ rag retrieve → retriever (no LLM)               │
   │  ├─ rag kb …     → kb_manager                       │
   │  └─ rag doctor   → health checks                    │
   └────────────────────┬────────────────────────────────┘
                        │
   ┌────────────────────▼────────────────────────────────┐
   │  Core RAG                                           │
   │  ├─ chunker.py    parse + split into chunks         │
   │  ├─ embedder.py   DashScope text-embedding-v3       │
   │  ├─ indexer.py    parse → chunk → embed → ES        │
   │  ├─ retriever.py  Dealer (BM25 + dense + rerank)    │
   │  ├─ reranker.py   DashScope gte-rerank              │
   │  ├─ generator.py  Qwen streaming answer             │
   │  └─ kb_manager.py list/info/delete                  │
   └────────────────────┬────────────────────────────────┘
                        │
   ┌────────────────────▼────────────────────────────────┐
   │  Storage                                            │
   │  └─ Elasticsearch  vectors + BM25 index             │
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

| Command | Purpose |
|---|---|
| `rag index <path> [--kb NAME] [--recursive]` | Parse, embed, index a file or directory |
| `rag ask "Q" [--kb NAME] [--top-k N] [--thinking] [--json]` | Ask a single question |
| `rag chat [--kb NAME] [--top-k N] [--thinking]` | Interactive REPL |
| `rag retrieve "Q" [--kb NAME] [--top-k N]` | Retrieval only (no LLM) — useful for tuning |
| `rag kb list \| info NAME \| delete NAME` | KB management |
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

The test suite covers:
- **config** — env-var precedence, missing-key contract
- **chunker** — supported-format dispatch, file-not-found, real Chinese TXT parsing
- **embedder** — batch-splitting at DashScope's 10-item cap, empty input
- **retriever** — query validation, ES→dataclass mapping, weight passthrough
- **indexer** — full pipeline + chunk-id determinism + cross-KB isolation
- **kb_manager** — list/info/delete + aggregation shape
- **generator** — prompt format, content/thinking/done event stream, error handling
- **repl** — slash commands, immutability of state on errors
- **CLI** — command registration, exit codes, destructive-op confirmation

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
│   │   ├── rag/            tokenizer + search engine (third-party)
│   │   ├── api/utils/      project base-path helper
│   │   └── conf/           ES mapping
│   ├── config.py
│   └── logger.py
├── tests/
├── docker-compose.yml      Elasticsearch only
├── pyproject.toml
└── .env.example
```

## Credits

The `deepdoc/` (parsers, OCR, layout/table recognition) and `rag/` (tokenizer, hybrid
search engine) directories are extracted and adapted from
[RAGFlow / InfiniFlow](https://github.com/infiniflow/ragflow), Apache 2.0.

## License

Apache-2.0
