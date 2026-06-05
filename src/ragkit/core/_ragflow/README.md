# `_ragflow/` — Vendored RAGFlow Modules

This subpackage contains code **vendored (copied + adapted) from
[RAGFlow / InfiniFlow](https://github.com/infiniflow/ragflow)** under the
Apache-2.0 license. It is kept here, under an underscore-prefixed package
name, so the boundary between **ragkit's own work** and **third-party
vendored code** is obvious at a glance.

> **TL;DR**: everything inside `_ragflow/` is not original ragkit work. The
> license headers, algorithms, and overall structure follow RAGFlow upstream.
> ragkit has applied **targeted modifications** — security fixes, import-path
> rewrites, and a few ergonomic adjustments — listed below.

- **Upstream**: <https://github.com/infiniflow/ragflow>
- **License**: Apache-2.0 (© InfiniFlow)
- **Vendored on**: 2026-06-05 (this commit)
- **Size**: ~6,300 LOC Python + ~390 MB resource files (ONNX models, dictionaries)

---

## 1. What's vendored

```
_ragflow/
├── README.md                   ← (this file)
├── __init__.py
├── api/                        Minimal helper layer (~46 LOC)
│   ├── constants.py            8 unused constants kept for upstream API parity
│   └── utils/file_utils.py     get_project_base_directory() — resolves to .../core/_ragflow/
├── conf/
│   └── mapping.json            Elasticsearch dynamic-template + field mapping
├── deepdoc/                    Document parsing + vision pipeline (~3,860 LOC)
│   ├── parser/                 PDF / DOCX / XLSX / PPTX / HTML / JSON / MD / TXT
│   └── vision/                 OCR / Layout / Table-structure recognition
└── rag/                        Chinese-aware hybrid retrieval (~2,348 LOC)
    ├── app/naive.py            Naive multi-format chunker
    ├── nlp/                    rag_tokenizer (Trie) + search_v2 (Dealer) + query / term_weight / synonym
    ├── utils/                  ESConnection + DocStoreConnection protocol
    ├── settings.py
    └── res/                    Resource files (~390 MB)
        ├── huqie.txt           Chinese tokenizer dictionary (7.9 MB)
        ├── huqie.txt.trie      Pre-built datrie cache (52 MB, auto-rebuilt if missing)
        └── deepdoc/            ONNX + XGBoost model weights
```

## 2. Why these modules?

Each vendored subtree provides a specific capability ragkit relies on:

### `deepdoc/` — document parsing & computer vision

| File | Capability ragkit uses |
|---|---|
| `parser/pdf_parser.py` | PDF parsing with OCR + layout detection + table recognition. Exports `RAGFlowPdfParser` (full pipeline) and `PlainParser` (text-only fallback). |
| `parser/docx_parser.py` | Word `.docx` extraction (text + tables, header-joining heuristics). |
| `parser/excel_parser.py` | `.xlsx` parsing — per-row "header: value" strings + HTML chunked view. |
| `parser/ppt_parser.py` | `.pptx` slide extraction (text frames, grouped shapes, tables). |
| `parser/html_parser.py` | HTML body extraction via readability + html_text. |
| `parser/json_parser.py` | Size-bounded recursive JSON chunker. |
| `parser/markdown_parser.py` | Markdown → text + detected tables (bordered + borderless). |
| `parser/txt_parser.py` | Token-budgeted plain-text chunker. |
| `vision/ocr.py` | PaddleOCR-compatible text detection + recognition (ONNX Runtime, CPU/GPU). |
| `vision/recognizer.py` | Base ONNX-inference wrapper + geometric box sorting. |
| `vision/layout_recognizer.py` | Page-region classifier (text / title / figure / table / equation / header / footer). |
| `vision/table_structure_recognizer.py` | Table cell / row / column reconstruction. |
| `vision/operators.py` | Image preprocessing op zoo (Decode, Standardize, Normalize, Resize, Pad, NMS, …). |
| `vision/postprocess.py` | `DBPostProcess` (text-detection mask → polygons) + `CTCLabelDecode` (logits → strings). |

### `rag/` — Chinese-aware hybrid retrieval

| File | Capability ragkit uses |
|---|---|
| `nlp/rag_tokenizer.py` | Dual-direction (forward + reverse) Trie tokenizer ("huqie") for Chinese + English. Builds a `datrie` automaton from `res/huqie.txt`. |
| `nlp/search_v2.py` | The `Dealer` retrieval engine: composes BM25 + dense vector queries, fuses via weighted sum, applies rank features, reranks by token similarity × vector similarity. |
| `nlp/query.py` | `FulltextQueryer` — rewrites the user question into a field-weighted ES match query; expands synonyms, weights terms. |
| `nlp/term_weight.py` | IDF / NER term-weight scorer used by `query.py`. |
| `nlp/synonym.py` | Synonym lookup (`synonym.json`, optional Redis hot-reload, WordNet fallback). |
| `nlp/model.py` | **Thin proxy** to `ragkit.core.embedder` / `ragkit.core.reranker` — the seam where ragkit's own LLM layer plugs into the upstream search engine. See §3.2. |
| `nlp/__init__.py` | Helpers: `is_english`, `tokenize`, `naive_merge`, `tokenize_chunks`, `concat_img`, `find_codec`, etc. |
| `app/naive.py` | "Naive" chunker that dispatches by file type to `deepdoc/parser/*`, splits sections, and packs them into ES-ready chunk dicts. |
| `utils/es_conn.py` | `ESConnection` — the concrete `DocStoreConnection` implementation wrapping `elasticsearch-py`. |
| `utils/doc_store_conn.py` | `DocStoreConnection` protocol + `MatchTextExpr` / `MatchDenseExpr` / `FusionExpr` / `OrderByExpr` / `SparseVector` dataclasses. |
| `settings.py` | Module-level constants (`RAG_CONF_PATH`, `DOC_MAXIMUM_SIZE`, `PAGERANK_FLD`, `TAG_FLD`, …). |

### `api/` — minimal compatibility layer (~46 LOC)

| File | Capability |
|---|---|
| `api/utils/file_utils.py` | `get_project_base_directory()` — anchors at `core/_ragflow/` so all resource lookups (`rag/res/...`, `conf/...`) resolve correctly. |
| `api/constants.py` | 8 unused constants from upstream RAGFlow's `api` namespace; kept to minimize divergence. |

## 3. What ragkit changed vs upstream

These are the **only** non-trivial deltas from upstream. Everything else is byte-equivalent or differs only in import path.

### 3.1 Namespaced import rewrites (every file affected)

Upstream RAGFlow uses top-level absolute imports:

```python
from rag.nlp import rag_tokenizer
from api.utils.file_utils import get_project_base_directory
from deepdoc.parser import PdfParser
```

In ragkit these become:

```python
from ragkit.core._ragflow.rag.nlp import rag_tokenizer
from ragkit.core._ragflow.api.utils.file_utils import get_project_base_directory
from ragkit.core._ragflow.deepdoc.parser import PdfParser
```

**31 internal cross-references** were rewritten this way to keep the
vendored code working under ragkit's namespace.

### 3.2 `nlp/model.py` is rewritten as a proxy ⭐ Most significant behavior change

| | Upstream RAGFlow | ragkit |
|---|---|---|
| Lines | ~300+ | **~38** (thin shim) |
| Backend | `LLMBundle` / `TenantLLMService` (DB-driven multi-tenant) | Direct call to `ragkit.core.embedder.embed_one` / `embed_batch` and `ragkit.core.reranker.rerank_scores` |
| Config source | RAGFlow database tenant config | `.env` + `Config.from_env()` |
| Model selection | Per-tenant in DB | Globally via env var |

This is the single seam where the vendored search engine plugs into
ragkit's own LLM layer — DashScope (OpenAI-compatible) instead of RAGFlow's
service infrastructure.

### 3.3 Security fix in `rag/nlp/search_v2.py` (tagged `ISS-001`)

Lines ~266–275 — originally:

```python
for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
```

ragkit replaced this with `ast.literal_eval()` + `try/except` because
`TAG_FLD` is read from Elasticsearch documents — any attacker with write
access to the index could have achieved RCE through the original `eval()`.
Marked inline: `# SECURITY (ISS-001):`.

### 3.4 Security fix in `deepdoc/vision/operators.py` (tagged `ISS-002`)

Lines ~109–118 in `NormalizeImage.__init__` — originally:

```python
scale = eval(scale)  if isinstance(scale, str) else scale
```

Replaced with `ast.literal_eval()` + `try/except (ValueError, SyntaxError)`
for the same arbitrary-code-execution reason. Marked inline:
`# SECURITY (ISS-002):`.

### 3.5 Robustness fix in `rag/utils/es_conn.py` (tagged `ISS-021`)

Lines ~57–74 — the `conf/mapping.json` load now raises a clear
`RuntimeError` naming the path on `OSError` / `json.JSONDecodeError`,
plus a second `RuntimeError` if the parsed JSON is missing the
`mappings` key. Upstream propagated raw decoder errors.

### 3.6 Ergonomic / cosmetic adjustments

| Adjustment | Where | Why |
|---|---|---|
| Stable logger name `ragkit.es_conn` | `rag/utils/es_conn.py:41` | Lets users filter ES log noise via a known logger |
| `.env`-driven ES config (`load_dotenv` + `ES_HOST` / `ES_USER` / `ES_PASSWORD`) | `rag/utils/es_conn.py` | Matches ragkit's config pattern; upstream uses service-config file |
| beartype runtime type-checking removed | `deepdoc/__init__.py` | CLI startup latency |
| Chinese-language `ValueError` + file-header pre-checks | `deepdoc/parser/excel_parser.py` | Better error messages for non-zip inputs |
| `LIGHTEN` toggle block commented out | `deepdoc/parser/pdf_parser.py:46-52` | ragkit always runs the dense path; no settings facade |
| Cross-module imports inside `deepdoc/` use absolute `ragkit.core._ragflow.deepdoc.*` paths | `pdf_parser.py:31`, `txt_parser.py:19` | Consistent with the rest of the migration |

### 3.7 What was NOT changed

The following modules are functionally **identical to upstream** apart
from import-path rewrites (no behavior changes):

- `rag/nlp/rag_tokenizer.py`, `rag/nlp/query.py`, `rag/nlp/term_weight.py`, `rag/nlp/synonym.py`
- `rag/app/naive.py`, `rag/nlp/__init__.py`, `rag/utils/__init__.py`, `rag/utils/doc_store_conn.py`, `rag/settings.py`
- All `deepdoc/parser/*.py` files except `pdf_parser.py` (LIGHTEN comment-out) and `excel_parser.py` (Chinese errors)
- All `deepdoc/vision/*.py` files except `operators.py` (ISS-002)
- `api/constants.py`, `api/utils/file_utils.py`

## 4. Resource files

The vendored code expects this exact tree under `_ragflow/rag/res/`:

```
res/
├── huqie.txt              7.9 MB  Chinese tokenizer source dictionary
│                                  Format: word \t freq \t POS, one per line
│                                  Loaded by RagTokenizer.loadDict_.
├── huqie.txt.trie         52  MB  Pre-built datrie cache of huqie.txt
│                                  Auto-rebuilt if missing (~30s cold start)
└── deepdoc/                       Model weights (downloaded from HuggingFace)
    ├── det.onnx            4.5 MB OCR text detection
    ├── rec.onnx           10  MB  OCR text recognition
    ├── tsr.onnx           12  MB  Table structure recognition
    ├── layout.onnx        72  MB  Generic layout analysis
    ├── layout.laws.onnx   72  MB  Layout — legal documents
    ├── layout.manual.onnx 72  MB  Layout — manuals
    ├── layout.paper.onnx  72  MB  Layout — academic papers
    ├── updown_concat_xgb.model    5.6 MB  XGBoost up/down paragraph concat
    └── ocr.res             26 KB  OCR character set / label file
```

Plus `_ragflow/conf/mapping.json` (Elasticsearch dynamic-template + field
mapping, required by `es_conn.py`).

Models are downloaded automatically from HuggingFace on first PDF parse:
- `InfiniFlow/deepdoc` (OCR + Layout + TSR models)
- `InfiniFlow/text_concat_xgb_v1.0` (XGBoost model)

Override the HF endpoint with `HF_ENDPOINT=https://hf-mirror.com` if
behind a regional restriction (default behavior; can be unset in
unrestricted networks).

## 5. How ragkit's own code interacts with `_ragflow/`

The boundary is well-defined — only **13 import points** in first-party
ragkit code reach into `_ragflow/`:

| First-party caller | What it uses from `_ragflow/` |
|---|---|
| `core/chunker.py` | `_ragflow.rag.app.naive.chunk` |
| `core/indexer.py` | `_ragflow.rag.utils.es_conn.ESConnection` |
| `core/retriever.py` | `_ragflow.rag.nlp.search_v2.Dealer` + `_ragflow.rag.utils.es_conn.ESConnection` |
| `core/kb_manager.py` | `_ragflow.rag.utils.es_conn.ESConnection` |
| `core/graph/searcher.py` | `_ragflow.rag.utils.es_conn.ESConnection` |
| `core/graph/es_indexer.py` | `_ragflow.rag.utils.es_conn.ESConnection` |
| `cli/graph_cmd.py` | `_ragflow.rag.utils.es_conn.ESConnection` (lazy) |
| `cli/commands.py` | `_ragflow.rag.utils.es_conn.ESConnection` + `_ragflow.api.utils.file_utils.get_project_base_directory` (lazy) |
| `cli/observe.py` | `_ragflow.rag.nlp.rag_tokenizer` + `_ragflow.rag.nlp.query.FulltextQueryer` (lazy, for `--debug` traces) |

Everything else (the entire `graph/` package, all of `cli/`, and the
top-level coordination files `chunker.py`, `embedder.py`, `generator.py`,
`indexer.py`, `kb_manager.py`, `reranker.py`, `retriever.py`) is
ragkit-original.

## 6. License

The vendored code in this subpackage retains its upstream Apache-2.0
license (© InfiniFlow). All ragkit-specific modifications are also released
under Apache-2.0 (the ragkit project's own license). The original copyright
headers remain at the top of every file.
