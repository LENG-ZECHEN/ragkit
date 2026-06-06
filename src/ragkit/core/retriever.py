"""Hybrid (BM25 + dense vector) retrieval with optional rerank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragkit import eval_context
from ragkit.core._ragflow.rag.nlp.search_v2 import Dealer
from ragkit.core._ragflow.rag.utils.es_conn import ESConnection


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk with metadata. Frozen so consumers can't mutate it."""

    rank: int
    document_id: str
    document_name: str
    content: str
    similarity: float
    vector_similarity: float
    term_similarity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.rank,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "content_with_weight": self.content,
            "similarity": self.similarity,
            "vector_similarity": self.vector_similarity,
            "term_similarity": self.term_similarity,
        }


_dealer: Dealer | None = None


def _get_dealer() -> Dealer:
    global _dealer
    if _dealer is None:
        _dealer = Dealer(dataStore=ESConnection())
    return _dealer


def retrieve(
    question: str,
    kb_name: str,
    *,
    top_k: int = 5,
    vector_similarity_weight: float = 0.6,
    similarity_threshold: float = 0.1,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval + rerank against one knowledge base.

    Args:
        question: User query, raw.
        kb_name: Knowledge base name (ES index).
        top_k: Number of chunks to return.
        vector_similarity_weight: Blend ratio for vector vs term similarity (0..1).
        similarity_threshold: Drop chunks below this score.
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")

    # Read overrides from the active eval context. In the no-override common
    # case this is a single dict.get() against an empty dict — effectively free.
    vsw = eval_context.get("vector_similarity_weight", vector_similarity_weight)
    st = eval_context.get("similarity_threshold", similarity_threshold)

    # Observability — no-ops unless observe.enable_debug() was called.
    from ragkit.cli import observe

    dealer = _get_dealer()
    observe.trace_query_rewriting(question, dealer.qryr)

    with observe.timed("vector pipeline (ES + rerank)"):
        raw = dealer.retrieval(
            question=question,
            embd_mdl=None,
            tenant_ids=kb_name,
            kb_ids=None,
            page=1,
            page_size=top_k,
            similarity_threshold=st,
            vector_similarity_weight=vsw,
        )

    out: list[RetrievedChunk] = []
    for i, chunk in enumerate(raw.get("chunks", []), start=1):
        docnm = chunk.get("docnm_kwd", "") or ""
        # Original code stored full paths in docnm — display only the basename.
        docnm = docnm.split("/")[-1]
        out.append(
            RetrievedChunk(
                rank=i,
                document_id=chunk.get("doc_id", ""),
                document_name=docnm,
                content=chunk.get("content_with_weight", ""),
                similarity=float(chunk.get("similarity", 0.0)),
                vector_similarity=float(chunk.get("vector_similarity", 0.0)),
                term_similarity=float(chunk.get("term_similarity", 0.0)),
            )
        )

    observe.trace_vector_retrieval_summary(int(raw.get("total", 0)), len(out))
    observe.trace_final_topk_scores(out)
    return out
