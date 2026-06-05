"""Central configuration for ragkit.

All settings come from environment variables (loaded via python-dotenv).
Loading this module has no side effects beyond reading `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Frozen snapshot of runtime configuration."""

    # DashScope (Alibaba Bailian) — used for LLM, Embedding, and Rerank
    dashscope_api_key: str
    dashscope_base_url: str

    # Model identifiers
    llm_model: str
    embedding_model: str
    embedding_dim: int

    # Per-request timeout for all OpenAI-compatible client calls (seconds).
    # Without this the SDK default is ~10min — bad for stuck upstreams.
    llm_timeout: int

    # Elasticsearch
    es_host: str
    es_user: str
    es_password: str

    # Storage
    storage_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            dashscope_base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            llm_model=os.getenv("RAG_LLM_MODEL", "qwen-plus"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v3"),
            embedding_dim=int(os.getenv("RAG_EMBEDDING_DIM", "1024")),
            llm_timeout=int(os.getenv("RAG_LLM_TIMEOUT", "60")),
            es_host=os.getenv("ES_HOST", "http://localhost:9200"),
            es_user=os.getenv("ES_USER", "elastic"),
            es_password=os.getenv("ES_PASSWORD", "infini_rag_flow"),
            storage_dir=Path(os.getenv("RAG_STORAGE_DIR", "./storage")).resolve(),
        )

    def require_api_key(self) -> None:
        if not self.dashscope_api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. Copy .env.example to .env "
                "and fill in your key, or export it in your shell."
            )


def get_config() -> Config:
    """Return the live configuration (re-reads env every call for testability)."""
    return Config.from_env()
