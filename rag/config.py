from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (Azure AI Foundry, OpenAI-compatible v1 endpoint)
    azure_foundry_endpoint: str = ""
    azure_foundry_key: str = ""
    llm_model: str = "gpt-4.1"
    llm_fast_model: str = "gpt-4.1"  # query rewriting / judging
    llm_timeout_s: float = 30.0

    # Vector store: QDRANT_URL for a server, otherwise embedded on disk
    qdrant_url: str = ""
    qdrant_path: str = "data/qdrant"
    collection: str = "tickets"

    # Local models (no API cost, data stays on the box)
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Ingestion
    raw_csv: str = "data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    languages: str = "en"
    chunk_chars: int = 1200
    chunk_overlap: int = 200

    # Retrieval
    candidates_k: int = 40  # per retriever, before fusion
    rerank_k: int = 30  # fused candidates sent to the cross-encoder
    context_k: int = 5  # chunks given to the LLM
    min_rerank_score: float = -4.0  # below this we abstain and escalate

    # Cache
    cache_ttl_s: int = 3600
    cache_sim: float = 0.95

    trace_path: str = "logs/traces.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
