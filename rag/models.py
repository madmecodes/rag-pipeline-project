"""Lazily loaded local models and the LLM client. Loaded once per process."""
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from openai import OpenAI

from rag.config import get_settings


@lru_cache
def dense() -> TextEmbedding:
    return TextEmbedding(get_settings().dense_model)


@lru_cache
def sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding(get_settings().sparse_model)


@lru_cache
def reranker() -> TextCrossEncoder:
    return TextCrossEncoder(get_settings().rerank_model)


@lru_cache
def llm() -> OpenAI:
    s = get_settings()
    if not s.azure_foundry_endpoint or not s.azure_foundry_key:
        raise RuntimeError("AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_KEY must be set")
    return OpenAI(base_url=s.azure_foundry_endpoint, api_key=s.azure_foundry_key,
                  timeout=s.llm_timeout_s, max_retries=8)


def embed_query(text: str):
    """bge models expect an instruction prefix on queries, not on passages."""
    return list(dense().query_embed([text]))[0], list(sparse().query_embed([text]))[0]
