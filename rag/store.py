"""Qdrant collection with a dense (cosine) and a sparse (BM25 + IDF) named vector per chunk."""
from functools import lru_cache

from qdrant_client import QdrantClient, models

from rag.config import get_settings

DENSE, SPARSE = "dense", "bm25"


@lru_cache
def client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url) if s.qdrant_url else QdrantClient(path=s.qdrant_path)


def ensure_collection(dim: int) -> None:
    s, c = get_settings(), client()
    if c.collection_exists(s.collection):
        return
    c.create_collection(
        s.collection,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE,
                                                   on_disk=True)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        # int8 scalar quantization: ~4x less RAM for the HNSW graph, rescored on originals
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
    )
    # payload indexes make ACL / metadata filters cheap during HNSW traversal
    for field in ("doc_id", "acl", "queue", "type", "language"):
        c.create_payload_index(s.collection, field, models.PayloadSchemaType.KEYWORD)


def doc_hashes() -> dict[str, str]:
    """doc_id -> content_hash for everything currently indexed (used for incremental sync)."""
    s, c = get_settings(), client()
    if not c.collection_exists(s.collection):
        return {}
    out, offset = {}, None
    while True:
        points, offset = c.scroll(s.collection, limit=2048, offset=offset,
                                  with_payload=["doc_id", "content_hash"], with_vectors=False)
        for p in points:
            out[p.payload["doc_id"]] = p.payload["content_hash"]
        if offset is None:
            return out


def delete_docs(doc_ids: list[str]) -> None:
    if not doc_ids:
        return
    s = get_settings()
    client().delete(s.collection, models.FilterSelector(filter=models.Filter(must=[
        models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])))
