"""Online retrieval: permission-filtered hybrid search (BM25 + dense, fused with RRF),
then cross-encoder reranking, then parent-document expansion."""
from dataclasses import dataclass

from qdrant_client import models

from rag import store
from rag.config import get_settings
from rag.models import embed_query, reranker


@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    title: str
    text: str  # parent document text given to the LLM
    queue: str
    score: float  # rerank score (or fusion score when reranking is off)


def acl_filter(groups: list[str], filters: dict | None = None) -> models.Filter:
    """Enforced inside the vector search, so unauthorized chunks are never even candidates."""
    must = [models.FieldCondition(key="acl", match=models.MatchAny(any=groups or ["everyone"]))]
    for key, val in (filters or {}).items():
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
    return models.Filter(must=must)


def search(query: str, groups: list[str], mode: str = "hybrid", rerank: bool = True,
           filters: dict | None = None, k: int | None = None) -> list[Hit]:
    s = get_settings()
    k = k or s.context_k
    dv, sv = embed_query(query)
    flt = acl_filter(groups, filters)
    sparse_q = models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
    limit = s.rerank_k if rerank else k

    if mode == "dense":
        res = store.client().query_points(s.collection, query=dv.tolist(), using=store.DENSE,
                                          query_filter=flt, limit=limit, with_payload=True)
    elif mode == "sparse":
        res = store.client().query_points(s.collection, query=sparse_q, using=store.SPARSE,
                                          query_filter=flt, limit=limit, with_payload=True)
    else:
        res = store.client().query_points(
            s.collection,
            prefetch=[
                models.Prefetch(query=dv.tolist(), using=store.DENSE, filter=flt, limit=s.candidates_k),
                models.Prefetch(query=sparse_q, using=store.SPARSE, filter=flt, limit=s.candidates_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True)

    points = res.points
    if rerank and points:
        scores = list(reranker().rerank(query, [p.payload["chunk_text"] for p in points]))
        ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
    else:
        ranked = [(p, p.score) for p in points]

    # several chunks of one ticket -> keep the best, return the whole parent document
    hits, seen = [], set()
    for p, score in ranked:
        if p.payload["doc_id"] in seen:
            continue
        seen.add(p.payload["doc_id"])
        hits.append(Hit(p.payload["doc_id"], str(p.id), p.payload["title"],
                        p.payload["parent_text"], p.payload["queue"], float(score)))
        if len(hits) == k:
            break
    return hits
