"""Offline pipeline: load -> clean -> document -> chunk -> embed -> upsert, incrementally.

Each resolved ticket becomes one knowledge document (issue + resolution). Documents get a
stable doc_id and a content hash, so re-running only embeds new or changed documents and
deletes documents that disappeared from the source.

    python -m rag.ingest [--limit N] [--full]
"""
import argparse
import hashlib
import re
import time
import uuid

import pandas as pd
from qdrant_client import models

from rag import store
from rag.config import get_settings
from rag.guard import mask_pii
from rag.models import dense, sparse

# Access control: which employee groups may read knowledge from each queue.
# In production this is inherited from the source system's ACLs (SharePoint, ServiceNow...).
QUEUE_ACL = {
    "Human Resources": ["hr"],
    "Billing and Payments": ["finance", "support"],
    "Sales and Pre-Sales": ["sales", "support"],
}
DEFAULT_ACL = ["everyone"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_documents(limit: int | None = None) -> list[dict]:
    s = get_settings()
    df = pd.read_csv(s.raw_csv)
    df = df[df.language.isin(s.languages.split(","))].dropna(subset=["body", "answer"])
    docs = []
    for row in df.itertuples():
        subject, body, answer = clean(row.subject), clean(row.body), clean(row.answer)
        if len(body) < 30 or len(answer) < 30:
            continue
        doc_id = hashlib.sha1(f"{subject}\n{body}".encode()).hexdigest()[:16]
        text = f"Issue: {mask_pii(body)}\n\nResolution: {mask_pii(answer)}"
        tags = [t for t in (getattr(row, f"tag_{i}") for i in range(1, 9)) if isinstance(t, str)]
        docs.append({
            "doc_id": doc_id,
            "title": subject or body[:80],
            "text": text,
            "queue": row.queue,
            "type": row.type,
            "priority": row.priority,
            "language": row.language,
            "tags": tags,
            "acl": QUEUE_ACL.get(row.queue, DEFAULT_ACL),
            "content_hash": hashlib.sha1(text.encode()).hexdigest(),
        })
    # identical tickets collapse to one document
    docs = list({d["doc_id"]: d for d in docs}.values())
    return docs[:limit] if limit else docs


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Recursive, structure-aware split: paragraphs, then sentences, then hard cut."""
    if len(text) <= size:
        return [text]
    for sep in ("\n\n", "\n", ". ", " "):
        parts = text.split(sep)
        if len(parts) > 1:
            break
    else:
        return [text[i:i + size] for i in range(0, len(text), size - overlap)]
    chunks, cur = [], ""
    for p in parts:
        piece = p + sep
        if len(cur) + len(piece) > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + piece  # carry overlap so facts aren't cut in half
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return [c for part in chunks
            for c in (split_text(part, size, overlap) if len(part) > size * 1.5 else [part])]


def chunk(doc: dict) -> list[dict]:
    s = get_settings()
    # Contextual header (cheap version of Anthropic's contextual retrieval): every chunk
    # carries the document title and category so it stays meaningful on its own.
    header = f"[{doc['queue']} / {doc['type']}] {doc['title']}\n"
    return [{"chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['doc_id']}#{i}")),
             "chunk_index": i, "embed_text": header + piece, "chunk_text": piece}
            for i, piece in enumerate(split_text(doc["text"], s.chunk_chars, s.chunk_overlap))]


def ingest(limit: int | None = None, full: bool = False, batch: int = 256) -> dict:
    s = get_settings()
    t0 = time.time()
    docs = load_documents(limit)
    probe = next(iter(dense().embed(["probe"])))
    if full and store.client().collection_exists(s.collection):
        store.client().delete_collection(s.collection)
    store.ensure_collection(len(probe))

    existing = store.doc_hashes()
    current = {d["doc_id"] for d in docs}
    changed = [d for d in docs if existing.get(d["doc_id"]) != d["content_hash"]]
    removed = [i for i in existing if i not in current] if not limit else []

    # changed docs may now have fewer chunks: drop old chunks before re-inserting
    store.delete_docs([d["doc_id"] for d in changed if d["doc_id"] in existing] + removed)

    rows = [(d, c) for d in changed for c in chunk(d)]
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        texts = [c["embed_text"] for _, c in part]
        dvecs = list(dense().embed(texts))
        svecs = list(sparse().embed(texts))
        store.client().upsert(s.collection, points=[
            models.PointStruct(
                id=c["chunk_id"],
                vector={store.DENSE: dv.tolist(),
                        store.SPARSE: models.SparseVector(indices=sv.indices.tolist(),
                                                          values=sv.values.tolist())},
                payload={**{k: v for k, v in d.items() if k != "text"},
                         "chunk_index": c["chunk_index"], "chunk_text": c["chunk_text"],
                         "parent_text": d["text"]},
            ) for (d, c), dv, sv in zip(part, dvecs, svecs)])
        print(f"  upserted {min(i + batch, len(rows))}/{len(rows)} chunks", flush=True)

    stats = {"documents": len(docs), "changed": len(changed), "removed": len(removed),
             "unchanged": len(docs) - len(changed), "chunks_written": len(rows),
             "seconds": round(time.time() - t0, 1)}
    print(stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="re-embed everything")
    a = ap.parse_args()
    ingest(a.limit, a.full)
