"""The full online request path, instrumented stage by stage."""
import json
import os
import time
import uuid
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

from rag import generate
from rag.cache import cache
from rag.config import get_settings
from rag.guard import check_input, check_output
from rag.models import embed_query
from rag.retrieve import search

STAGE_LATENCY = Histogram("rag_stage_seconds", "Latency per pipeline stage", ["stage"])
OUTCOMES = Counter("rag_answers_total", "Answer outcomes", ["outcome"])
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", ["kind"])

IDK = "I don't know."


class Trace:
    def __init__(self, query: str, groups: list[str]):
        self.data = {"trace_id": uuid.uuid4().hex[:12], "ts": time.time(), "query": query,
                     "groups": groups, "stages_ms": {}}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            STAGE_LATENCY.labels(name).observe(dt)
            self.data["stages_ms"][name] = round(dt * 1000, 1)

    def flush(self) -> None:
        path = get_settings().trace_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.data, default=str) + "\n")


def ask(query: str, groups: list[str], history: list[dict] | None = None,
        filters: dict | None = None, use_cache: bool = True) -> dict:
    s = get_settings()
    tr = Trace(query, groups)
    try:
        with tr.stage("guard_input"):
            q = check_input(query)
        with tr.stage("rewrite"):
            q = generate.rewrite(q, history or [])
        tr.data["rewritten"] = q

        if use_cache and not filters:
            with tr.stage("cache"):
                qvec = embed_query(q)[0]
                hit = cache.get(groups, qvec)
            if hit:
                OUTCOMES.labels("cache_hit").inc()
                tr.data["outcome"] = "cache_hit"
                return {**hit, "trace_id": tr.data["trace_id"], "cached": True}

        with tr.stage("retrieve"):
            hits = search(q, groups, filters=filters)
        tr.data["retrieved"] = [{"doc_id": h.doc_id, "score": round(h.score, 3)} for h in hits]

        # Abstain before spending LLM tokens when nothing relevant was found
        if not hits or hits[0].score < s.min_rerank_score:
            OUTCOMES.labels("escalated").inc()
            tr.data["outcome"] = "escalated_low_retrieval"
            return {"answer": IDK, "sources": [], "escalate": True, "grounded": False,
                    "trace_id": tr.data["trace_id"], "cached": False}

        with tr.stage("generate"):
            text, usage = generate.answer(q, hits)
        for k, v in usage.items():
            TOKENS.labels(k).inc(v)
        tr.data["usage"] = usage

        with tr.stage("guard_output"):
            cited, grounded = check_output(text, len(hits))
        abstained = text.strip().startswith("I don't know")
        escalate = abstained or not grounded
        outcome = "abstained" if abstained else ("ungrounded" if not grounded else "answered")
        OUTCOMES.labels(outcome).inc()
        tr.data["outcome"] = outcome

        result = {
            "answer": text,
            "sources": [{"n": i, "doc_id": h.doc_id, "title": h.title, "queue": h.queue,
                         "score": round(h.score, 3)}
                        for i, h in enumerate(hits, 1) if i in cited],
            "escalate": escalate, "grounded": grounded, "cached": False,
        }
        if use_cache and not filters and not escalate:
            cache.put(groups, qvec, result)
        return {**result, "trace_id": tr.data["trace_id"]}
    except Exception as e:
        tr.data["error"] = repr(e)
        raise
    finally:
        tr.flush()
