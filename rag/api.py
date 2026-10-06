"""HTTP service.

Identity: in production the caller's groups come from a verified SSO token (Okta / Entra).
Here they come from the X-User-Groups header to keep the demo self-contained.
"""
import threading

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, Field

from rag import pipeline, store
from rag.cache import cache
from rag.config import get_settings
from rag.guard import GuardrailViolation
from rag.models import dense, reranker, sparse

app = FastAPI(title="atomic-work-rag")
_ingest_lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    history: list[Message] = []
    filters: dict[str, str] | None = None


class Feedback(BaseModel):
    trace_id: str
    helpful: bool
    comment: str = ""


def groups_from(header: str | None) -> list[str]:
    return sorted({g.strip() for g in (header or "").split(",") if g.strip()} | {"everyone"})


@app.on_event("startup")
def warm() -> None:
    # load models once so the first request isn't slow
    dense(), sparse(), reranker()


@app.get("/healthz")
def healthz():
    s = get_settings()
    ok = store.client().collection_exists(s.collection)
    count = store.client().count(s.collection).count if ok else 0
    if not ok:
        raise HTTPException(503, "index not built")
    return {"status": "ok", "chunks": count}


@app.post("/ask")
def ask(req: AskRequest, x_user_groups: str | None = Header(default=None)):
    try:
        return pipeline.ask(req.query, groups_from(x_user_groups),
                            [m.model_dump() for m in req.history], req.filters)
    except GuardrailViolation as e:
        raise HTTPException(400, f"guardrail: {e}")


@app.post("/feedback")
def feedback(fb: Feedback):
    pipeline.OUTCOMES.labels("feedback_up" if fb.helpful else "feedback_down").inc()
    with open("logs/feedback.jsonl", "a") as f:
        f.write(fb.model_dump_json() + "\n")
    return {"ok": True}


@app.post("/admin/reindex")
def reindex(bg: BackgroundTasks):
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "reindex already running")

    def run():
        from rag.ingest import ingest
        try:
            ingest()
            cache.clear()  # cached answers may cite changed documents
        finally:
            _ingest_lock.release()

    bg.add_task(run)
    return {"status": "started"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
