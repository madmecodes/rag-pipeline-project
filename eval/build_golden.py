"""Build a golden set: for a random sample of indexed documents, an LLM writes the question an
employee would actually type (short, paraphrased, no copied phrases). The source document is
the ground-truth answer, so retrieval can be scored with recall@k / MRR.

In production you'd seed this from real user queries labeled by support agents.

    python -m eval.build_golden --n 150
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from rag.config import get_settings
from rag.ingest import load_documents
from rag.models import llm

PROMPT = """Here is a resolved IT support ticket.
Write the short question (one or two sentences) that an employee would type into a help chatbot
when they have this problem. Paraphrase: do not copy distinctive phrases or the subject line.
Return only the question.

Ticket:
{text}"""


def make_question(doc: dict) -> dict | None:
    try:
        r = llm().chat.completions.create(
            model=get_settings().llm_fast_model, temperature=0.7, max_tokens=120,
            messages=[{"role": "user", "content": PROMPT.format(text=doc["text"][:2500])}])
        q = r.choices[0].message.content.strip().strip('"')
        return {"question": q, "doc_id": doc["doc_id"], "queue": doc["queue"], "acl": doc["acl"]}
    except Exception as e:
        print("skip", doc["doc_id"], e)
        return None


def main(n: int, seed: int, out: str) -> None:
    docs = load_documents()
    random.Random(seed).shuffle(docs)
    with ThreadPoolExecutor(3) as ex:
        rows = [r for r in ex.map(make_question, docs[:n]) if r]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval/golden.jsonl")
    a = ap.parse_args()
    main(a.n, a.seed, a.out)
