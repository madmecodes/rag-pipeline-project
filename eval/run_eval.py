"""Offline evaluation, split by stage so failures can be localised.

1. Retrieval: recall@k and MRR for sparse / dense / hybrid / hybrid+rerank.
2. Security: an ACL leak test. A user without the right group must never get restricted docs.
3. Generation (LLM judge on a subset): faithfulness, answer relevance, abstain rate, grounded rate.

Exits non-zero if any metric falls below eval/thresholds.json (used as a CI gate).

    python -m eval.run_eval [--gen-n 30] [--skip-gen]
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from rag import pipeline
from rag.config import get_settings
from rag.generate import build_context
from rag.models import llm
from rag.retrieve import search

CONFIGS = {
    "bm25": dict(mode="sparse", rerank=False),
    "dense": dict(mode="dense", rerank=False),
    "hybrid_rrf": dict(mode="hybrid", rerank=False),
    "hybrid_rrf+rerank": dict(mode="hybrid", rerank=True),
}
KS = (1, 5, 10)


def retrieval_eval(golden: list[dict]) -> dict:
    out = {}
    for name, cfg in CONFIGS.items():
        ranks, lat = [], []
        for g in golden:
            t0 = time.perf_counter()
            hits = search(g["question"], g["acl"], k=max(KS), **cfg)
            lat.append(time.perf_counter() - t0)
            ids = [h.doc_id for h in hits]
            ranks.append(ids.index(g["doc_id"]) + 1 if g["doc_id"] in ids else None)
        res = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3) for k in KS}
        res["mrr"] = round(sum(1 / r for r in ranks if r) / len(ranks), 3)
        res["p50_ms"] = round(statistics.median(lat) * 1000)
        res["p95_ms"] = round(sorted(lat)[int(len(lat) * 0.95) - 1] * 1000)
        out[name] = res
        print(f"{name:20s} {res}", flush=True)
    return out


def acl_leak_test(golden: list[dict]) -> dict:
    """Ask restricted questions as an unprivileged user; restricted docs must never appear."""
    restricted = [g for g in golden if "everyone" not in g["acl"]]
    leaks = 0
    for g in restricted:
        for h in search(g["question"], ["everyone"], k=10):
            if h.doc_id == g["doc_id"] or h.queue in ("Human Resources",):
                leaks += 1
    return {"restricted_queries": len(restricted), "leaks": leaks}


JUDGE = """You grade a RAG answer. Reply with JSON only:
{{"faithful": 0 or 1, "relevant": 0 or 1}}
faithful = every claim in the answer is supported by the sources (an "I don't know" counts as faithful).
relevant = the answer addresses the question (an "I don't know" counts as not relevant).

Question: {q}

Sources:
{ctx}

Answer: {a}"""


def judge_one(g: dict) -> dict:
    r = pipeline.ask(g["question"], g["acl"], use_cache=False)
    hits = search(g["question"], g["acl"])
    v = llm().chat.completions.create(
        model=get_settings().llm_fast_model, temperature=0, max_tokens=50,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE.format(q=g["question"], ctx=build_context(hits)[:8000],
                                                           a=r["answer"])}])
    j = json.loads(v.choices[0].message.content)
    return {"faithful": j.get("faithful", 0), "relevant": j.get("relevant", 0),
            "abstained": r["answer"].startswith("I don't know"), "grounded": r["grounded"],
            "retrieved_gold": g["doc_id"] in [s["doc_id"] for s in r["sources"]]}


def generation_eval(golden: list[dict], n: int) -> dict:
    with ThreadPoolExecutor(2) as ex:
        rows = list(ex.map(judge_one, golden[:n]))
    m = lambda k: round(sum(r[k] for r in rows) / len(rows), 3)
    res = {"n": len(rows), "faithfulness": m("faithful"), "answer_relevance": m("relevant"),
           "abstain_rate": m("abstained"), "grounded_rate": m("grounded"),
           "cites_gold_doc": m("retrieved_gold")}
    print("generation", res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--gen-n", type=int, default=30)
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(a.golden)]
    report = {"n_questions": len(golden), "retrieval": retrieval_eval(golden),
              "acl": acl_leak_test(golden)}
    print("acl", report["acl"])
    if not a.skip_gen:
        report["generation"] = generation_eval(golden, a.gen_n)
    json.dump(report, open("eval/report.json", "w"), indent=2)

    th = json.load(open("eval/thresholds.json"))
    best = report["retrieval"]["hybrid_rrf+rerank"]
    failures = [f"{k} {best[k]} < {v}" for k, v in th["retrieval"].items() if best[k] < v]
    if report["acl"]["leaks"] > 0:
        failures.append(f"ACL leaks: {report['acl']['leaks']}")
    if "generation" in report:
        failures += [f"{k} {report['generation'][k]} < {v}" for k, v in th["generation"].items()
                     if report["generation"][k] < v]
    if failures:
        print("EVAL GATE FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("EVAL GATE PASSED")


if __name__ == "__main__":
    main()
