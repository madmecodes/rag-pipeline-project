"""Guardrails: PII masking, prompt-injection screening (user input and retrieved text),
and output validation (citations must point at real sources)."""
import re

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<card>"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "<phone>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
]

INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+"
    r"(instructions|prompts|rules)|you are now|system prompt|reveal your|"
    r"act as (an?|the) |developer mode|jailbreak", re.I)

MAX_QUERY_CHARS = 2000


class GuardrailViolation(ValueError):
    pass


def mask_pii(text: str) -> str:
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def check_input(query: str) -> str:
    q = query.strip()
    if not q:
        raise GuardrailViolation("empty query")
    if len(q) > MAX_QUERY_CHARS:
        raise GuardrailViolation(f"query longer than {MAX_QUERY_CHARS} characters")
    if INJECTION.search(q):
        raise GuardrailViolation("query looks like a prompt-injection attempt")
    return mask_pii(q)


def sanitize_context(text: str) -> str:
    """Indirect injection: retrieved documents are data, so drop instruction-like lines."""
    return "\n".join(l for l in text.splitlines() if not INJECTION.search(l))


CITE = re.compile(r"\[(\d+)\]")


def check_output(answer: str, n_sources: int) -> tuple[list[int], bool]:
    """Return cited source numbers and whether the answer is grounded (cites only real sources)."""
    cited = sorted({int(m) for m in CITE.findall(answer)})
    grounded = bool(cited) and all(1 <= c <= n_sources for c in cited)
    return cited, grounded
