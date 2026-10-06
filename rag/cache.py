"""Semantic answer cache. Keyed by the caller's permission groups so one user's cached
answer can never leak documents to a user with fewer permissions."""
import threading
import time

import numpy as np

from rag.config import get_settings


class SemanticCache:
    def __init__(self, max_items: int = 1000):
        self.items: list[tuple[str, np.ndarray, float, dict]] = []
        self.max_items = max_items
        self.lock = threading.Lock()

    def get(self, groups: list[str], vec: np.ndarray) -> dict | None:
        s, key, now = get_settings(), ",".join(sorted(groups)), time.time()
        with self.lock:
            self.items = [it for it in self.items if now - it[2] < s.cache_ttl_s]
            best, best_sim = None, s.cache_sim
            for k, v, _, payload in self.items:
                if k == key:
                    sim = float(np.dot(v, vec) / (np.linalg.norm(v) * np.linalg.norm(vec)))
                    if sim >= best_sim:
                        best, best_sim = payload, sim
            return best

    def put(self, groups: list[str], vec: np.ndarray, payload: dict) -> None:
        with self.lock:
            self.items.append((",".join(sorted(groups)), vec, time.time(), payload))
            self.items = self.items[-self.max_items:]

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


cache = SemanticCache()
