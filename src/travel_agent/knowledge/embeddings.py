"""Embedding backends.

The default is a hashing embedder with no heavy dependencies. Across 15 short
documents covering 3 cities, lexical overlap carries enough signal, and the
router checks city metadata as well. It is also deterministic and works
offline, which keeps retrieval reproducible in tests.

The alternative, all-MiniLM-L6-v2, drags in torch and costs roughly two
minutes on a cold import - too slow to sit in front of a Streamlit start.
Set EMBEDDING_BACKEND=minilm when semantic recall is worth that.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class HashingEmbedder(Embedder):
    """Hashed bag of words + character trigrams, L2-normalised.

    Character trigrams give partial robustness to spelling variants
    ("Kyoto"/"kyto"), which a pure word model would miss entirely.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall((text or "").lower())

        for token in tokens:
            self._add(vector, f"w:{token}", 1.0)
            for i in range(len(token) - 2):
                self._add(vector, f"c:{token[i : i + 3]}", 0.35)
        for a, b in zip(tokens, tokens[1:]):
            self._add(vector, f"b:{a}_{b}", 0.6)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def _add(self, vector: list[float], key: str, weight: float) -> None:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0  # signed hashing limits collisions
        vector[index] += sign * weight


class MiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 via sentence-transformers, loaded from local cache."""

    name = "minilm"
    dim = 384

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=2)
def get_embedder(backend: str = "hashing") -> Embedder:
    if backend == "minilm":
        try:
            return MiniLMEmbedder()
        except Exception:  # noqa: BLE001 - missing cache or torch: degrade quietly
            return HashingEmbedder()
    return HashingEmbedder()
