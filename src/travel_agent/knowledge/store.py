"""ChromaDB-backed city knowledge store.

We drive Chroma with explicit vectors rather than an EmbeddingFunction so the
embedding backend stays swappable and no model download is ever attempted at
query time.

``lookup`` returns both the retrieved text and the evidence the router needs to
decide between the vector-store path and the web-search path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .seed_data import CITY_DOCUMENTS, SEEDED_CITIES, normalise_city


@dataclass(slots=True)
class KnowledgeHit:
    doc_id: str
    city: str
    country: str
    section: str
    text: str
    similarity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "city": self.city,
            "section": self.section,
            "similarity": round(self.similarity, 4),
        }


@dataclass(slots=True)
class KnowledgeLookup:
    """Result of a retrieval attempt, including why it counted as hit or miss."""

    found: bool
    city: str = ""
    country: str = ""
    text: str = ""
    hits: list[KnowledgeHit] = field(default_factory=list)
    top_similarity: float = 0.0
    reason: str = ""


class CityKnowledgeStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder: Embedder = get_embedder(self.settings.embedding_backend)
        self._collection = self._open_collection()

    # ------------------------------------------------------------- plumbing
    def _open_collection(self) -> Any:
        import chromadb

        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        # Namespacing by backend prevents mixing vectors of different geometry
        # when someone flips EMBEDDING_BACKEND on an existing store.
        name = f"{self.settings.collection}-{self.embedder.name}"
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def seed(self, force: bool = False) -> int:
        """Populate the store. Idempotent unless ``force`` is set."""
        if force:
            try:
                self._collection.delete(ids=[d["id"] for d in CITY_DOCUMENTS])
            except Exception:  # pragma: no cover - empty collection
                pass
        elif self._collection.count() >= len(CITY_DOCUMENTS):
            return 0

        texts = [f"{d['city']}. {d['section']}. {d['text']}" for d in CITY_DOCUMENTS]
        self._collection.upsert(
            ids=[d["id"] for d in CITY_DOCUMENTS],
            documents=[d["text"] for d in CITY_DOCUMENTS],
            embeddings=self.embedder.encode(texts),
            metadatas=[
                {
                    "city": d["city"],
                    "city_key": normalise_city(d["city"]),
                    "country": d["country"],
                    "section": d["section"],
                }
                for d in CITY_DOCUMENTS
            ],
        )
        return len(CITY_DOCUMENTS)

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:  # pragma: no cover
            return 0

    def cities(self) -> list[str]:
        return list(SEEDED_CITIES)

    # -------------------------------------------------------------- queries
    def lookup(self, city: str, question: str = "", k: int = 4) -> KnowledgeLookup:
        """Retrieve for ``city``, reporting whether the store actually covers it.

        Two independent signals are used. The metadata filter is authoritative:
        if documents are tagged with this city the store owns the answer. Vector
        similarity is the fallback for fuzzy or misspelled input.
        """
        self.seed()
        city_key = normalise_city(city)
        query_text = f"{city}. {question}".strip()
        query_vector = self.embedder.encode_one(query_text or city)

        if city_key:
            filtered = self._query(query_vector, k=k, where={"city_key": city_key})
            if filtered:
                return KnowledgeLookup(
                    found=True,
                    city=filtered[0].city,
                    country=filtered[0].country,
                    text=self._stitch(filtered),
                    hits=filtered,
                    top_similarity=filtered[0].similarity,
                    reason=f"'{filtered[0].city}' is in the curated vector store",
                )

        # No exact city match: fall back to open similarity search.
        loose = self._query(query_vector, k=k)
        top = loose[0].similarity if loose else 0.0
        if loose and top >= self.settings.similarity_floor:
            return KnowledgeLookup(
                found=True,
                city=loose[0].city,
                country=loose[0].country,
                text=self._stitch(loose),
                hits=loose,
                top_similarity=top,
                reason=f"semantic match above floor ({top:.2f} >= {self.settings.similarity_floor:.2f})",
            )

        return KnowledgeLookup(
            found=False,
            city=city,
            hits=loose,
            top_similarity=top,
            reason=(
                f"no documents for '{city or 'unknown'}' and best similarity "
                f"{top:.2f} < floor {self.settings.similarity_floor:.2f}"
            ),
        )

    def _query(
        self, vector: list[float], k: int, where: dict[str, Any] | None = None
    ) -> list[KnowledgeHit]:
        try:
            raw = self._collection.query(
                query_embeddings=[vector],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001 - a broken store must not kill the graph
            return []

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        hits: list[KnowledgeHit] = []
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            hits.append(
                KnowledgeHit(
                    doc_id=doc_id,
                    city=str(meta.get("city", "")),
                    country=str(meta.get("country", "")),
                    section=str(meta.get("section", "")),
                    text=text or "",
                    # cosine space: distance = 1 - cosine similarity
                    similarity=max(0.0, 1.0 - float(dist)),
                )
            )
        return hits

    @staticmethod
    def _stitch(hits: list[KnowledgeHit]) -> str:
        return "\n\n".join(f"[{h.section}] {h.text}" for h in hits)


@lru_cache(maxsize=1)
def get_store() -> CityKnowledgeStore:
    store = CityKnowledgeStore()
    store.seed()
    return store
