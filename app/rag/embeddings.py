from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(
    model_name: str,
    device: str | None,
) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


class SentenceTransformerEmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = _load_model(model_name, device)

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        values = list(texts)
        if not values:
            return []

        embeddings = self.model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return [list(map(float, vector)) for vector in embeddings]


def build_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> SentenceTransformerEmbeddingService:
    return SentenceTransformerEmbeddingService(
        model_name=model_name,
        device=device,
    )
