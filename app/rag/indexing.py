from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.ingestion.schemas import LoadedDocument
from app.rag.chunking import chunk_document
from app.rag.embeddings import (
    SentenceTransformerEmbeddingService,
)
from app.rag.vector_store import JsonVectorStore


class RagIndexer:
    def __init__(
        self,
        embedding_service: SentenceTransformerEmbeddingService,
        vector_store: JsonVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_document(
        self,
        document_id: str,
        document: LoadedDocument,
    ) -> int:
        chunks = chunk_document(document)
        embeddings = self.embedding_service.embed_texts(chunks)

        records: list[dict[str, Any]] = []
        for position, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            records.append(
                {
                    "document_id": document_id,
                    "chunk_index": position,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "title": document.title,
                        "source_uri": document.source_uri,
                    },
                }
            )

        self.vector_store.add_records(records)
        return len(records)


@lru_cache(maxsize=1)
def get_default_indexer() -> RagIndexer:
    from app.rag.embeddings import (
        SentenceTransformerEmbeddingService,
    )

    return RagIndexer(
        embedding_service=SentenceTransformerEmbeddingService(),
        vector_store=JsonVectorStore(),
    )


def index_document(
    document_id: str,
    document: LoadedDocument,
) -> int:
    return get_default_indexer().index_document(
        document_id,
        document,
    )


def search_documents(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    indexer = get_default_indexer()
    query_embedding = indexer.embedding_service.embed_query(query)
    return indexer.vector_store.search(
        query_embedding,
        limit=limit,
    )
