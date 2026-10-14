from pathlib import Path

from app.ingestion.schemas import LoadedDocument
from app.rag.indexing import RagIndexer
from app.rag.vector_store import JsonVectorStore


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(index + 1)]
            for index, text in enumerate(texts)
        ]


def test_index_document_persists_chunks(tmp_path: Path):
    store = JsonVectorStore(tmp_path / "index.json")
    indexer = RagIndexer(FakeEmbeddingService(), store)
    document = LoadedDocument(
        title="Policy",
        content="First paragraph.\n\nSecond paragraph.",
        source_uri="policy.txt",
    )

    count = indexer.index_document("doc-1", document)

    assert count >= 1
    assert (tmp_path / "index.json").exists()


def test_vector_store_returns_highest_similarity_first(tmp_path: Path):
    store = JsonVectorStore(tmp_path / "index.json")
    store.add_records([
        {
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": "similar",
            "embedding": [1.0, 0.0],
            "metadata": {},
        },
        {
            "document_id": "doc-2",
            "chunk_index": 0,
            "text": "different",
            "embedding": [0.0, 1.0],
            "metadata": {},
        },
    ])

    results = store.search([1.0, 0.0], limit=2)

    assert results[0]["text"] == "similar"
    assert results[0]["score"] > results[1]["score"]
