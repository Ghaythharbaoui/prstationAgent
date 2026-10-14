from app.rag.embeddings import SentenceTransformerEmbeddingService


def test_embedding_service_returns_vectors_for_multiple_texts():
    service = SentenceTransformerEmbeddingService(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = ["hello world", "vector search demo"]
    vectors = service.embed_texts(texts)

    assert len(vectors) == 2
    assert all(len(vector) == service.dimension for vector in vectors)
    assert vectors[0] != vectors[1]


def test_embedding_service_handles_single_query():
    service = SentenceTransformerEmbeddingService(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vector = service.embed_query("retrieval augmented generation")

    assert len(vector) == service.dimension
