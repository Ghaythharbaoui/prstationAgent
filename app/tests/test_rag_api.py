from fastapi.testclient import TestClient

from app.api import rag
from app.main import app


client = TestClient(app)


def test_query_endpoint_returns_ranked_chunks(monkeypatch):
    def fake_search(query: str, limit: int):
        assert query == "leave policy"
        assert limit == 2
        return [
            {
                "document_id": "doc-1",
                "chunk_index": 0,
                "text": "Leave requests require manager approval.",
                "score": 0.91,
                "metadata": {"title": "Policy"},
            }
        ]

    monkeypatch.setattr(rag, "search_documents", fake_search)

    response = client.post(
        "/rag/query",
        json={"query": "leave policy", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["document_id"] == "doc-1"


def test_query_endpoint_rejects_empty_query():
    response = client.post(
        "/rag/query",
        json={"query": ""},
    )

    assert response.status_code == 422
