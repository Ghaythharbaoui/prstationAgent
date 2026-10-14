import asyncio
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.indexing import search_documents


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RagQueryResult(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]


class RagQueryResponse(BaseModel):
    query: str
    results: list[RagQueryResult]


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(request: RagQueryRequest) -> RagQueryResponse:
    results = await asyncio.to_thread(
        search_documents,
        request.query,
        request.top_k,
    )
    return RagQueryResponse(
        query=request.query,
        results=results,
    )
