from fastapi import FastAPI

from api.ingestion import router as ingestion_router


app = FastAPI(
    title="RAG AI Agent",
    description="An AI assistant that answers questions using your documents.",
    version="0.2.0",
)

app.include_router(ingestion_router)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "RAG AI Agent API is running",
    }