from fastapi import FastAPI

app = FastAPI(
    title="RAG AI Agent",
    description="An AI assistant that answers questions using your documents.",
    version="0.1.0",
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "RAG AI Agent API is running",
    }