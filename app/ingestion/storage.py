import json

from pathlib import Path
from uuid import uuid4

from app.ingestion.schemas import LoadedDocument


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCES_DIR = PROJECT_ROOT / "data" / "sources"

LOADED_DOCUMENTS_DIR = (
    PROJECT_ROOT / "data" / "loaded_documents"
)


def save_loaded_document(
    filename: str,
    original_content: bytes,
    document: LoadedDocument,
) -> str:
    """
    Store the original file and normalized document.

    Returns:
        Generated document ID.
    """
    document_id = uuid4().hex

    extension = Path(filename).suffix.lower()

    stored_source_name = (
        f"{document_id}{extension}"
    )

    source_path = (
        SOURCES_DIR / stored_source_name
    )

    normalized_path = (
        LOADED_DOCUMENTS_DIR
        / f"{document_id}.json"
    )

    SOURCES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOADED_DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "document_id": document_id,
        "original_filename": filename,
        "stored_source_filename": stored_source_name,
        "loaded_document": document.model_dump(),
    }

    try:
        source_path.write_bytes(
            original_content
        )

        normalized_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    except OSError:
        source_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)
        raise

    return document_id