import asyncio
import logging

from pathlib import PurePosixPath
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.ingestion.registry import load_file

from app.ingestion.schemas import (
    FileIngestionBatchResponse,
    FileIngestionResult,
)

from app.ingestion.storage import (
    save_loaded_document,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ingestion",
    tags=["Ingestion"],
)

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

MAX_FILES_PER_REQUEST = 20


def normalize_filename(filename: str | None) -> str:
    """
    Normalize an upload name without trusting it as a
    filesystem path.

    Relative folder names are preserved as metadata.
    """
    raw_name = (
        filename or "uploaded_file"
    ).replace("\\", "/")

    parts = [
        part
        for part in PurePosixPath(raw_name).parts
        if part not in {"", ".", "..", "/"}
    ]

    # Remove a Windows drive prefix, if present.
    if parts and parts[0].endswith(":"):
        parts = parts[1:]

    normalized = "/".join(parts)

    return normalized or "uploaded_file"


@router.post(
    "/files",
    response_model=FileIngestionBatchResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_files(
    files: Annotated[
        list[UploadFile],
        File(description="Files to ingest"),
    ],
) -> FileIngestionBatchResponse:
    """
    Load multiple local files through their dedicated readers.

    Supported:
        PDF, DOCX, PPTX, CSV, XLSX, XLSM,
        HTML, HTM, TXT and Markdown.
    """
    if len(files) > MAX_FILES_PER_REQUEST:
        for uploaded_file in files:
            await uploaded_file.close()

        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Upload up to {MAX_FILES_PER_REQUEST} "
                "files per request."
            ),
        )

    results: list[FileIngestionResult] = []

    for uploaded_file in files:
        filename = normalize_filename(
            uploaded_file.filename
        )

        try:
            content = await uploaded_file.read(
                MAX_FILE_SIZE_BYTES + 1
            )

            if len(content) > MAX_FILE_SIZE_BYTES:
                results.append(
                    FileIngestionResult(
                        filename=filename,
                        status="failed",
                        error="File exceeds the 15 MB limit.",
                    )
                )
                continue

            try:
                document = await asyncio.to_thread(
                    load_file,
                    filename,
                    content,
                )

                document_id = await asyncio.to_thread(
                    save_loaded_document,
                    filename=filename,
                    original_content=content,
                    document=document,
                )

                from app.rag.indexing import index_document

                await asyncio.to_thread(
                    index_document,
                    document_id,
                    document,
                )

            except ValueError as exc:
                results.append(
                    FileIngestionResult(
                        filename=filename,
                        status="failed",
                        error=str(exc),
                    )
                )
                continue

            except OSError:
                logger.exception(
                    "Storage failed for uploaded file %s",
                    filename,
                )

                results.append(
                    FileIngestionResult(
                        filename=filename,
                        status="failed",
                        error="Unable to save the loaded document.",
                    )
                )
                continue

            except Exception:
                logger.exception(
                    "Unexpected ingestion error for %s",
                    filename,
                )

                results.append(
                    FileIngestionResult(
                        filename=filename,
                        status="failed",
                        error="An unexpected ingestion error occurred.",
                    )
                )
                continue

            results.append(
                FileIngestionResult(
                    document_id=document_id,
                    filename=filename,
                    status="loaded",
                    loader=document.metadata.get(
                        "loader",
                        "unknown",
                    ),
                    characters_extracted=len(
                        document.content
                    ),
                    preview=document.content[:250],
                )
            )

        finally:
            await uploaded_file.close()

    loaded_count = sum(
        result.status == "loaded"
        for result in results
    )

    failed_count = len(results) - loaded_count

    return FileIngestionBatchResponse(
        total_files=len(results),
        loaded_files=loaded_count,
        failed_files=failed_count,
        results=results,
    )