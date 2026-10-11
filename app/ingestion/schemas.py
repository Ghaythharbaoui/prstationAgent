from typing import Any, Literal

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """
    Standard representation of content extracted from a source.
    """

    title: str = Field(min_length=1)

    content: str = Field(min_length=1)

    source_type: Literal[
        "file",
        "web",
        "google_drive",
    ] = "file"

    source_uri: str | None = None

    content_format: Literal[
        "text",
        "markdown",
    ] = "text"

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class FileIngestionResult(BaseModel):
    document_id: str | None = None

    filename: str

    status: Literal[
        "loaded",
        "failed",
    ]

    loader: str | None = None

    characters_extracted: int = 0

    preview: str | None = None

    error: str | None = None


class FileIngestionBatchResponse(BaseModel):
    total_files: int

    loaded_files: int

    failed_files: int

    results: list[FileIngestionResult]