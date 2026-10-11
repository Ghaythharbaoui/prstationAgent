from pathlib import Path

from app.ingestion.loaders.docling_loader import (
    load_with_docling,
)

from app.ingestion.loaders.html_loader import (
    load_html_file,
)

from app.ingestion.loaders.spreadsheet_loader import (
    load_spreadsheet,
)

from app.ingestion.loaders.text_loader import (
    load_text_file,
)

from app.ingestion.schemas import LoadedDocument


DOCLING_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
}

SPREADSHEET_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xlsm",
}

HTML_EXTENSIONS = {
    ".html",
    ".htm",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
}


SUPPORTED_EXTENSIONS = (
    DOCLING_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
    | HTML_EXTENSIONS
    | TEXT_EXTENSIONS
)


def load_file(
    filename: str,
    content: bytes,
) -> LoadedDocument:
    """
    Route an uploaded file to its dedicated reader.
    """
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {extension or 'none'}."
        )

    if not content:
        raise ValueError(
            "The uploaded file is empty."
        )

    if extension in DOCLING_EXTENSIONS:
        return load_with_docling(
            filename,
            content,
        )

    if extension in SPREADSHEET_EXTENSIONS:
        return load_spreadsheet(
            filename,
            content,
        )

    if extension in HTML_EXTENSIONS:
        return load_html_file(
            filename,
            content,
        )

    if extension in TEXT_EXTENSIONS:
        return load_text_file(
            filename,
            content,
        )

    raise ValueError(
        "No loader is registered for this file type."
    )