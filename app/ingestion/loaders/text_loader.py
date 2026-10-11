from pathlib import Path

from app.ingestion.schemas import LoadedDocument


def load_text_file(
    filename: str,
    content: bytes,
) -> LoadedDocument:
    extension = Path(filename).suffix.lower()

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Text files must use UTF-8 encoding."
        ) from exc

    text = text.strip()

    if not text:
        raise ValueError(
            "The file contains no readable text."
        )

    is_markdown = extension == ".md"

    return LoadedDocument(
        title=Path(filename).stem or "Untitled",
        content=text,
        source_type="file",
        source_uri=filename,
        content_format=(
            "markdown" if is_markdown else "text"
        ),
        metadata={
            "loader": "python-text",
            "extension": extension,
        },
    )