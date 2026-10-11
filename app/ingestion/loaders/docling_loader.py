import os

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from pypdf import PdfReader

from app.ingestion.schemas import LoadedDocument


def _ocr_enabled() -> bool:
    return os.getenv(
        "RAG_PDF_OCR",
        "false",
    ).lower() in {"1", "true", "yes", "on"}


def _load_text_pdf(
    filename: str,
    content: bytes,
) -> LoadedDocument | None:
    reader = PdfReader(BytesIO(content))
    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]
    extracted_text = "\n\n".join(
        page.strip()
        for page in pages
        if page.strip()
    )

    if not extracted_text:
        return None

    return LoadedDocument(
        title=Path(filename).stem or "Untitled",
        content=extracted_text,
        source_type="file",
        source_uri=filename,
        content_format="text",
        metadata={
            "loader": "pypdf",
            "extension": ".pdf",
            "ocr_enabled": False,
            "page_count": len(reader.pages),
        },
    )


@lru_cache(maxsize=1)
def _get_converter():
    """
    Initialize the Docling converter once per process.
    """
    from docling.datamodel.base_models import InputFormat

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
    )

    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    pipeline_options = PdfPipelineOptions()

    ocr_enabled = _ocr_enabled()

    pipeline_options.do_ocr = ocr_enabled

    if ocr_enabled:
        ocr_language = os.getenv(
            "RAG_OCR_LANGUAGE",
            "iso:en",
        )

        pipeline_options.ocr_options = RapidOcrOptions(
            lang=[ocr_language]
        )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def load_with_docling(
    filename: str,
    content: bytes,
) -> LoadedDocument:
    extension = Path(filename).suffix.lower()

    if extension == ".pdf" and not _ocr_enabled():
        try:
            fast_document = _load_text_pdf(
                filename,
                content,
            )
        except Exception:
            fast_document = None

        if fast_document is not None:
            return fast_document

    temp_path: Path | None = None

    try:
        # Docling processes file paths. Preserve the original
        # extension so the correct input format is recognized.
        with NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        converter = _get_converter()

        result = converter.convert(
            str(temp_path)
        )

        extracted_text = (
            result.document
            .export_to_markdown()
            .strip()
        )

        if not extracted_text:
            raise ValueError(
                "No readable content was extracted."
            )

        return LoadedDocument(
            title=Path(filename).stem or "Untitled",
            content=extracted_text,
            source_type="file",
            source_uri=filename,
            content_format="markdown",
            metadata={
                "loader": "docling",
                "extension": extension,
                "ocr_enabled": (
                    extension == ".pdf" and _ocr_enabled()
                ),
            },
        )

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            f"Unable to process the {extension} document."
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)