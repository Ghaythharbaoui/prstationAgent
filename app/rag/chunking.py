from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    if not text or not text.strip():
        return []

    splitter = build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def chunk_document(
    document: Any,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    if hasattr(document, "content"):
        return chunk_text(
            getattr(document, "content"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if isinstance(document, dict):
        content = document.get("content", "")
        return chunk_text(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if isinstance(document, str):
        return chunk_text(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise TypeError(
        "document must be a string, dict with a content field, or an object with a content attribute."
    )