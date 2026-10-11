from io import BytesIO

import pandas as pd
import pytest

from app.ingestion import registry
from app.ingestion.registry import load_file
from app.ingestion.schemas import LoadedDocument


def test_load_txt_file():
    document = load_file(
        "policy.txt",
        b"Employees must submit leave requests.",
    )

    assert document.title == "policy"
    assert document.source_type == "file"
    assert document.content == (
        "Employees must submit leave requests."
    )
    assert document.metadata["loader"] == "python-text"


def test_load_markdown_file():
    document = load_file(
        "policy.md",
        b"# Leave Policy\n\nSubmit requests early.",
    )

    assert document.content_format == "markdown"
    assert document.content.startswith("# Leave Policy")


def test_load_html_file_removes_scripts():
    html = b"""
    <html>
      <head>
        <title>Company Policy</title>
        <script>alert('not document content')</script>
      </head>
      <body>
        <h1>Leave Policy</h1>
        <p>Employees must submit requests.</p>
      </body>
    </html>
    """

    document = load_file(
        "policy.html",
        html,
    )

    assert document.title == "Company Policy"
    assert "Leave Policy" in document.content
    assert "Employees must submit requests." in document.content
    assert "alert(" not in document.content


def test_load_csv_file():
    csv_content = (
        b"Employee,Department\n"
        b"Alice,IT\n"
        b"Bob,HR\n"
    )

    document = load_file(
        "employees.csv",
        csv_content,
    )

    assert document.title == "employees"
    assert document.content_format == "markdown"
    assert "Employee" in document.content
    assert "Alice" in document.content
    assert document.metadata["row_count"] == 2
    assert document.metadata["loader"] == "pandas"


def test_load_excel_file():
    buffer = BytesIO()

    dataframe = pd.DataFrame(
        {
            "Employee": ["Alice", "Bob"],
            "Department": ["IT", "HR"],
        }
    )

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        dataframe.to_excel(
            writer,
            sheet_name="Employees",
            index=False,
        )

    document = load_file(
        "employees.xlsx",
        buffer.getvalue(),
    )

    assert "Worksheet: Employees" in document.content
    assert "Alice" in document.content
    assert "Department" in document.content
    assert document.metadata["loader"] == "pandas/openpyxl"


def test_unsupported_file_type_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported file extension",
    ):
        load_file(
            "archive.zip",
            b"some content",
        )


def test_pdf_routes_to_docling(monkeypatch):
    def fake_docling_loader(
        filename: str,
        content: bytes,
    ) -> LoadedDocument:
        return LoadedDocument(
            title="Example PDF",
            content="Extracted content",
            metadata={
                "loader": "docling",
            },
        )

    monkeypatch.setattr(
        registry,
        "load_with_docling",
        fake_docling_loader,
    )

    document = load_file(
        "example.pdf",
        b"mock pdf",
    )

    assert document.title == "Example PDF"
    assert document.metadata["loader"] == "docling"