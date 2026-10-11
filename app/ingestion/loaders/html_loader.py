from bs4 import BeautifulSoup

from app.ingestion.schemas import LoadedDocument


def load_html_file(
    filename: str,
    content: bytes,
) -> LoadedDocument:
    try:
        html = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "HTML files must use UTF-8 encoding."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(" ", strip=True)

    if not text:
        raise ValueError(
            "The file contains no readable text."
        )

    title = soup.title.get_text(strip=True) if soup.title else ""

    return LoadedDocument(
        title=title or filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "Untitled",
        content=text,
        source_type="file",
        source_uri=filename,
        content_format="text",
        metadata={
            "loader": "beautifulsoup",
        },
    )