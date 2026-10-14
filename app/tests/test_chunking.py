from app.rag.chunking import chunk_document, chunk_text


def test_chunk_text_returns_multiple_segments_for_long_content():
    text = "This is sentence one. " * 200

    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_document_accepts_object_with_content():
    class SampleDoc:
        content = "alpha beta gamma delta epsilon " * 50

    chunks = chunk_document(SampleDoc(), chunk_size=120, chunk_overlap=20)

    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_chunk_document_rejects_invalid_input():
    try:
        chunk_document(123)
        assert False, "Expected TypeError for invalid document input"
    except TypeError:
        pass
