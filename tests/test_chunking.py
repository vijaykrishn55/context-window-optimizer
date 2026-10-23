import pytest
from context_optimizer.chunking import sliding_window_chunks
from context_optimizer.tokenizer import ApproximateCodec


def test_empty_document_has_no_chunks():
    assert sliding_window_chunks("", tokenizer=ApproximateCodec()) == []


def test_chunks_overlap_and_cover_content():
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = sliding_window_chunks(text, chunk_tokens=30, overlap_ratio=0.2, tokenizer=ApproximateCodec())
    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "abc" in chunks[0]
    assert text[-10:] in chunks[-1]
    assert any(chunks[i][-6:] == chunks[i + 1][:6] for i in range(len(chunks) - 1))


def test_overlap_below_assessment_minimum_rejected():
    with pytest.raises(ValueError, match="at least"):
        sliding_window_chunks("text", chunk_tokens=10, overlap_ratio=0.1, tokenizer=ApproximateCodec())
