# Run from RecallIQ with: python -m pytest
from app import split_passages, summarize

def test_split_passages_returns_content():
    assert split_passages("one two three", 2) == ["one two", "three"]

def test_summarize_returns_text():
    text = "TF-IDF ranks text passages. Cosine similarity compares vectors."
    assert "TF-IDF" in summarize(text)
