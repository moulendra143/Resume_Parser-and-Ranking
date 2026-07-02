"""Tests for the similarity scoring modules."""

import pytest
from src.similarity.tfidf_matcher import compute_tfidf_similarity


class TestTfidfMatcher:
    """Tests for TF-IDF cosine similarity."""

    def test_identical_texts_high_similarity(self):
        text = "python machine learning data science"
        score = compute_tfidf_similarity(text, text)
        assert score > 0.99

    def test_different_texts_lower_similarity(self):
        t1 = "python machine learning tensorflow keras"
        t2 = "cooking recipes italian food preparation"
        score = compute_tfidf_similarity(t1, t2)
        assert score < 0.3

    def test_empty_text_returns_zero(self):
        assert compute_tfidf_similarity("", "some text") == 0.0
        assert compute_tfidf_similarity("some text", "") == 0.0
        assert compute_tfidf_similarity("", "") == 0.0

    def test_partial_overlap(self):
        t1 = "python java sql machine learning"
        t2 = "python sql data analysis machine learning"
        score = compute_tfidf_similarity(t1, t2)
        assert 0.3 < score < 1.0
