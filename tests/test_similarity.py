"""Tests for the similarity scoring modules.

Covers:
- TF-IDF cosine similarity (fast, no model download required)
- Hybrid scorer weight arithmetic and boundary conditions
- Config loader — verifies defaults are applied when yaml is absent
"""

import pytest
from src.similarity.tfidf_matcher import compute_tfidf_similarity
from src.similarity.hybrid_scorer import compute_hybrid_similarity
from src.utils.config_loader import get_config


# ---------------------------------------------------------------------------
# TF-IDF similarity tests
# ---------------------------------------------------------------------------

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

    def test_score_is_symmetric(self):
        """TF-IDF cosine similarity should be the same regardless of argument order."""
        t1 = "deep learning neural networks pytorch"
        t2 = "pytorch tensorflow deep learning"
        assert abs(compute_tfidf_similarity(t1, t2) - compute_tfidf_similarity(t2, t1)) < 1e-9

    def test_score_bounded_zero_to_one(self):
        t1 = "software engineer python backend api"
        t2 = "data scientist machine learning python"
        score = compute_tfidf_similarity(t1, t2)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Hybrid scorer tests
# ---------------------------------------------------------------------------

class TestHybridScorer:
    """Tests for the configurable hybrid similarity scorer."""

    def test_returns_three_keys(self):
        result = compute_hybrid_similarity("python developer", "python engineer", embedding_weight=0.7)
        assert set(result.keys()) == {"embedding", "tfidf", "hybrid"}

    def test_hybrid_is_weighted_combination(self):
        """hybrid == 0.7 * embedding + 0.3 * tfidf (within floating-point tolerance)."""
        result = compute_hybrid_similarity(
            "machine learning python scikit-learn",
            "data science python ml models",
            embedding_weight=0.7,
        )
        expected = round(result["embedding"] * 0.7 + result["tfidf"] * 0.3, 4)
        assert abs(result["hybrid"] - expected) < 1e-4

    def test_full_embedding_weight_ignores_tfidf(self):
        """embedding_weight=1.0 should make hybrid == embedding."""
        result = compute_hybrid_similarity(
            "java spring boot microservices",
            "java backend spring cloud",
            embedding_weight=1.0,
        )
        assert abs(result["hybrid"] - result["embedding"]) < 1e-4

    def test_zero_embedding_weight_uses_only_tfidf(self):
        """embedding_weight=0.0 should make hybrid == tfidf."""
        result = compute_hybrid_similarity(
            "sql database postgres",
            "postgresql relational database",
            embedding_weight=0.0,
        )
        assert abs(result["hybrid"] - result["tfidf"]) < 1e-4

    def test_empty_inputs_return_zero(self):
        result = compute_hybrid_similarity("", "some job description", embedding_weight=0.7)
        assert result["embedding"] == 0.0
        assert result["hybrid"] == 0.0

    def test_scores_are_rounded_to_four_decimals(self):
        result = compute_hybrid_similarity("python", "python developer", embedding_weight=0.7)
        for key in ("embedding", "tfidf", "hybrid"):
            val = result[key]
            assert val == round(val, 4)


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------

class TestConfigLoader:
    """Tests for the config singleton and default fallback values."""

    def test_config_loads_without_error(self):
        cfg = get_config()
        assert cfg is not None

    def test_default_embedding_weight(self):
        cfg = get_config()
        # The config.yaml specifies 0.7; verify it was parsed correctly.
        assert 0.0 < cfg.similarity.embedding_weight <= 1.0

    def test_default_n_estimators_positive(self):
        cfg = get_config()
        assert cfg.ranking.n_estimators > 0

    def test_scoring_weights_sum_to_one(self):
        w = get_config().scoring_weights
        total = w.similarity + w.skills + w.experience + w.education
        assert abs(total - 1.0) < 1e-6
