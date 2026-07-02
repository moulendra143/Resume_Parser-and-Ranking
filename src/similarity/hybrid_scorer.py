"""
Hybrid similarity scorer.

Combines Sentence-Transformer embedding similarity with TF-IDF cosine
similarity using a configurable weight read from ``config/config.yaml``.
The idea is that embeddings capture *semantic* overlap while TF-IDF captures
*lexical* keyword overlap — combining both improves robustness across
different resume styles and job description formats.

Weight defaults (from config):
    embedding_weight: 0.7   (semantic signal dominates)
    tfidf_weight:     0.3   (lexical keyword anchor)

Rationale for 0.7/0.3 split — see Model Choices & Tradeoff Notes in README.
"""

from src.similarity.embedding_matcher import compute_similarity as embedding_sim
from src.similarity.tfidf_matcher import compute_tfidf_similarity as tfidf_sim
from src.utils.config_loader import get_config


def compute_hybrid_similarity(
    text1: str,
    text2: str,
    embedding_weight: float | None = None,
) -> dict:
    """Compute a weighted hybrid similarity score.

    Args:
        text1: First text (cleaned resume).
        text2: Second text (cleaned job description).
        embedding_weight: Weight for the embedding score (0–1).  If omitted,
            the value from ``config/config.yaml`` is used
            (``similarity.embedding_weight``).  The TF-IDF weight is
            automatically set to ``1 - embedding_weight``.

    Returns:
        Dictionary with ``embedding``, ``tfidf``, and ``hybrid`` scores,
        each rounded to 4 decimal places.
    """
    if embedding_weight is None:
        embedding_weight = get_config().similarity.embedding_weight

    tfidf_weight = 1.0 - embedding_weight

    emb_score = embedding_sim(text1, text2)
    tfidf_score = tfidf_sim(text1, text2)
    hybrid_score = (emb_score * embedding_weight) + (tfidf_score * tfidf_weight)

    return {
        "embedding": round(emb_score, 4),
        "tfidf": round(tfidf_score, 4),
        "hybrid": round(hybrid_score, 4),
    }
