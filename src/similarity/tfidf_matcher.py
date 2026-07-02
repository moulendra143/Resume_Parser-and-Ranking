"""
TF-IDF based similarity scoring.

Uses scikit-learn's TfidfVectorizer to compute cosine similarity between
resume text and job description text.  This provides a fast, interpretable
baseline that does not require a GPU or large model download.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF vectors.

    Args:
        text1: First text (typically the cleaned resume).
        text2: Second text (typically the cleaned job description).

    Returns:
        Cosine similarity score in [0, 1].
    """
    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0])
