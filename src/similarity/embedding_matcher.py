"""
Sentence-Transformer embedding similarity with lazy model loading.

The model is loaded only when first needed, not at import time,
which dramatically speeds up Streamlit page loads.
"""

from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model():
    """Lazy-load the sentence-transformer model on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using sentence transformers."""
    if not text1 or not text2:
        return 0.0

    model = _get_model()
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    cosine_score = util.cos_sim(embeddings1, embeddings2).item()
    return cosine_score
