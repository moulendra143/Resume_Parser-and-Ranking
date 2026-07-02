"""
Sentence-Transformer embedding similarity with lazy model loading.

The model name is read from config/config.yaml (similarity.embedding_model),
defaulting to all-MiniLM-L6-v2. The model is only imported and loaded when
first needed, keeping Streamlit startup time under 1 second.
"""
import os
import warnings
import logging

# Suppress PyTorch/Triton CUDA warnings & Windows distributed redirects logs
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline loading to bypass HF network checks
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*Redirects.*")
logging.getLogger("torch").setLevel(logging.ERROR)

from src.utils.config_loader import get_config

_model = None


def _get_model() -> "SentenceTransformer":
    """Lazy-load the sentence-transformer model on first use.

    The model name is sourced from ``config/config.yaml`` so it can be
    swapped (e.g. ``all-mpnet-base-v2`` for higher accuracy) without
    touching source code.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name = get_config().similarity.embedding_model
        _model = SentenceTransformer(model_name)
    return _model


def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using sentence transformers.

    Args:
        text1: First text (typically the cleaned resume).
        text2: Second text (typically the cleaned job description).

    Returns:
        Cosine similarity score in [-1, 1], typically [0, 1] for natural text.
    """
    if not text1 or not text2:
        return 0.0

    from sentence_transformers import util
    model = _get_model()
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    cosine_score = util.cos_sim(embeddings1, embeddings2).item()
    return cosine_score

