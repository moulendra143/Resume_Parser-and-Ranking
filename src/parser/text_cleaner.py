"""
spaCy-based text cleaner with lazy model loading.

The spaCy model is loaded only when first needed, not at import time.
"""

import re
import spacy

_nlp = None


def _get_nlp():
    """Lazy-load the spaCy model on first use."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def clean_text(text: str) -> str:
    """Clean and normalize raw text using spaCy."""
    if not text:
        return ""
    # Lowercase and remove extra whitespace
    text = re.sub(r"\s+", " ", text.lower()).strip()

    # Process text using spaCy
    nlp = _get_nlp()
    doc = nlp(text)

    # Keep alphanumeric words, ignore punctuation and stop words, lemmatize
    cleaned_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(cleaned_tokens)
