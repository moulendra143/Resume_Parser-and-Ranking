import re
import spacy

# Load spaCy small English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    """Clean and normalize raw text using spaCy."""
    # Lowercase and remove extra whitespace
    text = re.sub(r"\s+", " ", text.lower()).strip()
    
    # Process text using spaCy
    doc = nlp(text)
    
    # Keep alphanumeric words, ignore punctuation and stop words, lemmatize
    cleaned_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(cleaned_tokens)
