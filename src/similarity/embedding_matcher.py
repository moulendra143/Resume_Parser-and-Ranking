from sentence_transformers import SentenceTransformer, util

# Load a lightweight, pre-trained model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using sentence transformers."""
    if not text1 or not text2:
        return 0.0
        
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_score = util.cos_sim(embeddings1, embeddings2).item()
    return cosine_score
