"""Generate sentence embeddings locally using sentence-transformers."""
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None  # lazy load

def get_model():
    global _model
    if _model is None:
        # all-MiniLM-L6-v2 is small, fast, and solid for similarity tasks
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    return get_model().encode(text, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))