"""One lazy embedding model shared by indexing and retrieval."""
from functools import lru_cache
from threading import RLock

import numpy as np

from config import CACHE_PATH, EMBEDDING_MODEL

_MODEL_LOCK = RLock()


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBEDDING_MODEL, cache_folder=str(CACHE_PATH / "huggingface" / "sentence_transformers")
    )


def encode_texts(texts):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    with _MODEL_LOCK:
        return np.asarray(get_embedding_model().encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=True,
        ), dtype=np.float32)


def create_embeddings(chunks):
    return encode_texts([chunk["text"] for chunk in chunks])
