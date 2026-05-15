"""
FAISS-based vector store for email embeddings.

Phase 2 of the architecture:
  Email Embeddings, Metadata → Vector Database (FAISS / Metadata)
  → Semantic Search → Query Embedding

Supports building, saving, loading, and searching a FAISS index.
"""

import os
import pickle
import numpy as np
import faiss

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    EMBEDDING_DIM, FAISS_INDEX_FILE, FAISS_METADATA_FILE, FAISS_DIR,
)


class VectorStore:
    """FAISS-backed vector store with metadata persistence."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    # ── Build Index ───────────────────────────────────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Build a FAISS index from embeddings.

        Args:
            embeddings: Array of shape (N, EMBEDDING_DIM), L2-normalized.
            metadata:   List of dicts with email info (subject, body, label, etc.)
        """
        assert embeddings.shape[1] == EMBEDDING_DIM, (
            f"Expected dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )
        assert len(embeddings) == len(metadata), (
            "Embeddings and metadata must have same length"
        )

        # Inner-product index (equivalent to cosine similarity for L2-normed vectors)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

        print(f"[VectorStore] Built index with {self.index.ntotal} vectors.")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(
        self,
        index_path: str = FAISS_INDEX_FILE,
        metadata_path: str = FAISS_METADATA_FILE,
    ) -> None:
        """Persist the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[VectorStore] Saved index -> {index_path}")
        print(f"[VectorStore] Saved metadata -> {metadata_path}")

    def load(
        self,
        index_path: str = FAISS_INDEX_FILE,
        metadata_path: str = FAISS_METADATA_FILE,
    ) -> None:
        """Load a previously saved FAISS index and metadata."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run `python pipeline.py index` first."
            )
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(
            f"[VectorStore] Loaded index ({self.index.ntotal} vectors) "
            f"from {index_path}"
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[dict]:
        """
        Find the top-k most similar emails to a query embedding.

        Args:
            query_embedding: 1-D array of shape (EMBEDDING_DIM,).
            top_k: Number of results to return.

        Returns:
            List of dicts, each with 'score' + original metadata keys.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build_index() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            result = {**self.metadata[idx], "similarity_score": float(score)}
            results.append(result)

        return results
