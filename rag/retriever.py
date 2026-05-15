"""
High-level semantic retriever for the RAG pipeline.

Combines the Embedder and VectorStore to provide a simple API:
  query(email_text) → list of similar historical emails
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RAG_TOP_K
from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Retriever:
    """
    Retrieve similar historical emails for a given query.

    Usage:
        retriever = Retriever()
        retriever.load_index()
        results = retriever.query("URGENT: Project deadline tomorrow")
    """

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.store = VectorStore()

    def load_index(self) -> None:
        """Load the pre-built FAISS index from disk."""
        self.store.load()

    def query(self, email_text: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        Embed a query email and retrieve the top-k most similar historical emails.

        Args:
            email_text: Raw or cleaned email text.
            top_k:      Number of results.

        Returns:
            List of dicts with keys:
              - subject, body_snippet, priority_label, similarity_score
        """
        query_embedding = self.embedder.embed_text(email_text)
        results = self.store.search(query_embedding, top_k=top_k)
        return results
