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

        Over-fetches from the index and deduplicates on the fly so that
        duplicate copies of the same email (common in the Enron corpus)
        do not consume multiple result slots.

        Args:
            email_text: Raw or cleaned email text.
            top_k:      Number of results.

        Returns:
            List of dicts with keys:
              - subject, body_snippet, priority_label, similarity_score
        """
        query_embedding = self.embedder.embed_text(email_text)

        # Over-fetch to account for duplicates in the index
        raw_results = self.store.search(query_embedding, top_k=top_k * 4)

        # Deduplicate by body snippet content
        seen = set()
        unique_results = []
        for res in raw_results:
            snippet = res.get("body_snippet", "").strip()
            if snippet not in seen:
                seen.add(snippet)
                unique_results.append(res)
            if len(unique_results) == top_k:
                break

        return unique_results
