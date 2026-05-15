"""
BERT-based email embedder for the RAG pipeline.

Phase 2 of the architecture:
  Historical Email Archives → Chunking & Embedding (BERT-based Embedder)
  → Email Embeddings, Metadata → Vector Database (FAISS)

Uses sentence-transformers/all-MiniLM-L6-v2 for high-quality semantic
embeddings optimized for similarity search.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, CHUNK_MAX_WORDS


class Embedder:
    """Chunk and embed email texts using a sentence-transformer model."""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Embedder] Loading {EMBEDDING_MODEL_NAME} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(self.device)
        self.model.eval()

    # ── Chunking ──────────────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
        """
        Split long emails into overlapping chunks of max_words.
        Short emails are returned as a single-element list.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        overlap = max_words // 4  # 25% overlap between chunks
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += max_words - overlap

        return chunks

    # ── Mean Pooling ──────────────────────────────────────────────────────────

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Average the token embeddings, weighted by attention mask."""
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    # ── Embed Single Text ─────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string. Returns a 1-D numpy array of shape (384,).
        For long texts, embeds each chunk and averages.
        """
        chunks = self.chunk_text(text)
        if not chunks:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        all_embeddings = []
        for chunk in chunks:
            encoded = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            embedding = self._mean_pooling(outputs, encoded["attention_mask"])
            # L2 normalize for cosine similarity
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            all_embeddings.append(embedding.cpu().numpy().flatten())

        # Average chunk embeddings
        return np.mean(all_embeddings, axis=0).astype(np.float32)

    # ── Batch Embed ───────────────────────────────────────────────────────────

    def embed_batch(
        self, texts: list[str], batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts. Returns array of shape (N, 384).
        """
        embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="[Embedder] Embedding emails")

        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            for text in batch_texts:
                embeddings.append(self.embed_text(text))

        return np.vstack(embeddings)
