import pandas as pd
import torch
import numpy as np
import re
import nltk

from transformers import AutoTokenizer, AutoModel
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


class FeatureExtractor:
    def __init__(self, df):
        self.df = df

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Stage 3] Loading BERT on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.model = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        ).to(self.device)

        self.model.eval()

        self.sia = SentimentIntensityAnalyzer()

        self.urgency_words = [
            "urgent",
            "asap",
            "deadline",
            "important",
            "tomorrow",
            "today",
            "immediately"
        ]

        self.action_phrases = [
            "can you",
            "please send",
            "need you",
            "please review",
            "submit",
            "approve",
            "respond",
            "schedule",
            "meeting"
        ]

    # -------------------------
    # BERT Embeddings
    # -------------------------
    @torch.no_grad()
    def get_bert_embedding(self, text):
        if not isinstance(text, str) or not text.strip():
            return np.zeros(768)

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.cpu().numpy().flatten()

    # -------------------------
    # Urgency Score
    # -------------------------
    def urgency_score(self, text):
        if not isinstance(text, str):
            return 0

        text = text.lower()
        score = 0

        for word in self.urgency_words:
            if re.search(rf"\b{word}\b", text):
                score += 1

        return score

    # -------------------------
    # Action Request Score
    # -------------------------
    def action_score(self, text):
        if not isinstance(text, str):
            return 0

        text = text.lower()
        score = 0

        for phrase in self.action_phrases:
            if phrase in text:
                score += 1

        return score

    # -------------------------
    # Sentiment Score
    # -------------------------
    def sentiment_score(self, text):
        if not isinstance(text, str):
            return 0

        return self.sia.polarity_scores(text)["compound"]

    # -------------------------
    # Sender Score
    # -------------------------
    def sender_score(self, sender):
        if not isinstance(sender, str):
            return 0

        sender = sender.lower()

        if ".edu" in sender:
            return 3
        elif "ceo" in sender:
            return 4
        elif "manager" in sender:
            return 3
        elif "noreply" in sender:
            return 0
        else:
            return 2

    # -------------------------
    # Thread Score
    # -------------------------
    def thread_score(self, subject):
        if not isinstance(subject, str):
            return 0

        subject = subject.lower()
        score = 0

        if "re:" in subject:
            score += 1

        if "fwd:" in subject:
            score += 1

        return score

    # -------------------------
    # Extract Features
    # -------------------------
    def extract_features(self):
        print("[Stage 3] Extracting contextual features...")

        self.df["urgency_score"] = self.df["clean_body_classify"].apply(
            self.urgency_score
        )

        self.df["action_score"] = self.df["clean_body_classify"].apply(
            self.action_score
        )

        self.df["sentiment_score"] = self.df["clean_body_classify"].apply(
            self.sentiment_score
        )

        self.df["sender_score"] = self.df["from"].apply(
            self.sender_score
        )

        self.df["thread_score"] = self.df["subject"].apply(
            self.thread_score
        )

        print("[Stage 3] Extracting BERT embeddings...")

        embeddings = []

        for text in self.df["clean_body_classify"]:
            embeddings.append(self.get_bert_embedding(text))

        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f"bert_{i}" for i in range(768)]
        )

        self.df = pd.concat(
            [self.df.reset_index(drop=True), embedding_df],
            axis=1
        )

        return self.df