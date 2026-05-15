"""
Rule-based priority label generator.

Computes urgency, action-request, sentiment, sender-reputation, and
thread-depth scores, then maps a composite score to High / Medium / Low.
"""

import re
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LABELED_CSV, PRIORITY_LABELS


class LabelGenerator:
    """Generate priority labels for unlabeled email data."""

    URGENCY_WORDS = [
        "urgent", "asap", "deadline", "important", "critical",
        "immediately", "priority", "emergency", "time-sensitive",
        "today", "tomorrow", "eod", "end of day",
    ]

    ACTION_PHRASES = [
        "can you", "please send", "need you", "please review",
        "submit", "approve", "respond", "schedule", "meeting",
        "follow up", "action required", "reply needed",
    ]

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    # ── Individual Score Functions ────────────────────────────────────────────

    def urgency_score(self, text: str) -> int:
        """Count urgency keyword matches in text."""
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for w in self.URGENCY_WORDS
                   if re.search(rf"\b{w}\b", text_lower))

    def action_score(self, text: str) -> int:
        """Count action-request phrase matches."""
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for phrase in self.ACTION_PHRASES if phrase in text_lower)

    def sentiment_score(self, text: str) -> float:
        """VADER compound sentiment score (−1 to +1)."""
        if not isinstance(text, str):
            return 0.0
        return self.sia.polarity_scores(text)["compound"]

    def sender_score(self, sender: str) -> int:
        """Heuristic sender-reputation score."""
        if not isinstance(sender, str):
            return 0
        sender = sender.lower()
        if "ceo" in sender or "president" in sender:
            return 4
        if "manager" in sender or "director" in sender:
            return 3
        if ".edu" in sender or ".gov" in sender:
            return 3
        if "noreply" in sender or "no-reply" in sender:
            return 0
        return 2

    def thread_score(self, subject: str) -> int:
        """Score based on reply/forward depth (re:, fwd:)."""
        if not isinstance(subject, str):
            return 0
        subject = subject.lower()
        score = 0
        if "re:" in subject:
            score += 1
        if "fwd:" in subject:
            score += 1
        return score

    # ── Composite Label Assignment ────────────────────────────────────────────

    def subject_urgency_score(self, subject: str) -> int:
        """Check subject line for urgency markers (often more telling than body)."""
        if not isinstance(subject, str):
            return 0
        subject = subject.lower()
        score = 0
        for word in ["urgent", "asap", "important", "action", "deadline",
                      "critical", "immediate", "priority", "emergency"]:
            if word in subject:
                score += 2
        return score

    def _assign_priority(self, row: pd.Series) -> str:
        """Map composite score to a priority label."""
        score = 0
        score += row["urgency_score"] * 2
        score += row["action_score"] * 2
        score += row.get("subject_urgency", 0)
        score += row["thread_score"]
        # Sender score contributes less (almost everyone is @enron.com = 2)
        score += max(0, row["sender_score"] - 2)
        if row["sentiment_score"] < -0.3:
            score += 1

        if score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add priority_label column to the DataFrame.

        Expects columns: clean_body_classify, from, subject
        Returns the DataFrame with new score columns + priority_label.
        """
        print("[LabelGenerator] Computing feature scores...")

        df = df.copy()
        df["urgency_score"] = df["clean_body_classify"].apply(self.urgency_score)
        df["action_score"] = df["clean_body_classify"].apply(self.action_score)
        df["sentiment_score"] = df["clean_body_classify"].apply(self.sentiment_score)
        df["sender_score"] = df["from"].apply(self.sender_score)
        df["thread_score"] = df["subject"].apply(self.thread_score)
        df["subject_urgency"] = df["subject"].apply(self.subject_urgency_score)

        df["priority_label"] = df.apply(self._assign_priority, axis=1)

        print("[LabelGenerator] Priority distribution:")
        print(df["priority_label"].value_counts().to_string())

        return df

    def generate_and_save(self, df: pd.DataFrame,
                          output_path: str = LABELED_CSV) -> pd.DataFrame:
        """Generate labels and save the labeled DataFrame."""
        df = self.generate_labels(df)
        save_cols = ["date", "from", "to", "subject",
                     "clean_body_summary", "clean_body_classify",
                     "urgency_score", "action_score", "sentiment_score",
                     "sender_score", "thread_score", "subject_urgency",
                     "priority_label"]
        # Only keep columns that exist
        save_cols = [c for c in save_cols if c in df.columns]
        df[save_cols].to_csv(output_path, index=False)
        print(f"[LabelGenerator] Saved labeled data -> {output_path}")
        return df
