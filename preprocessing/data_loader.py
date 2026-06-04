"""
Data loader for the Enron email dataset.

Downloads from Kaggle via kagglehub, parses raw RFC-822 email messages,
and applies the Cleaner pipeline to produce a processed DataFrame.
"""

import os
import pandas as pd
import kagglehub
from email import message_from_string

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    KAGGLE_DATASET, SAMPLE_SIZE, PROCESSED_CSV, DATA_DIR,
)
from preprocessing.cleaner import Cleaner


class DataLoader:
    """Load, parse, clean, and export the Enron email dataset."""

    def __init__(self, sample_size: int | None = SAMPLE_SIZE):
        self.sample_size = sample_size
        self._cleaner = Cleaner()
        self.df: pd.DataFrame | None = None

    # ── Download & Load ───────────────────────────────────────────────────────
    def load(self) -> "DataLoader":
        """Download the Enron dataset and load into a DataFrame."""
        print("[DataLoader] Downloading Enron dataset...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        file_path = os.path.join(path, "emails.csv")
        self.df = pd.read_csv(file_path)

        if self.sample_size is not None:
            self.df = self.df.head(self.sample_size)

        print(f"[DataLoader] Loaded {len(self.df)} emails.")
        return self

    # ── Parse Raw Email ───────────────────────────────────────────────────────
    @staticmethod
    def _parse_email(message: str) -> dict:
        """Extract from/to/subject/date/body from a raw RFC-822 message."""
        try:
            msg = message_from_string(message)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(errors="ignore")
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore")

            return {
                "from": msg.get("From"),
                "to": msg.get("To"),
                "subject": msg.get("Subject"),
                "date": msg.get("Date"),
                "body": body.strip(),
            }
        except Exception:
            return {"from": None, "to": None, "subject": None,
                    "date": None, "body": None}

    def parse(self) -> "DataLoader":
        """Parse all raw email messages into structured columns."""
        print("[DataLoader] Parsing raw emails...")
        parsed = self.df["message"].apply(self._parse_email)
        parsed_df = pd.DataFrame(parsed.tolist())
        self.df = pd.concat([self.df, parsed_df], axis=1)
        print(f"[DataLoader] Parsed {len(self.df)} emails.")
        return self

    # ── Clean ─────────────────────────────────────────────────────────────────
    def clean(self) -> "DataLoader":
        """Apply cleaning pipelines and drop unusable rows."""
        print("[DataLoader] Cleaning email bodies...")
        self.df["clean_body_summary"] = self.df["body"].apply(
            self._cleaner.clean_for_summarization
        )
        self.df["clean_body_classify"] = self.df["body"].apply(
            self._cleaner.clean_for_classification
        )

        before = len(self.df)
        self.df = self.df[
            self.df["clean_body_classify"].str.strip() != ""
        ].reset_index(drop=True)
        print(f"[DataLoader] Dropped {before - len(self.df)} unusable rows.")

        # Deduplicate emails with identical cleaned body text
        before_dup = len(self.df)
        self.df = self.df.drop_duplicates(
            subset=["clean_body_classify"]
        ).reset_index(drop=True)
        print(f"[DataLoader] Dropped {before_dup - len(self.df)} duplicate emails.")

        return self

    # ── Save ──────────────────────────────────────────────────────────────────
    def save(self, output_path: str = PROCESSED_CSV) -> str:
        """Export processed emails to CSV."""
        final_cols = ["date", "from", "to", "subject",
                      "clean_body_summary", "clean_body_classify"]
        self.df[final_cols].to_csv(output_path, index=False)
        print(f"[DataLoader] Saved {len(self.df)} rows -> {output_path}")
        return output_path

    # ── Convenience: Run Full Pipeline ────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """Run load → parse → clean → save, return the DataFrame."""
        self.load().parse().clean().save()
        return self.df


# ── Standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    loader.run()
