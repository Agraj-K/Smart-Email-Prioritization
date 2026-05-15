"""
PyTorch Dataset for DistilBERT email priority classification.

Tokenizes email text with the DistilBERT tokenizer and returns
input_ids, attention_mask, and label tensors.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CLASSIFIER_MODEL_NAME, MAX_SEQ_LENGTH, LABEL2ID


class EmailPriorityDataset(Dataset):
    """Map-style dataset for DistilBERT fine-tuning."""

    def __init__(self, texts: list[str], labels: list[str],
                 tokenizer: AutoTokenizer | None = None,
                 max_length: int = MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx]) if self.texts[idx] else ""
        label = LABEL2ID[self.labels[idx]]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
