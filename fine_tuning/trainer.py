"""
DistilBERT fine-tuning trainer for email priority classification.

Phase 1 of the architecture:
  Raw Email Data → Data Preprocessing → Pre-trained Model (DistilBERT)
  → Supervised Fine-Tuning → Fine-Tuned Model (with weights)
"""

import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Suppress expected MISSING/UNEXPECTED weight warnings when loading
# base distilbert for classification (classification head is new).
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CLASSIFIER_MODEL_NAME, NUM_LABELS, LABEL2ID, ID2LABEL,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE,
    NUM_EPOCHS, WEIGHT_DECAY, WARMUP_RATIO, TRAIN_SPLIT,
    EARLY_STOPPING_PATIENCE, RANDOM_SEED, MODEL_DIR,
)
from fine_tuning.dataset import EmailPriorityDataset
from fine_tuning.evaluator import Evaluator


class FineTuner:
    """Fine-tune DistilBERT for 3-class email priority classification."""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[FineTuner] Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)

        # Load pre-trained DistilBERT with a classification head
        self.model = DistilBertForSequenceClassification.from_pretrained(
            CLASSIFIER_MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

        self.evaluator = Evaluator()

    # ── Build DataLoaders ─────────────────────────────────────────────────────

    def _build_dataloaders(
        self, texts: list[str], labels: list[str]
    ) -> tuple[DataLoader, DataLoader]:
        """Split data and create train/val DataLoaders."""
        dataset = EmailPriorityDataset(texts, labels, self.tokenizer)

        train_size = int(len(dataset) * TRAIN_SPLIT)
        val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(RANDOM_SEED)
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False
        )

        print(f"[FineTuner] Train: {train_size} | Val: {val_size}")
        return train_loader, val_loader

    # ── Training Loop ─────────────────────────────────────────────────────────

    def train(self, texts: list[str], labels: list[str]) -> dict:
        """
        Fine-tune the model.

        Args:
            texts:  List of cleaned email bodies.
            labels: List of priority labels ("High"/"Medium"/"Low").

        Returns:
            Dictionary with training history (loss, accuracy per epoch).
        """
        train_loader, val_loader = self._build_dataloaders(texts, labels)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training state
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        print(f"\n[FineTuner] Starting training for {NUM_EPOCHS} epochs...")
        for epoch in range(1, NUM_EPOCHS + 1):
            # ── Train ─────────────────────────────────────────────────────
            self.model.train()
            total_train_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)

            # ── Validate ──────────────────────────────────────────────────
            val_loss, val_acc, _, _ = self._evaluate(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            print(
                f"  Epoch {epoch}: train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            # ── Early Stopping ────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model()
                print(f"  [OK] Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"  [STOP] Early stopping at epoch {epoch}")
                    break

        # Final evaluation with full report
        print("\n[FineTuner] Final evaluation on validation set:")
        _, _, all_preds, all_labels = self._evaluate(val_loader)
        self.evaluator.print_report(all_labels, all_preds)

        return history

    # ── Evaluate ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self, dataloader: DataLoader
    ) -> tuple[float, float, list[int], list[int]]:
        """Run evaluation on a dataloader. Returns loss, accuracy, preds, labels."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        return avg_loss, accuracy, all_preds, all_labels

    # ── Save / Load Model ─────────────────────────────────────────────────────

    def _save_model(self) -> None:
        """Save model + tokenizer to MODEL_DIR."""
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)

    def load_model(self, path: str = MODEL_DIR) -> None:
        """Load a saved fine-tuned model."""
        print(f"[FineTuner] Loading model from {path}...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.eval()
        print("[FineTuner] Model loaded.")

    # ── Single Prediction ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, text: str) -> tuple[str, float, dict]:
        """
        Classify a single email.

        Returns:
            (predicted_label, confidence, all_probabilities)
        """
        self.model.eval()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        label = ID2LABEL[pred_idx]

        all_probs = {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)}

        return label, confidence, all_probs
