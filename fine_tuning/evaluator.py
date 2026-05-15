"""
Classification evaluation utilities.

Prints accuracy, weighted/macro F1, per-class precision/recall,
and a confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PRIORITY_LABELS, ID2LABEL


class Evaluator:
    """Evaluate classification predictions against ground-truth labels."""

    @staticmethod
    def print_report(y_true: list[int], y_pred: list[int]) -> dict:
        """
        Print a full classification report and confusion matrix.

        Args:
            y_true: Ground-truth label indices.
            y_pred: Predicted label indices.

        Returns:
            Dictionary with accuracy, macro_f1, weighted_f1.
        """
        # Convert indices to label names
        true_labels = [ID2LABEL[int(y)] for y in y_true]
        pred_labels = [ID2LABEL[int(y)] for y in y_pred]

        acc = accuracy_score(true_labels, pred_labels)

        print("\n" + "=" * 55)
        print(f"{'Classification Report':^55}")
        print("=" * 55)
        print(classification_report(
            true_labels, pred_labels,
            labels=PRIORITY_LABELS,
            zero_division=0,
        ))

        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=PRIORITY_LABELS)
        print("Confusion Matrix:")
        header = "            " + "  ".join(f"{l:>8}" for l in PRIORITY_LABELS)
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>8}" for v in row)
            print(f"  {PRIORITY_LABELS[i]:<10}{row_str}")
        print("=" * 55)

        report = classification_report(
            true_labels, pred_labels,
            labels=PRIORITY_LABELS,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
