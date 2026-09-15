"""
evaluate.py

Evaluation metrics for the baseline (and, in Phase 3, RoBERTa) classifier.

Per the brief's §15: accuracy is never reported alone. This module always
returns per-class precision/recall/F1, macro F1, weighted F1, and — for
binary classification — ROC-AUC and PR-AUC, plus the confusion matrix.
Recall on the positive (crisis/suicide) class gets particular attention
via `false_negative_rate`, since a missed high-risk post is the costliest
failure mode for this system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict  # {label: {precision, recall, f1, support}}
    confusion_matrix: list  # 2D list, rows=true, cols=predicted
    roc_auc: float | None
    pr_auc: float | None
    false_negative_rate_positive_class: float | None
    false_positive_rate_positive_class: float | None

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "false_negative_rate_positive_class": self.false_negative_rate_positive_class,
            "false_positive_rate_positive_class": self.false_positive_rate_positive_class,
        }


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    positive_label: int = 1,
) -> EvaluationResult:
    """
    y_proba: predicted probability of the positive class (1D array), used
    for ROC-AUC/PR-AUC. Pass None to skip those two metrics (e.g. for a
    model that only outputs hard labels).
    """
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    roc_auc = pr_auc = None
    if y_proba is not None and len(set(y_true)) == 2:
        roc_auc = round(float(roc_auc_score(y_true, y_proba)), 4)
        pr_auc = round(float(average_precision_score(y_true, y_proba)), 4)

    fnr = fpr = None
    if positive_label in labels:
        pos_idx = labels.index(positive_label)
        neg_labels = [l for l in labels if l != positive_label]
        tp = cm[pos_idx][pos_idx]
        fn = sum(cm[pos_idx]) - tp
        if (tp + fn) > 0:
            fnr = round(fn / (tp + fn), 4)
        if neg_labels:
            neg_idx = labels.index(neg_labels[0])
            fp = cm[neg_idx][pos_idx]
            tn = sum(cm[neg_idx]) - fp
            if (fp + tn) > 0:
                fpr = round(fp / (fp + tn), 4)

    return EvaluationResult(
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class=per_class,
        confusion_matrix=cm,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        false_negative_rate_positive_class=fnr,
        false_positive_rate_positive_class=fpr,
    )


def print_evaluation(result: EvaluationResult, positive_label_name: str = "positive") -> None:
    print(f"\nAccuracy:      {result.accuracy}")
    print(f"Macro F1:      {result.macro_f1}")
    print(f"Weighted F1:   {result.weighted_f1}")
    if result.roc_auc is not None:
        print(f"ROC-AUC:       {result.roc_auc}")
        print(f"PR-AUC:        {result.pr_auc}")
    print("\nPer-class metrics:")
    for label, m in result.per_class.items():
        print(f"  class {label}: precision={m['precision']} recall={m['recall']} "
              f"f1={m['f1']} support={m['support']}")
    if result.false_negative_rate_positive_class is not None:
        print(f"\nFalse-negative rate ({positive_label_name} missed as negative): "
              f"{result.false_negative_rate_positive_class}")
        print(f"False-positive rate ({positive_label_name} wrongly flagged): "
              f"{result.false_positive_rate_positive_class}")
        print("  ^ For crisis detection, false negatives (missed risk) are the costlier "
              "error — watch this number closely across experiments.")
    print(f"\nConfusion matrix (rows=true, cols=predicted): {result.confusion_matrix}")
