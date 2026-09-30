"""
evaluation.py — Metrics, confusion matrices, and statistical significance tests.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

import config


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_classes: int = 3,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute accuracy, per-class precision/recall/F1, confusion matrix,
    and (if class probabilities are supplied) top-2 accuracy.

    Parameters
    ----------
    y_true, y_pred : np.ndarray of int class indices, shape (N,)
    y_proba : optional np.ndarray, shape (N, n_classes) of class probabilities/scores.
              If provided, top-2 accuracy is computed; otherwise top-2 falls
              back to equal to top-1 accuracy.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_normalized = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    support = np.zeros(n_classes, dtype=int)

    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support[c] = np.sum(y_true == c)

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    if y_proba is not None:
        top2_preds = np.argsort(-y_proba, axis=1)[:, :2]
        top2_correct = np.any(top2_preds == y_true[:, None], axis=1)
        top2_accuracy = float(np.mean(top2_correct))
    else:
        top2_accuracy = accuracy

    per_class = {
        class_names[c]: {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }
        for c in range(n_classes)
    }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "top2_accuracy": top2_accuracy,
        "per_class": per_class,
    }


def wilcoxon_test(method_a_scores: np.ndarray, method_b_scores: np.ndarray) -> Dict:
    """Pairwise Wilcoxon signed-rank test between two methods' per-run accuracy scores.

    Returns {'statistic': float, 'p_value': float, 'significant': bool (p<0.05)}.
    If all differences are zero (identical scores), the test is undefined;
    we return p_value=1.0, significant=False in that case.
    """
    a = np.asarray(method_a_scores, dtype=np.float64)
    b = np.asarray(method_b_scores, dtype=np.float64)

    diffs = a - b
    if np.allclose(diffs, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    try:
        statistic, p_value = stats.wilcoxon(a, b)
    except ValueError:
        # e.g. sample size too small, or all-zero differences edge case
        return {"statistic": float("nan"), "p_value": 1.0, "significant": False}

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two paired/independent samples (pooled std)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / pooled_std)
