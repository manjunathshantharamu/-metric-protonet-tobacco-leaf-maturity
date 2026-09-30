"""
methods/transfer_learning.py — Method 1: Frozen CNN + trained linear classifier.

Architecture:
    Image -> [Frozen MobileNetV3-Small] -> 576-dim features ->
              [Linear classifier (trained)] -> Softmax

Procedure:
    1. Extract frozen CNN features for support and query sets (done upstream,
       once per episode — see run_experiment.py).
    2. Train a logistic regression classifier on the support features.
    3. Classify query features with the trained classifier.

We use scikit-learn's LogisticRegression rather than a hand-rolled PyTorch
linear layer: for a few-shot support set (k=5..30 per class) this is a
convex problem with a handful of features relative to sample count, so
LBFGS converges reliably and deterministically without needing to tune a
separate training loop (epochs/lr/optimizer) for this baseline method.
"""

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

import evaluation


def run_transfer_learning(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    feature_extractor=None,
    n_classes: int = 3,
) -> Dict:
    # NOTE: scikit-learn >= 1.5 deprecated (and >= 1.7 removed) the
    # `multi_class` argument — 'lbfgs' now handles multinomial softmax
    # regression automatically for multi-class problems, so we build the
    # kwargs dict conditionally for compatibility across sklearn versions.
    lr_kwargs = dict(C=1.0, max_iter=1000, solver="lbfgs")
    try:
        clf = LogisticRegression(multi_class="multinomial", **lr_kwargs)
    except TypeError:
        clf = LogisticRegression(**lr_kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf.fit(support_features, support_labels)

    probs_partial = clf.predict_proba(query_features)

    # LogisticRegression may drop classes absent from the support set (should
    # not happen given k >= 5 per class, but guard against it defensively so
    # the returned probability matrix always has n_classes columns).
    probs = np.zeros((len(query_features), n_classes), dtype=np.float64)
    for i, c in enumerate(clf.classes_):
        probs[:, c] = probs_partial[:, i]

    preds = np.argmax(probs, axis=1)

    metrics = evaluation.compute_metrics(
        query_labels, preds, y_proba=probs, n_classes=n_classes
    )
    metrics["predictions"] = preds
    metrics["probabilities"] = probs

    return metrics
