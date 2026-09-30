"""
methods/protonet.py — Method 2: Standard Prototypical Network.

Reference: Snell, Swersky & Zemel (2017), "Prototypical Networks for
Few-Shot Learning", NeurIPS.

Architecture:
    Image -> [Frozen CNN] -> 576-dim features
    Support set -> Prototype per class = mean(features per class)
    Query image -> Classify by nearest prototype (Euclidean distance)

No training required — inference only. Very fast; serves as the non-learned
baseline that Metric ProtoNet (methods/metric_protonet.py) builds on.
"""

from typing import Dict

import numpy as np

import config
import evaluation


def run_protonet(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    feature_extractor=None,
    n_classes: int = 3,
    temperature: float = None,
) -> Dict:
    """Standard Prototypical Network: nearest-prototype classification under
    plain (unweighted) squared Euclidean distance.

    `feature_extractor` is accepted for API consistency with the other
    method functions but is not used here since features are precomputed.
    """
    if temperature is None:
        temperature = config.PROTONET_TEMPERATURE

    prototypes = np.zeros((n_classes, support_features.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        mask = support_labels == c
        prototypes[c] = support_features[mask].mean(axis=0)

    # Squared Euclidean distance from every query to every prototype.
    diffs = query_features[:, None, :] - prototypes[None, :, :]  # (Nq, C, D)
    dists = np.sum(diffs ** 2, axis=2)  # (Nq, C)

    logits = -dists * temperature
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    preds = np.argmax(probs, axis=1)

    metrics = evaluation.compute_metrics(
        query_labels, preds, y_proba=probs, n_classes=n_classes
    )
    metrics["predictions"] = preds
    metrics["probabilities"] = probs
    metrics["prototypes"] = prototypes

    return metrics
