"""
methods/metric_protonet.py — Method 3: Metric Prototypical Network (PROPOSED METHOD).

This is the novel contribution of the study. It extends standard ProtoNet
with a learnable diagonal distance metric that automatically suppresses
irrelevant feature dimensions.

Core insight
------------
Pre-trained CNN features (576-dim) are over-complete for domain-specific
few-shot tasks. They encode information irrelevant to tobacco leaf ripeness
(leaf shape, texture, generic ImageNet semantic categories). Standard
ProtoNet uses Euclidean distance, treating ALL 576 dimensions equally,
allowing irrelevant dimensions to degrade classification.

Solution
--------
Learn a diagonal distance metric w in R^576 (one weight per feature
dimension) from the support set via gradient descent:

    d(query, prototype) = sum_i  w_i * (q_i - p_i)^2

where w_i ~ 1 means dimension i is useful, and w_i ~ 0 means dimension i is
suppressed as irrelevant. This is pure deep learning — no hand-crafted
features, no color statistics, no external signal; w is learned purely by
backpropagation on the support set's cross-entropy loss.

Implementation note — autograd vs. manual gradients
-----------------------------------------------------
The closed-form gradient of the loss w.r.t. w has two components:
  * Direct term:   each sample's own distance to each prototype
  * Indirect term: each sample's contribution to its own class prototype,
                    which in turn affects every other sample's distance to
                    that prototype (since prototypes are themselves a
                    function of w through the weighted features).

Rather than hand-deriving and hand-coding this gradient (error-prone for a
result that will be reported in a paper), `w` is implemented as a
`torch.nn.Parameter` and the entire forward pass (weighted distance ->
logits -> softmax -> cross-entropy) is built from standard differentiable
PyTorch ops. `loss.backward()` then computes the exact same direct+indirect
gradient automatically and exactly. This is mathematically identical to the
closed form above, with no risk of a derivation bug:

    # Direct term (autograd derives this automatically):
    #   d_attended[i] += 2 * (attended[i] - proto[c]) * d_dists[i, c] * (1 - delta(y_i,c)/n_c)
    # Indirect term (also automatic, via the prototype's dependence on w):
    #   d_attended[k in class c] += sum_i(d_dists[i,c] * 2*(attended[i]-proto[c])) * (-1/n_c)
"""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import config
import evaluation


def run_metric_protonet(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    feature_extractor=None,
    n_classes: int = 3,
    n_epochs: int = None,
    lr: float = None,
    weight_decay: float = None,
    temperature: float = None,
    init_weights: np.ndarray = None,
    freeze_weights: bool = False,
    device: str = "cpu",
) -> Dict:
    """Train the learnable diagonal metric w on the support set, then classify
    the query set by nearest weighted-distance prototype.

    Parameters
    ----------
    init_weights : optional np.ndarray (D,)
        Custom initial weights for w (used by the ablation study, e.g. A2's
        random-frozen-weights condition). Defaults to ones(D).
    freeze_weights : bool
        If True, w is never updated (used by ablation A1: uniform frozen,
        and A2: random frozen). Training loop still runs so the returned
        dict has a consistent shape, but no optimizer step is taken.
    """
    n_epochs = config.METRIC_PROTONET_EPOCHS if n_epochs is None else n_epochs
    lr = config.METRIC_PROTONET_LR if lr is None else lr
    weight_decay = config.METRIC_PROTONET_WEIGHT_DECAY if weight_decay is None else weight_decay
    temperature = config.METRIC_PROTONET_TEMPERATURE if temperature is None else temperature

    torch.manual_seed(config.RANDOM_SEED)

    device = torch.device(device)
    D = support_features.shape[1]

    support_t = torch.tensor(support_features, dtype=torch.float32, device=device)
    support_labels_t = torch.tensor(support_labels, dtype=torch.long, device=device)
    query_t = torch.tensor(query_features, dtype=torch.float32, device=device)

    if init_weights is None:
        w_init = torch.ones(D, dtype=torch.float32, device=device)
    else:
        w_init = torch.tensor(init_weights, dtype=torch.float32, device=device)

    w = torch.nn.Parameter(w_init.clone(), requires_grad=not freeze_weights)

    optimizer = torch.optim.Adam([w], lr=lr, weight_decay=weight_decay) if not freeze_weights else None

    def compute_logits(features, labels_for_proto, proto_source_features):
        """Compute weighted-distance logits of `features` against prototypes
        built from `proto_source_features` grouped by `labels_for_proto`."""
        prototypes = torch.stack([
            proto_source_features[labels_for_proto == c].mean(dim=0)
            for c in range(n_classes)
        ])  # (C, D)
        diff = features.unsqueeze(1) - prototypes.unsqueeze(0)  # (N, C, D)
        weighted_sq = w.unsqueeze(0).unsqueeze(0) * diff ** 2   # (N, C, D)
        dists = weighted_sq.sum(dim=2)                          # (N, C)
        logits = -dists * temperature
        return logits, prototypes

    if not freeze_weights:
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            logits, _ = compute_logits(support_t, support_labels_t, support_t)
            loss = F.cross_entropy(logits, support_labels_t)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        query_logits, prototypes = compute_logits(query_t, support_labels_t, support_t)
        probs = F.softmax(query_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    metrics = evaluation.compute_metrics(
        query_labels, preds, y_proba=probs, n_classes=n_classes
    )
    metrics["predictions"] = preds
    metrics["probabilities"] = probs
    metrics["metric_weights"] = w.detach().cpu().numpy()
    metrics["prototypes"] = prototypes.detach().cpu().numpy()

    return metrics
