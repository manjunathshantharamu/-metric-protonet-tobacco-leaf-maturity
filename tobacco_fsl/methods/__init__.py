"""
methods/ — Few-shot classification methods compared in this study.

    transfer_learning.py  Method 1: Frozen CNN + trained linear classifier
    protonet.py            Method 2: Standard Prototypical Network (Snell et al., 2017)
    metric_protonet.py     Method 3: Metric ProtoNet (PROPOSED) — learnable diagonal metric
    full_finetune.py       Method 4: Full fine-tune (upper bound, uses all data)

Note: CLIP zero-shot and the legacy color-attention ProtoNet were evaluated
during development but are not part of this codebase — CLIP produced flat,
near-random (~20%) accuracy due to domain gap, and color_protonet was an
earlier, weaker version of the proposed metric_protonet method.
"""

from . import transfer_learning
from . import protonet
from . import metric_protonet
from . import full_finetune

__all__ = ["transfer_learning", "protonet", "metric_protonet", "full_finetune"]
