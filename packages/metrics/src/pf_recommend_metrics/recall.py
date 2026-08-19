from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """|top-k ∩ relevant| / |relevant|. Empty relevant set is undefined → 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary NDCG@K. Empty relevant set → 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = {item for item in relevant if item}
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
