from __future__ import annotations

from collections import defaultdict

from pf_recommend_runtime.artifacts import LoadedModel
from pf_recommend_schemas import ItemScore


def recommend_for_user(model: LoadedModel, user_id: str, k: int) -> tuple[list[ItemScore], str, bool]:
    """Return (items, model_name, used_fallback). Unknown users get popularity."""
    history = model.user_history.get(user_id, [])
    seen = set(history)
    if not history:
        items = _with_titles(
            [row for row in model.popularity if row.item_id not in seen][:k],
            model,
            reason="cold_start_popularity",
        )
        return items, "popularity", True

    scores: dict[str, float] = defaultdict(float)
    for item_id in history:
        for neighbor in model.similar.get(item_id, []):
            if neighbor.item_id in seen:
                continue
            scores[neighbor.item_id] += neighbor.score

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    picked = [
        ItemScore(item_id=item_id, score=score, title=model.title_of(item_id), reason="item_item")
        for item_id, score in ranked[:k]
    ]
    if len(picked) < k:
        have = {row.item_id for row in picked} | seen
        for row in model.popularity:
            if row.item_id in have:
                continue
            picked.append(
                ItemScore(
                    item_id=row.item_id,
                    score=row.score,
                    title=model.title_of(row.item_id),
                    reason="popularity_fill",
                )
            )
            if len(picked) >= k:
                break
        return picked[:k], "item_item", len(ranked) == 0

    return picked, "item_item", False


def similar_items(model: LoadedModel, item_id: str, k: int) -> list[ItemScore] | None:
    """Neighbors for a known item. None means the item is not in this namespace."""
    if item_id not in model.catalog and item_id not in model.similar:
        return None
    neighbors = model.similar.get(item_id, [])
    if neighbors:
        return _with_titles(neighbors[:k], model, reason="item_item")
    fallback = [row for row in model.popularity if row.item_id != item_id][:k]
    return _with_titles(fallback, model, reason="popularity")


def _with_titles(rows: list[ItemScore], model: LoadedModel, reason: str) -> list[ItemScore]:
    out: list[ItemScore] = []
    for row in rows:
        out.append(
            ItemScore(
                item_id=row.item_id,
                score=row.score,
                title=row.title or model.title_of(row.item_id),
                reason=row.reason or reason,
            )
        )
    return out
