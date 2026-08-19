from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from pf_recommend_metrics import ndcg_at_k, recall_at_k
from pf_recommend_schemas import Interaction, ItemMeta, ItemScore, ModelMetrics
from pf_recommend_runtime.artifacts import LoadedModel
from pf_recommend_runtime.ranking import recommend_for_user


def build_popularity(train: list[Interaction], catalog: dict[str, ItemMeta]) -> list[ItemScore]:
    counts: Counter[str] = Counter(row.item_id for row in train)
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        ItemScore(
            item_id=item_id,
            score=float(score),
            title=catalog.get(item_id).title if catalog.get(item_id) else item_id,
            reason="popularity",
        )
        for item_id, score in ranked
    ]


def build_item_item(
    train: list[Interaction],
    catalog: dict[str, ItemMeta],
    *,
    neighbor_k: int,
) -> dict[str, list[ItemScore]]:
    users = sorted({row.user_id for row in train})
    items = sorted({row.item_id for row in train})
    if len(items) < 2 or len(users) < 1:
        return {item_id: [] for item_id in items}

    user_index = {user_id: idx for idx, user_id in enumerate(users)}
    item_index = {item_id: idx for idx, item_id in enumerate(items)}
    pairs = {(row.item_id, row.user_id) for row in train}
    data = np.ones(len(pairs), dtype=np.float32)
    rows = np.fromiter((item_index[item_id] for item_id, _ in pairs), dtype=np.int32, count=len(pairs))
    cols = np.fromiter((user_index[user_id] for _, user_id in pairs), dtype=np.int32, count=len(pairs))
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(items), len(users)))
    similarity = cosine_similarity(matrix, dense_output=False)

    neighbors: dict[str, list[ItemScore]] = {}
    for item_id, idx in item_index.items():
        row = similarity.getrow(idx)
        order = np.argsort(-row.data)
        picked: list[ItemScore] = []
        for pos in order:
            other_idx = int(row.indices[pos])
            if other_idx == idx:
                continue
            score = float(row.data[pos])
            if score <= 0.0:
                continue
            other_id = items[other_idx]
            title = catalog.get(other_id).title if catalog.get(other_id) else other_id
            picked.append(ItemScore(item_id=other_id, score=score, title=title, reason="item_item"))
            if len(picked) >= neighbor_k:
                break
        neighbors[item_id] = picked
    return neighbors


def user_history(train: list[Interaction]) -> dict[str, list[str]]:
    history: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in sorted(train, key=lambda item: item.at):
        if row.item_id in seen[row.user_id]:
            continue
        seen[row.user_id].add(row.item_id)
        history[row.user_id].append(row.item_id)
    return dict(history)


def evaluate(
    model: LoadedModel,
    test: list[Interaction],
    *,
    k: int,
    force_popularity: bool,
) -> ModelMetrics:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in test:
        relevant[row.user_id].add(row.item_id)

    recalls: list[float] = []
    ndcgs: list[float] = []
    cold = 0
    for user_id, positives in relevant.items():
        history = model.user_history.get(user_id, [])
        if not history:
            cold += 1
            continue
        holdout = [item_id for item_id in positives if item_id not in set(history)]
        if not holdout:
            continue
        if force_popularity:
            seen = set(history)
            recommended = [row.item_id for row in model.popularity if row.item_id not in seen][:k]
        else:
            items, _, _ = recommend_for_user(model, user_id, k)
            recommended = [row.item_id for row in items]
        recalls.append(recall_at_k(recommended, holdout, k))
        ndcgs.append(ndcg_at_k(recommended, holdout, k))

    n_eval = len(recalls)
    return ModelMetrics(
        k=k,
        n_eval_users=n_eval,
        n_cold_start_users=cold,
        recall_at_k=(sum(recalls) / n_eval) if n_eval else None,
        ndcg_at_k=(sum(ndcgs) / n_eval) if n_eval else None,
    )


def to_loaded_model(
    *,
    namespace: str,
    version: str,
    trained_at: str,
    k: int,
    popularity: list[ItemScore],
    similar: dict[str, list[ItemScore]],
    history: dict[str, list[str]],
    catalog: dict[str, ItemMeta],
    cutoff: datetime,
) -> LoadedModel:
    sample = list(history.keys())[:8]
    return LoadedModel(
        namespace=namespace,
        version=version,
        trained_at=trained_at,
        k=k,
        metrics={},
        popularity=popularity,
        similar=similar,
        user_history=history,
        catalog=catalog,
        sample_user_ids=sample,
        cold_start_user_id="new-user",
        n_users=len(history),
        n_items=len(catalog),
        cutoff=cutoff.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        path=None,
        manifest_mtime=0.0,
    )
