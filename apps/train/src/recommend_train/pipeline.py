from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pf_recommend_metrics import split_by_time
from pf_recommend_runtime import ArtifactStore
from pf_recommend_schemas import ModelMetrics

from recommend_train.data import load_events, load_movielens, parse_cutoff
from recommend_train.download import ensure_movielens_small
from recommend_train.fit import (
    build_item_item,
    build_popularity,
    evaluate,
    to_loaded_model,
    user_history,
)


def run_train(
    *,
    out: Path,
    namespace: str,
    k: int,
    neighbor_k: int,
    min_rating: float,
    test_fraction: float,
    cutoff: str | None,
    ratings: Path | None,
    movies: Path | None,
    events: Path | None,
    items: Path | None,
    download: str | None,
    cache_dir: Path,
) -> Path:
    if download == "ml-latest-small":
        ratings, movies = ensure_movielens_small(cache_dir)
        namespace = namespace or "movies"

    if ratings is not None:
        interactions, catalog = load_movielens(
            ratings, movies, namespace=namespace, min_rating=min_rating
        )
    elif events is not None:
        interactions, catalog = load_events(events, items, namespace=namespace)
    else:
        raise ValueError("provide --ratings, --events, or --download ml-latest-small")

    split = split_by_time(interactions, cutoff=parse_cutoff(cutoff), test_fraction=test_fraction)
    history = user_history(split.train)
    popularity = build_popularity(split.train, catalog)
    similar = build_item_item(split.train, catalog, neighbor_k=neighbor_k)
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    version = f"{trained_at.replace(':', '').replace('.', '')}-itemitem"
    draft = to_loaded_model(
        namespace=namespace,
        version=version,
        trained_at=trained_at,
        k=k,
        popularity=popularity,
        similar=similar,
        history=history,
        catalog=catalog,
        cutoff=split.cutoff,
    )
    metrics = {
        "popularity": evaluate(draft, split.test, k=k, force_popularity=True),
        "item_item": evaluate(draft, split.test, k=k, force_popularity=False),
    }
    # Serving fail-closes when CF is worse than popularity; still persist both metrics.
    _print_report(namespace, split.cutoff, metrics, k)

    store = ArtifactStore(out)
    return store.write(
        namespace=namespace,
        version=version,
        trained_at=trained_at,
        k=k,
        metrics=metrics,
        popularity=popularity,
        similar=similar,
        user_history=history,
        catalog=catalog,
        sample_user_ids=draft.sample_user_ids,
        cold_start_user_id=draft.cold_start_user_id,
        cutoff=draft.cutoff,
        extra={
            "n_train": len(split.train),
            "n_test": len(split.test),
            "min_rating": min_rating if ratings is not None else None,
            "split": "time",
        },
    )


def _print_report(namespace: str, cutoff: datetime, metrics: dict[str, ModelMetrics], k: int) -> None:
    print(f"namespace={namespace} cutoff={cutoff.astimezone(timezone.utc).isoformat()} split=time k={k}")
    print(f"{'model':<12} {'recall@k':>10} {'ndcg@k':>10} {'users':>7} {'cold':>6}")
    for name, row in metrics.items():
        recall = "n/a" if row.recall_at_k is None else f"{row.recall_at_k:.4f}"
        ndcg = "n/a" if row.ndcg_at_k is None else f"{row.ndcg_at_k:.4f}"
        print(f"{name:<12} {recall:>10} {ndcg:>10} {row.n_eval_users:>7} {row.n_cold_start_users:>6}")
