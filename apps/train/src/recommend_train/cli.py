from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from recommend_train.pipeline import run_train

    parser = argparse.ArgumentParser(
        description="Train popularity + item-item models with a time-based holdout.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Artifact root (models/)")
    parser.add_argument("--namespace", required=True, help="movies | jobs | commerce")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--neighbor-k", type=int, default=50)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--cutoff", default=None, help="ISO-8601 or unix seconds; train is strictly before this")
    parser.add_argument("--ratings", type=Path, help="MovieLens ratings.csv")
    parser.add_argument("--movies", type=Path, help="MovieLens movies.csv")
    parser.add_argument("--events", type=Path, help="Generic events CSV: user_id,item_id,type,at")
    parser.add_argument("--items", type=Path, help="Generic items CSV: item_id,title,tags")
    parser.add_argument(
        "--download",
        choices=["ml-latest-small"],
        help="Fetch a public GroupLens MovieLens zip into --cache-dir, then train",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/movielens"))
    args = parser.parse_args(argv)

    try:
        run_train(
            out=args.out,
            namespace=args.namespace,
            k=args.k,
            neighbor_k=args.neighbor_k,
            min_rating=args.min_rating,
            test_fraction=args.test_fraction,
            cutoff=args.cutoff,
            ratings=args.ratings,
            movies=args.movies,
            events=args.events,
            items=args.items,
            download=args.download,
            cache_dir=args.cache_dir,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
