from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pf_recommend_schemas import Interaction, ItemMeta


def parse_cutoff(raw: str | None) -> datetime | None:
    if raw is None or raw == "":
        return None
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_movielens(
    ratings_path: Path,
    movies_path: Path | None,
    *,
    namespace: str,
    min_rating: float,
) -> tuple[list[Interaction], dict[str, ItemMeta]]:
    ratings = pd.read_csv(ratings_path)
    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv missing columns: {sorted(missing)}")

    catalog: dict[str, ItemMeta] = {}
    if movies_path and movies_path.is_file():
        movies = pd.read_csv(movies_path)
        for row in movies.itertuples(index=False):
            movie_id = str(getattr(row, "movieId"))
            title = str(getattr(row, "title", movie_id))
            genres = str(getattr(row, "genres", "") or "")
            tags = [part for part in genres.split("|") if part and part != "(no genres listed)"]
            catalog[movie_id] = ItemMeta(item_id=movie_id, title=title, tags=tags)

    interactions: list[Interaction] = []
    for row in ratings.itertuples(index=False):
        rating = float(getattr(row, "rating"))
        if rating < min_rating:
            continue
        item_id = str(getattr(row, "movieId"))
        if item_id not in catalog:
            catalog[item_id] = ItemMeta(item_id=item_id, title=item_id)
        interactions.append(
            Interaction(
                namespace=namespace,
                user_id=str(getattr(row, "userId")),
                item_id=item_id,
                type="rating",
                at=datetime.fromtimestamp(int(getattr(row, "timestamp")), tz=timezone.utc),
                value=rating,
            )
        )
    if not interactions:
        raise ValueError("no interactions after applying min-rating")
    return interactions, catalog


def load_events(
    events_path: Path,
    items_path: Path | None,
    *,
    namespace: str,
) -> tuple[list[Interaction], dict[str, ItemMeta]]:
    events = pd.read_csv(events_path)
    required = {"user_id", "item_id", "at"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events.csv missing columns: {sorted(missing)}")

    catalog: dict[str, ItemMeta] = {}
    if items_path and items_path.is_file():
        items = pd.read_csv(items_path)
        for row in items.itertuples(index=False):
            item_id = str(getattr(row, "item_id"))
            title = str(getattr(row, "title", item_id))
            tags_raw = str(getattr(row, "tags", "") or "")
            tags = [part.strip() for part in tags_raw.split("|") if part.strip()]
            catalog[item_id] = ItemMeta(item_id=item_id, title=title, tags=tags)

    interactions: list[Interaction] = []
    for row in events.itertuples(index=False):
        item_id = str(getattr(row, "item_id"))
        if item_id not in catalog:
            catalog[item_id] = ItemMeta(item_id=item_id, title=item_id)
        at_raw = getattr(row, "at")
        if isinstance(at_raw, (int, float)) or str(at_raw).isdigit():
            at = datetime.fromtimestamp(int(at_raw), tz=timezone.utc)
        else:
            at = datetime.fromisoformat(str(at_raw).replace("Z", "+00:00"))
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
        event_type = str(getattr(row, "type", "implicit") or "implicit")
        allowed = {"rating", "view", "click", "apply", "bookmark", "implicit"}
        if event_type not in allowed:
            event_type = "implicit"
        interactions.append(
            Interaction(
                namespace=namespace,
                user_id=str(getattr(row, "user_id")),
                item_id=item_id,
                type=event_type,  # type: ignore[arg-type]
                at=at,
            )
        )
    if not interactions:
        raise ValueError("events.csv produced no interactions")
    return interactions, catalog
