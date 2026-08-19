from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_movielens_small(cache_dir: Path) -> tuple[Path, Path]:
    """Download ml-latest-small (~1MB zip) if missing. Not committed to git."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "ml-latest-small"
    ratings = extract_dir / "ratings.csv"
    movies = extract_dir / "movies.csv"
    if ratings.is_file() and movies.is_file():
        return ratings, movies

    zip_path = cache_dir / "ml-latest-small.zip"
    if not zip_path.is_file():
        urlretrieve(MOVIELENS_SMALL_URL, zip_path)  # noqa: S310 — documented public dataset
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_dir)
    if not ratings.is_file() or not movies.is_file():
        raise FileNotFoundError("MovieLens zip did not contain ratings.csv and movies.csv")
    return ratings, movies
