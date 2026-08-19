from pathlib import Path

from fastapi.testclient import TestClient

from recommend_api.main import create_app
from recommend_train.pipeline import run_train

ROOT = Path(__file__).resolve().parents[1]


def test_api_reloads_when_manifest_mtime_changes(tmp_path: Path) -> None:
    run_train(
        out=tmp_path,
        namespace="movies",
        k=5,
        neighbor_k=10,
        min_rating=4.0,
        test_fraction=0.2,
        cutoff="1593561600",
        ratings=ROOT / "testdata" / "ml-tiny" / "ratings.csv",
        movies=ROOT / "testdata" / "ml-tiny" / "movies.csv",
        events=None,
        items=None,
        download=None,
        cache_dir=tmp_path / "cache",
    )
    client = TestClient(create_app(tmp_path))
    first = client.get("/v1/models").json()["models"][0]["version"]
    run_train(
        out=tmp_path,
        namespace="movies",
        k=5,
        neighbor_k=10,
        min_rating=4.0,
        test_fraction=0.2,
        cutoff="1593561600",
        ratings=ROOT / "testdata" / "ml-tiny" / "ratings.csv",
        movies=ROOT / "testdata" / "ml-tiny" / "movies.csv",
        events=None,
        items=None,
        download=None,
        cache_dir=tmp_path / "cache",
    )
    second = client.get("/v1/models").json()["models"][0]["version"]
    assert first != second
