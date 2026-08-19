from pathlib import Path

from fastapi.testclient import TestClient

from recommend_api.main import create_app
from recommend_train.pipeline import run_train

ROOT = Path(__file__).resolve().parents[1]


def _trained_client(tmp_path: Path) -> TestClient:
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
    run_train(
        out=tmp_path,
        namespace="jobs",
        k=3,
        neighbor_k=5,
        min_rating=0.0,
        test_fraction=0.2,
        cutoff="1593561600",
        ratings=None,
        movies=None,
        events=ROOT / "testdata" / "jobs-tiny" / "events.csv",
        items=ROOT / "testdata" / "jobs-tiny" / "items.csv",
        download=None,
        cache_dir=tmp_path / "cache",
    )
    run_train(
        out=tmp_path,
        namespace="commerce",
        k=3,
        neighbor_k=5,
        min_rating=0.0,
        test_fraction=0.2,
        cutoff="1593561600",
        ratings=None,
        movies=None,
        events=ROOT / "testdata" / "commerce-tiny" / "events.csv",
        items=ROOT / "testdata" / "commerce-tiny" / "items.csv",
        download=None,
        cache_dir=tmp_path / "cache",
    )
    return TestClient(create_app(tmp_path))


def test_health_and_models(tmp_path: Path) -> None:
    client = _trained_client(tmp_path)
    assert client.get("/health").json() == {"ok": True}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert "movies" in ready.json()["namespaces"]
    models = client.get("/v1/models").json()["models"]
    names = {row["namespace"] for row in models}
    assert names == {"jobs", "movies", "commerce"}


def test_recommend_known_user_and_cold_start(tmp_path: Path) -> None:
    client = _trained_client(tmp_path)
    known = client.get("/v1/recommend", params={"namespace": "movies", "user_id": "1", "k": 5})
    assert known.status_code == 200
    body = known.json()
    assert body["fallback"] is False
    assert body["items"]
    assert all("item_id" in item for item in body["items"])

    cold = client.get("/v1/recommend", params={"namespace": "movies", "user_id": "new-user", "k": 5})
    assert cold.status_code == 200
    cold_body = cold.json()
    assert cold_body["fallback"] is True
    assert cold_body["model"] == "popularity"


def test_similar_items_jobs_contract_for_p10(tmp_path: Path) -> None:
    """P10 calls GET /v1/similar-items?namespace=jobs&item_id=&k= and reads items[].item_id."""
    client = _trained_client(tmp_path)
    res = client.get(
        "/v1/similar-items",
        params={"namespace": "jobs", "item_id": "job_go_api", "k": 5},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["namespace"] == "jobs"
    ids = [item["item_id"] for item in body["items"]]
    assert "job_go_k8s" in ids
    assert "job_go_api" not in ids


def test_similar_items_commerce_sku_contract_for_p06(tmp_path: Path) -> None:
    """P06 BFF maps item_id to catalog SKU (MUG-1 / TEE-1 / STK-1)."""
    client = _trained_client(tmp_path)
    res = client.get(
        "/v1/similar-items",
        params={"namespace": "commerce", "item_id": "MUG-1", "k": 5},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["namespace"] == "commerce"
    ids = [item["item_id"] for item in body["items"]]
    assert "TEE-1" in ids
    assert "MUG-1" not in ids


def test_commerce_cold_start_recommend_falls_back(tmp_path: Path) -> None:
    client = _trained_client(tmp_path)
    res = client.get(
        "/v1/recommend",
        params={"namespace": "commerce", "user_id": "brand-new-shopper", "k": 5},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["fallback"] is True
    assert body["items"]


def test_unknown_jobs_item_is_not_ok_so_p10_can_fallback(tmp_path: Path) -> None:
    client = _trained_client(tmp_path)
    res = client.get(
        "/v1/similar-items",
        params={"namespace": "jobs", "item_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV", "k": 5},
    )
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_missing_namespace_is_not_ok(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    res = client.get("/v1/similar-items", params={"namespace": "jobs", "item_id": "x", "k": 5})
    assert res.status_code == 404
    ready = client.get("/ready")
    assert ready.status_code == 503
