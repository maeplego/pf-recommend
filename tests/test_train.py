from pathlib import Path

from recommend_train.pipeline import run_train

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "testdata" / "ml-tiny"
JOBS = ROOT / "testdata" / "jobs-tiny"


def test_train_writes_time_split_metrics(tmp_path: Path) -> None:
    out = run_train(
        out=tmp_path,
        namespace="movies",
        k=5,
        neighbor_k=10,
        min_rating=4.0,
        test_fraction=0.2,
        cutoff="1593561600",
        ratings=TINY / "ratings.csv",
        movies=TINY / "movies.csv",
        events=None,
        items=None,
        download=None,
        cache_dir=tmp_path / "cache",
    )
    manifest = (out / "manifest.json").read_text(encoding="utf-8")
    assert '"split": "time"' in (out / "metrics.json").read_text(encoding="utf-8") or "time" in manifest
    metrics_text = (out / "metrics.json").read_text(encoding="utf-8")
    assert "recall_at_k" in metrics_text
    assert "item_item" in metrics_text
    assert "popularity" in metrics_text
    extra = (out / "manifest.json").read_text(encoding="utf-8")
    assert "random" not in extra


def test_jobs_namespace_learns_go_neighbors(tmp_path: Path) -> None:
    out = run_train(
        out=tmp_path,
        namespace="jobs",
        k=3,
        neighbor_k=5,
        min_rating=0.0,
        test_fraction=0.2,
        cutoff="1593561600",
        ratings=None,
        movies=None,
        events=JOBS / "events.csv",
        items=JOBS / "items.csv",
        download=None,
        cache_dir=tmp_path / "cache",
    )
    similar = (out / "similar.json").read_text(encoding="utf-8")
    assert "job_go_k8s" in similar
    assert "job_go_api" in similar
