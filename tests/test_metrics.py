from datetime import datetime, timezone

from pf_recommend_metrics import ndcg_at_k, recall_at_k, split_by_time
from pf_recommend_schemas import Interaction


def _row(user: str, item: str, ts: int) -> Interaction:
    return Interaction(
        namespace="movies",
        user_id=user,
        item_id=item,
        type="rating",
        at=datetime.fromtimestamp(ts, tz=timezone.utc),
        value=5,
    )


def test_recall_at_k_counts_hits_in_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], ["c", "z"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c", "z"], 3) == 0.5
    assert recall_at_k(["c", "a"], ["c"], 10) == 1.0


def test_ndcg_rewards_earlier_hits() -> None:
    early = ndcg_at_k(["hit", "miss"], ["hit"], 2)
    late = ndcg_at_k(["miss", "hit"], ["hit"], 2)
    assert early > late
    assert early == 1.0


def test_time_split_keeps_future_out_of_train() -> None:
    rows = [
        _row("1", "a", 100),
        _row("1", "b", 200),
        _row("2", "a", 150),
        _row("2", "c", 300),
    ]
    cutoff = datetime.fromtimestamp(200, tz=timezone.utc)
    split = split_by_time(rows, cutoff=cutoff)
    assert split.cutoff == cutoff
    assert {row.item_id for row in split.train} == {"a"}
    assert {row.item_id for row in split.test} == {"b", "c"}
    split.assert_no_future_leak()
    assert max(row.at for row in split.train) < min(row.at for row in split.test)


def test_time_split_fraction_uses_timestamp_not_shuffle() -> None:
    rows = [_row("u", str(i), 1000 + i) for i in range(10)]
    split = split_by_time(rows, test_fraction=0.2)
    assert all(row.at < split.cutoff for row in split.train)
    assert all(row.at >= split.cutoff for row in split.test)
    assert "random_split" not in dir(__import__("pf_recommend_metrics"))
