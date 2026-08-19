from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pf_recommend_schemas import Interaction


@dataclass(frozen=True)
class TimeSplit:
    train: list[Interaction]
    test: list[Interaction]
    cutoff: datetime

    def assert_no_future_leak(self) -> None:
        """Fail if any train row is at or after cutoff (future must not train)."""
        for row in self.train:
            if row.at >= self.cutoff:
                raise ValueError("train split leaked a row at or after cutoff")
        for row in self.test:
            if row.at < self.cutoff:
                raise ValueError("test split contains a row before cutoff")


def split_by_time(
    interactions: list[Interaction],
    *,
    cutoff: datetime | None = None,
    test_fraction: float = 0.2,
) -> TimeSplit:
    """Hold out the later interactions. Random split is intentionally not provided.

    Train is strictly before cutoff. Rows at cutoff belong to test.
    If cutoff is omitted, it is the timestamp of the row at (1 - test_fraction).
    """
    if not interactions:
        raise ValueError("interactions must not be empty")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    ordered = sorted(interactions, key=lambda row: (row.at, row.user_id, row.item_id))
    if cutoff is None:
        idx = int(len(ordered) * (1.0 - test_fraction))
        idx = min(max(idx, 1), len(ordered) - 1)
        cutoff = ordered[idx].at
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        # If many rows share the same timestamp, keep all of them on one side.
        while idx > 0 and ordered[idx - 1].at == cutoff:
            idx -= 1
        cutoff = ordered[idx].at

    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)

    train = [row for row in ordered if row.at < cutoff]
    test = [row for row in ordered if row.at >= cutoff]
    if not train:
        raise ValueError("time split produced an empty train set; choose an later cutoff")
    if not test:
        raise ValueError("time split produced an empty test set; choose an earlier cutoff")

    split = TimeSplit(train=train, test=test, cutoff=cutoff)
    split.assert_no_future_leak()
    return split
