from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal["rating", "view", "click", "apply", "bookmark", "implicit"]
NamespaceName = Literal["movies", "jobs", "commerce"]


class Interaction(BaseModel):
    namespace: str
    user_id: str
    item_id: str
    type: EventType = "implicit"
    at: datetime
    value: float | None = None


class ItemMeta(BaseModel):
    item_id: str
    title: str = ""
    tags: list[str] = Field(default_factory=list)


class ItemScore(BaseModel):
    item_id: str
    score: float
    title: str = ""
    reason: str = ""


class ModelMetrics(BaseModel):
    k: int
    n_eval_users: int = 0
    n_cold_start_users: int = 0
    recall_at_k: float | None = None
    ndcg_at_k: float | None = None
