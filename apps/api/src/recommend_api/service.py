from __future__ import annotations

from recommend_api.registry import ModelRegistry
from pf_recommend_runtime import LoadedModel, recommend_for_user, similar_items
from pf_recommend_schemas import ItemScore


class UnknownNamespaceError(LookupError):
    pass


class UnknownItemError(LookupError):
    pass


class RecommendService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def require(self, namespace: str) -> LoadedModel:
        model = self.registry.get(namespace)
        if model is None:
            raise UnknownNamespaceError(namespace)
        return model

    def recommend(self, namespace: str, user_id: str, k: int) -> tuple[LoadedModel, list[ItemScore], str, bool]:
        model = self.require(namespace)
        items, name, fallback = recommend_for_user(model, user_id, k)
        return model, items, name, fallback

    def similar(self, namespace: str, item_id: str, k: int) -> tuple[LoadedModel, list[ItemScore]]:
        model = self.require(namespace)
        items = similar_items(model, item_id, k)
        if items is None:
            raise UnknownItemError(item_id)
        return model, items
