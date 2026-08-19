from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recommend_api.registry import ModelRegistry
from recommend_api.service import RecommendService, UnknownItemError, UnknownNamespaceError

MAX_K = 50
DEFAULT_K = 10


class ItemOut(BaseModel):
    item_id: str
    score: float = 0.0
    title: str = ""
    reason: str = ""


class RecommendOut(BaseModel):
    namespace: str
    user_id: str
    model: str
    version: str
    fallback: bool
    items: list[ItemOut]


class SimilarOut(BaseModel):
    namespace: str
    item_id: str
    model: str
    version: str
    items: list[ItemOut]


class EventIn(BaseModel):
    namespace: str
    user_id: str
    item_id: str
    type: str = "view"
    at: datetime | None = None


class ModelOut(BaseModel):
    namespace: str
    version: str
    trained_at: str
    cutoff: str
    n_users: int
    n_items: int
    sample_user_ids: list[str]
    cold_start_user_id: str
    metrics: dict
    extra: dict = Field(default_factory=dict)


def _clamp_k(k: int) -> int:
    return max(1, min(MAX_K, k))


def _models_dir() -> Path:
    return Path(os.environ.get("RECOMMEND_MODELS_DIR", "models")).resolve()


def _cors_origins() -> list[str]:
    raw = os.environ.get("RECOMMEND_CORS_ORIGIN", "http://localhost:3008")
    return [part.strip() for part in raw.split(",") if part.strip()]


def create_app(models_dir: Path | None = None) -> FastAPI:
    root = models_dir or _models_dir()
    registry = ModelRegistry(root)
    service = RecommendService(registry)
    app = FastAPI(
        title="pf-recommend",
        version="0.1.0",
        description="P07 inference API. Train with the batch CLI, not over HTTP.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.registry = registry
    app.state.service = service
    app.state.models_dir = root

    @app.exception_handler(UnknownNamespaceError)
    async def namespace_missing(_request: Request, exc: UnknownNamespaceError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "not_found", "message": f"no model for namespace={exc}"}},
            status_code=404,
        )

    @app.exception_handler(UnknownItemError)
    async def item_missing(_request: Request, exc: UnknownItemError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "not_found", "message": f"unknown item_id={exc}"}},
            status_code=404,
        )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/ready")
    def ready() -> JSONResponse:
        names = registry.namespaces()
        if not names:
            return JSONResponse(
                {"ok": False, "error": {"code": "unavailable", "message": "no trained namespaces"}},
                status_code=503,
            )
        return JSONResponse({"ok": True, "namespaces": names})

    @app.get("/v1/models")
    def list_models() -> dict:
        models = [
            ModelOut(
                namespace=model.namespace,
                version=model.version,
                trained_at=model.trained_at,
                cutoff=model.cutoff,
                n_users=model.n_users,
                n_items=model.n_items,
                sample_user_ids=model.sample_user_ids,
                cold_start_user_id=model.cold_start_user_id,
                metrics={name: row.model_dump() for name, row in model.metrics.items()},
                extra=model.extra,
            ).model_dump()
            for model in registry.all()
        ]
        return {"models": models}

    @app.get("/v1/recommend", response_model=RecommendOut)
    def recommend(
        namespace: str = Query(..., min_length=1),
        user_id: str = Query(..., min_length=1),
        k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
    ) -> RecommendOut:
        model, items, name, fallback = service.recommend(namespace, user_id, _clamp_k(k))
        return RecommendOut(
            namespace=namespace,
            user_id=user_id,
            model=name,
            version=model.version,
            fallback=fallback,
            items=[ItemOut.model_validate(row.model_dump()) for row in items],
        )

    @app.get("/v1/similar-items", response_model=SimilarOut)
    def similar(
        namespace: str = Query(..., min_length=1),
        item_id: str = Query(..., min_length=1),
        k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
    ) -> SimilarOut:
        """P10 calls this as GET /v1/similar-items?namespace=jobs&item_id=&k=."""
        model, items = service.similar(namespace, item_id, _clamp_k(k))
        return SimilarOut(
            namespace=namespace,
            item_id=item_id,
            model="item_item" if items and items[0].reason == "item_item" else "popularity",
            version=model.version,
            items=[ItemOut.model_validate(row.model_dump()) for row in items],
        )

    @app.post("/v1/events")
    def ingest_event(body: EventIn) -> dict:
        """Append-only log for a later retrain. Not used for online learning."""
        service.require(body.namespace)
        log_dir = root / "events"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = (body.at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        payload = {
            "namespace": body.namespace,
            "user_id": body.user_id,
            "item_id": body.item_id,
            "type": body.type,
            "at": stamp.isoformat().replace("+00:00", "Z"),
        }
        with (log_dir / f"{body.namespace}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return {"ok": True}

    return app


app = create_app()
