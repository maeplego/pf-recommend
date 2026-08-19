from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pf_recommend_schemas import ItemMeta, ItemScore, ModelMetrics


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _item_score(raw: dict[str, Any]) -> ItemScore:
    return ItemScore.model_validate(raw)


@dataclass
class LoadedModel:
    namespace: str
    version: str
    trained_at: str
    k: int
    metrics: dict[str, ModelMetrics]
    popularity: list[ItemScore]
    similar: dict[str, list[ItemScore]]
    user_history: dict[str, list[str]]
    catalog: dict[str, ItemMeta]
    sample_user_ids: list[str]
    cold_start_user_id: str
    n_users: int
    n_items: int
    cutoff: str
    path: Path | None
    manifest_mtime: float
    extra: dict[str, Any] = field(default_factory=dict)

    def title_of(self, item_id: str) -> str:
        meta = self.catalog.get(item_id)
        if meta and meta.title:
            return meta.title
        return item_id


class ArtifactStore:
    """File-backed registry. manifest.json is written last so a reload is atomic."""

    def __init__(self, root: Path):
        self.root = root

    def namespace_dir(self, namespace: str) -> Path:
        return self.root / namespace

    def write(
        self,
        *,
        namespace: str,
        version: str,
        trained_at: str,
        k: int,
        metrics: dict[str, ModelMetrics],
        popularity: list[ItemScore],
        similar: dict[str, list[ItemScore]],
        user_history: dict[str, list[str]],
        catalog: dict[str, ItemMeta],
        sample_user_ids: list[str],
        cold_start_user_id: str,
        cutoff: str,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        target = self.namespace_dir(namespace)
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "popularity.json", [row.model_dump() for row in popularity])
        _write_json(
            target / "similar.json",
            {item_id: [row.model_dump() for row in neighbors] for item_id, neighbors in similar.items()},
        )
        _write_json(target / "user_history.json", user_history)
        _write_json(target / "catalog.json", {item_id: meta.model_dump() for item_id, meta in catalog.items()})
        _write_json(
            target / "metrics.json",
            {name: model.model_dump() for name, model in metrics.items()},
        )
        manifest = {
            "namespace": namespace,
            "version": version,
            "trained_at": trained_at,
            "k": k,
            "n_users": len(user_history),
            "n_items": len(catalog),
            "sample_user_ids": sample_user_ids,
            "cold_start_user_id": cold_start_user_id,
            "cutoff": cutoff,
            "metrics": {name: model.model_dump() for name, model in metrics.items()},
            "extra": extra or {},
        }
        _write_json(target / "manifest.json", manifest)
        return target

    def load(self, namespace: str) -> LoadedModel:
        target = self.namespace_dir(namespace)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no model for namespace={namespace}")
        manifest = _read_json(manifest_path)
        popularity = [_item_score(row) for row in _read_json(target / "popularity.json")]
        similar_raw = _read_json(target / "similar.json")
        similar = {
            item_id: [_item_score(row) for row in neighbors]
            for item_id, neighbors in similar_raw.items()
        }
        user_history = {str(uid): [str(i) for i in items] for uid, items in _read_json(target / "user_history.json").items()}
        catalog_raw = _read_json(target / "catalog.json")
        catalog = {item_id: ItemMeta.model_validate(meta) for item_id, meta in catalog_raw.items()}
        metrics = {
            name: ModelMetrics.model_validate(payload)
            for name, payload in manifest.get("metrics", {}).items()
        }
        return LoadedModel(
            namespace=str(manifest["namespace"]),
            version=str(manifest["version"]),
            trained_at=str(manifest["trained_at"]),
            k=int(manifest.get("k", 10)),
            metrics=metrics,
            popularity=popularity,
            similar=similar,
            user_history=user_history,
            catalog=catalog,
            sample_user_ids=[str(uid) for uid in manifest.get("sample_user_ids", [])],
            cold_start_user_id=str(manifest.get("cold_start_user_id", "new-user")),
            n_users=int(manifest.get("n_users", len(user_history))),
            n_items=int(manifest.get("n_items", len(catalog))),
            cutoff=str(manifest.get("cutoff", "")),
            path=target,
            manifest_mtime=manifest_path.stat().st_mtime,
            extra=dict(manifest.get("extra") or {}),
        )

    def list_namespaces(self) -> list[str]:
        if not self.root.is_dir():
            return []
        found: list[str] = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                found.append(child.name)
        return found
