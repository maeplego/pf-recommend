from __future__ import annotations

from pathlib import Path

from pf_recommend_runtime import ArtifactStore, LoadedModel


class ModelRegistry:
    """Loads JSON artifacts and reloads a namespace when its manifest mtime changes."""

    def __init__(self, root: Path):
        self.store = ArtifactStore(root)
        self._models: dict[str, LoadedModel] = {}

    def refresh(self) -> None:
        for namespace in self.store.list_namespaces():
            self._refresh_one(namespace)

    def get(self, namespace: str) -> LoadedModel | None:
        self._refresh_one(namespace)
        return self._models.get(namespace)

    def all(self) -> list[LoadedModel]:
        self.refresh()
        return [self._models[name] for name in sorted(self._models)]

    def namespaces(self) -> list[str]:
        self.refresh()
        return list(self._models)

    def _refresh_one(self, namespace: str) -> None:
        manifest = self.store.namespace_dir(namespace) / "manifest.json"
        if not manifest.is_file():
            self._models.pop(namespace, None)
            return
        mtime = manifest.stat().st_mtime
        current = self._models.get(namespace)
        if current is not None and current.manifest_mtime == mtime:
            return
        self._models[namespace] = self.store.load(namespace)
