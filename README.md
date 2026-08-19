# pf-recommend

P07 recommendation training and inference. **Not a production recommender.** MovieLens (or the fictional CI subset) only — no real customer logs.

Learning job and HTTP inference are separate processes. They share metrics, interaction schemas, and JSON artifacts.

## Layout

| Path | Role |
| --- | --- |
| `apps/api` | FastAPI inference |
| `apps/train` | Batch CLI (time-based split, popularity + item-item) |
| `apps/demo-web` | User switcher for MovieLens-shaped data |
| `packages/metrics` | Recall@K, NDCG@K, time-based split |
| `packages/schemas` | Interaction / item contracts |
| `packages/runtime` | Artifact load + lookup ranking (no full matrix on the request path) |
| `testdata/` | Tiny fictional fixtures for CI and Compose |
| `models/` | Local artifacts (gitignored) |

## Compose (required demo)

```powershell
cd deploy
copy .env.example .env
docker compose --env-file .env up --build
```

- API: http://localhost:8098/health  ·  OpenAPI: http://localhost:8098/docs
- Demo: http://localhost:3008
- Train runs once on `testdata/ml-tiny` (movies) and `testdata/jobs-tiny` (jobs), then exits. The API volume-mounts the artifacts.

Stop with `docker compose down`.

## Tests (host)

```powershell
python -m pip install -e ".[dev,api,train]"
python -m pytest
```

CI uses the fictional fixture, not a GroupLens download.

## Train on public MovieLens

`ml-latest-small` is about 100k ratings (~1MB zip). It is **not** stored in git. Download at train time:

```powershell
python -m pip install -e ".[train]"
python -m recommend_train --download ml-latest-small --namespace movies --out models --k 10
```

Source: [GroupLens MovieLens](https://grouplens.org/datasets/movielens/). Follow their license for redistribution. The `testdata/ml-tiny` titles are invented and are not MovieLens content.

Do not point `--download` at ml-25m in this repo; the inference artifacts are JSON neighbor lists, not a giant `npy`.

## API (P10 contract)

P10 talent-api calls this shape when `RECOMMEND_API_URL` is set. This repo does not call P10.

### `GET /v1/similar-items`

| Query | Type | Notes |
| --- | --- | --- |
| `namespace` | string | P10 sends `jobs`. Also `movies` (and later `commerce`). |
| `item_id` | string | Opaque string (MovieLens id or P10 job ULID). |
| `k` | int | Default 10, max 50. |

**200**

```json
{
  "namespace": "jobs",
  "item_id": "job_go_api",
  "model": "item_item",
  "version": "…",
  "items": [
    { "item_id": "job_go_k8s", "score": 0.91, "title": "Go platform engineer", "reason": "item_item" }
  ]
}
```

P10 only needs `items[].item_id`. Extra fields are safe to ignore.

**404** `{ "error": { "code": "not_found", "message": "…" } }` — unknown namespace or item. P10 treats non-OK as skill-overlap fallback.

**503** `/ready` when no namespace has been trained yet.

Until a jobs model is trained on P10 ids, similar-items for real job ULIDs 404s and P10 should keep its fallback. The Compose jobs fixture uses fictional ids (`job_go_api`, …) only to prove the path.

### Other routes

| Method | Path | Role |
| --- | --- | --- |
| GET | `/health` | Liveness `{ "ok": true }` |
| GET | `/ready` | At least one loaded namespace |
| GET | `/v1/recommend?namespace=&user_id=&k=10` | User ranking. Unknown user → popularity, `fallback: true` |
| GET | `/v1/models` | Version, time-split cutoff, Recall@K / NDCG@K |
| POST | `/v1/events` | Append-only JSONL for a later retrain. No online learning |

There is **no** public `/admin/train`. Retrain with the CLI.

## Evaluation

Training always holds out later interactions (`packages/metrics.split_by_time`). Random split is not implemented and is not a completion criterion. Users with no pre-cutoff history are counted as cold-start, not as CF eval users.

## Limits

- File-backed registry (`manifest.json` written last). Postgres / MinIO are not wired.
- Item-item cosine on implicit co-occurrence, plus a popularity baseline. No implicit/LightFM, no ANN cluster, no A/B.
- P06 commerce adapter is not implemented. P10 overlay wiring is not implemented here.
- Redis cache is not used; lookups are dicts from JSON.

Design: `project/portfolio-plan/recommend/DESIGN.md`
