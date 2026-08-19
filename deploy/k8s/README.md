# P07 recommend Kubernetes

Inference API in namespace `p07`. Init container `pf-recommend-train` writes models to an emptyDir; the API serves them from `RECOMMEND_MODELS_DIR=/models`.

Do not apply this folder alone for the demo — overlay D (commerce) and overlay C (talent) reference it.

| Service | Port | Notes |
| --- | --- | --- |
| `api` | 8098 | `GET /v1/recommend`, `GET /v1/similar-items`. Ingress `recommend-api.localhost` on overlays C/D |

Train runs once per pod start (tiny fixtures). Serving fail-closes in P06/P10 adapters; this API still returns 404 for unknown item ids.
