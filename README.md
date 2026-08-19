# pf-recommend

学習用の推薦です。学習ジョブと HTTP 推論は別プロセスで、指標・スキーマ・JSON 成果物を共有します。使うデータは MovieLens か、CI 用の架空サブセットだけです。**実顧客ログは使いません。本番レコメンドの置き換えではありません。**

| ディレクトリ | 役割 |
| --- | --- |
| `apps/api` | FastAPI 推論 |
| `apps/train` | バッチ学習（時間分割、人気 + item-item） |
| `apps/demo-web` | ユーザー切替デモ |
| `packages/metrics` | Recall@K、NDCG@K |
| `testdata/` | CI / Compose 用の架空データ |
| `models/` | 成果物（Git に入れない） |

## 起動

```powershell
cd deploy
copy .env.example .env
docker compose --env-file .env up --build
```

| URL | 用途 |
| --- | --- |
| http://localhost:3008 | デモ UI |
| http://localhost:8098/health | API |
| http://localhost:8098/docs | OpenAPI |

学習は `testdata/ml-tiny`（映画）と `testdata/jobs-tiny`（求人）で一度走り、終了します。API はその成果物を読みます。

## テスト

```powershell
python -m pip install -e ".[dev,api,train]"
python -m pytest
```

公開 MovieLens `ml-latest-small`（約 100k 件）を使うときは、学習時にダウンロードします。Git には置きません。ライセンスは [GroupLens](https://grouplens.org/datasets/movielens/) に従ってください。`testdata/ml-tiny` のタイトルは創作です。

## API の要点

- `GET /v1/similar-items?namespace=&item_id=&k=` — 類似アイテム。求人側は `namespace=jobs`
- `GET /v1/recommend?namespace=&user_id=` — ユーザー向け。未知ユーザーは人気へフォールバック
- 未知のアイテムは 404。呼び出し側はスキル重複などへ戻します
- オンライン学習はありません。再学習は CLI です

設計の詳細は [portfolio-plan](https://github.com/maeplego/portfolio-plan) の `portfolio-plan/recommend/docs/` です。
