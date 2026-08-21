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

Compose 起動時、`recommend-train` が次の 3 namespace を一度学習して終了し、API がその成果物を読みます。

| namespace | データ |
| --- | --- |
| `movies` | `testdata/ml-tiny` |
| `jobs` | `testdata/jobs-tiny`（P10） |
| `commerce` | `testdata/commerce-tiny`（P06。item_id＝SKU） |

## テスト

```powershell
python -m pip install -e ".[dev,api,train]"
python -m pytest
```

公開 MovieLens `ml-latest-small`（約 100k 件）を使うときは、学習時にダウンロードします。Git には置きません。ライセンスは [GroupLens](https://grouplens.org/datasets/movielens/) に従ってください。`testdata/ml-tiny` のタイトルは創作です。

## API の要点

- `GET /v1/similar-items?namespace=&item_id=&k=` — 類似アイテム。求人は `namespace=jobs`、EC は `namespace=commerce`
- `GET /v1/recommend?namespace=&user_id=` — ユーザー向け。未知ユーザーは人気へフォールバック
- `POST /v1/events` — 追記ログのみ。**オンライン学習も自動再学習も無い**。再学習は手動で `recommend-train`（または Compose の train ジョブ）を再実行する
- 未知のアイテムは 404。呼び出し側（commerce BFF / talent）はカタログ順やスキル重なりなどへ戻します

設計の詳細は [portfolio-plan](https://github.com/maeplego/portfolio-plan) の `portfolio-plan/recommend/docs/` です。

## ライセンスと利用条件

本リポジトリは **デモ・学習・社内評価用** です。現状品質に **保証はありません**。

- 許可: クローン、ローカル実行、学習、非本番の評価
- 別契約が必要: 本番運用、有償サービスへの組込み、再販・托管の提供

詳細は [LICENSE](./LICENSE) と [licensing.md](https://github.com/maeplego/portfolio-plan/blob/master/portfolio-plan/licensing.md) を参照してください。

