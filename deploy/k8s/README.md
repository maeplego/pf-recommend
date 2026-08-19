# Kubernetes マニフェスト（P07 recommend）

推論 API（namespace `p07`）です。起動時に train が小さなフィクスチャを書き、API が読みます。このフォルダだけを apply しないでください。commerce overlay と talent overlay から参照します。

`recommend-api.localhost` で `GET /v1/recommend` と `GET /v1/similar-items` です。未知の item id は 404 です。呼び出し側はフォールバックします。
