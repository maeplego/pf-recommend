#!/bin/sh
set -eu
OUT="${RECOMMEND_MODELS_DIR:-/models}"
mkdir -p "$OUT"

python -m recommend_train \
  --ratings testdata/ml-tiny/ratings.csv \
  --movies testdata/ml-tiny/movies.csv \
  --namespace movies \
  --out "$OUT" \
  --k 5 \
  --cutoff 1593561600

python -m recommend_train \
  --events testdata/jobs-tiny/events.csv \
  --items testdata/jobs-tiny/items.csv \
  --namespace jobs \
  --out "$OUT" \
  --k 3 \
  --cutoff 1593561600
