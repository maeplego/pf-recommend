This is a **fictional** MovieLens-shaped subset for CI and Compose.

- Schema matches GroupLens `ratings.csv` / `movies.csv` (`userId,movieId,rating,timestamp` and `movieId,title,genres`).
- Titles are invented. Do not treat this as the public MovieLens dataset.
- Unix cutoff used in tests and Compose: `1593561600` (2020-07-01 UTC). Train is strictly before that instant.

To train on public MovieLens latest-small (~100k ratings, ~1MB zip, not committed):

```powershell
python -m recommend_train --download ml-latest-small --namespace movies --out models --k 10
```
