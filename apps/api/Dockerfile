FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY packages packages
COPY apps/api apps/api
COPY apps/train apps/train
RUN pip install --no-cache-dir -e ".[api]"
ENV PYTHONUNBUFFERED=1
ENV RECOMMEND_HTTP_PORT=8098
EXPOSE 8098
CMD ["sh", "-c", "uvicorn recommend_api.main:app --host 0.0.0.0 --port ${RECOMMEND_HTTP_PORT:-8098}"]
