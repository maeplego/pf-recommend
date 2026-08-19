FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY packages packages
COPY apps/api apps/api
COPY apps/train apps/train
COPY testdata testdata
RUN pip install --no-cache-dir -e ".[train]"
ENV PYTHONUNBUFFERED=1
COPY apps/train/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
