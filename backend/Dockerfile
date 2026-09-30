FROM python:3.12-slim

# xgboost's Linux wheel needs the OpenMP runtime at import time (the macOS
# equivalent of this project's `brew install libomp` requirement).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the repo root (not backend/) so this can also COPY model/ --
# see docker-compose.yml and render.yaml, which both set the context to ".".
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
