FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://package-mirror.liara.ir/repository/pypi/ -r requirements.txt


# Project files
COPY . .

# Run pipeline
ENTRYPOINT ["python", "-m", "src.data.ingest"]
