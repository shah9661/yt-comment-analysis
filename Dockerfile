FROM python:3.11-slim

WORKDIR /app

COPY fastapi_app/requirements.txt fastapi_app/

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r fastapi_app/requirements.txt && \
    python -m nltk.downloader stopwords wordnet

COPY fastapi_app ./fastapi_app

EXPOSE 8000

CMD ["uvicorn", "fastapi_app.app:app", "--host", "0.0.0.0", "--port", "8000"]