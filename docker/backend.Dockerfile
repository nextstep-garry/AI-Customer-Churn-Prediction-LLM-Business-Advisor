FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
COPY rag/requirements.txt rag/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r rag/requirements.txt
COPY backend backend
COPY models models
COPY rag rag
RUN useradd -m app && mkdir -p /app/data && chown -R app /app/data
USER app
ENV MODEL_DIR=/app/models DATA_DIR=/app/data HOST=0.0.0.0 PORT=5000
WORKDIR /app/backend
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/ready').status==200 else 1)"
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "300", "wsgi:app"]
